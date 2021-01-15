#include "em.h"
#include "hls_math.h"

void GetData(DATA _data[DATA_NUM*3], hls::stream<Vector<DIM, DATA> > &data, hls::stream<Vector<DIM, DATA> > &data2) {
    for(uint32_t i=0; i<DATA_NUM; i++) {
        Vector<DIM, DATA> tmp;
        for(uint32_t j=0; j<3; j++){
            tmp.vec[j] = _data[i*3+j];
        }
        data.write(tmp);
        data2.write(tmp);
    }
}

void CalProb(hls::stream<Vector<DIM, DATA> > &data, hls::stream<Vector<MAX_MODEL_NUM, PROB> > &probs, PRIOR priors[MAX_MODEL_NUM], MEANS means[MAX_MODEL_NUM][DIM], VARS vars[MAX_MODEL_NUM][DIM]){
    CalPROBF:for(uint32_t n=0; n<DATA_NUM; n++) {
        Vector<DIM, DATA> sample;
        data.read(sample);
    #pragma HLS DEPENDENCE variable=sample intra false
    #pragma HLS DEPENDENCE variable=sample inter false
        Vector<MAX_MODEL_NUM, PROB> local_probs;
        CalPROB:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
        #pragma HLS UNROLL factor=128
        local_probs.vec[i] = 1;
            probability:for (uint32_t d = 0; d < DIM; d++) {
                sample.vec[d] = (sample.vec[d] - means[i][d]) * vars[i][d];
                PROB tmp;
                if(sample.vec[d] >=0 && sample.vec[d] < datath1){
                    tmp = (PROB)-0.078125;
                    tmp *= sample.vec[d];
                    tmp += (PROB)0.40625;
                    local_probs.vec[i] *= tmp;
                } else if(sample.vec[d] >=datath1 && sample.vec[d] < datath2) {
                    tmp = (PROB)-0.21875;
                    tmp *= sample.vec[d];
                    tmp += (PROB)0.4609375;
                    local_probs.vec[i] *= tmp;
                } else if(sample.vec[d] >=datath2 && sample.vec[d] < datath3) {
                    tmp = (PROB)-0.0546875;
                    tmp *= sample.vec[d];
                    tmp += (PROB)0.1640625;
                    local_probs.vec[i] *= tmp;
                } else if(sample.vec[d] >=datath3 && sample.vec[d] < datath4) {
                    local_probs.vec[i] *= (PROB)0.00390625;
                } else if(sample.vec[d] >=datath4) {
                    local_probs.vec[i] *= (PROB)0.00001526;
                }
            }
            local_probs.vec[i] *= priors[i];
            local_probs.vec[i] *= vars[i][0];
            local_probs.vec[i] *= vars[i][1];
            local_probs.vec[i] *= vars[i][2];
        }
        probs.write(local_probs);
    }
}

void AccumProb(hls::stream<Vector<MAX_MODEL_NUM, PROB> > &probs, hls::stream<Vector<MAX_MODEL_NUM, RESP> > &resp) {
    Vector<MAX_MODEL_NUM, PROB> local_probs;
    #pragma HLS ARRAY_PARTITION variable=local_probs block factor=16 dim=1

    Vector<MAX_MODEL_NUM, RESP> local_resp;
    #pragma HLS ARRAY_PARTITION variable=local_resp block factor=16 dim=1

    Accum1:for(uint32_t n=0; n<DATA_NUM; n++) {
        PROB sum = 0;

        probs.read(local_probs);
        Accum2:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
        #pragma HLS UNROLL factor=128
            sum += local_probs.vec[i];
        }

        ap_ufixed<16, 16, AP_RND, AP_SAT> inv_sum = 1/sum;
        Accum3:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
        #pragma HLS UNROLL factor=128
            local_resp.vec[i] = local_probs.vec[i]*inv_sum;
        }
        resp.write(local_resp);
    }
}

void GetMax(hls::stream<Vector<MAX_MODEL_NUM, PROB> > &probs, hls::stream<Vector<MAX_MODEL_NUM, RESP> > &resp){
    Vector<MAX_MODEL_NUM, PROB> local_probs;
    #pragma HLS ARRAY_PARTITION variable=local_probs block factor=16 dim=1

    Vector<MAX_MODEL_NUM, RESP> local_resp;
    #pragma HLS ARRAY_PARTITION variable=local_resp block factor=16 dim=1

    GetMaxF:for(uint32_t n=0; n<DATA_NUM; n++) {
        PROB max = 1.0;
        uint32_t p = 0;
        probs.read(local_probs);
        GetMax1:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
		#pragma HLS UNROLL factor=128
            if(max < local_probs.vec[i]) {
                max = local_probs.vec[i];
                p = i;
            }
        }
        GetMax2:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
		#pragma HLS UNROLL factor=128
            if(i==p){
                local_resp.vec[i] = 1.0;
            } else {
                local_resp.vec[i] = 0.0;
            }
        }
        resp.write(local_resp);
    }

}

void ProcessProb(hls::stream<Vector<MAX_MODEL_NUM, PROB> > &probs, hls::stream<Vector<MAX_MODEL_NUM, RESP> > &resp, ap_uint<1> func) {
    if(func){
        AccumProb(probs, resp);
    } else {
        GetMax(probs, resp);
    }
}

void Update(hls::stream<Vector<MAX_MODEL_NUM, RESP> > &resp, PRIOR next_priors[MAX_MODEL_NUM], MEANS next_means[MAX_MODEL_NUM][DIM], VARS next_vars[MAX_MODEL_NUM][DIM], ap_uint<9> count[MAX_MODEL_NUM], hls::stream<Vector<DIM, DATA> > &data2, ap_uint<1> func){
//void Update(hls::stream<Vector<MAX_MODEL_NUM, RESP> > &resp, PRIOR next_priors[MAX_MODEL_NUM], MEANS next_means[MAX_MODEL_NUM][DIM], VARS next_vars[MAX_MODEL_NUM][DIM], ap_uint<9> count[MAX_MODEL_NUM], hls::stream<Vector<DIM, DATA> > &data2){
    UpdateF:for(uint32_t i=0; i<DATA_NUM; i++) {
        Vector<DIM, DATA> sample;
        data2.read(sample);

        Vector<MAX_MODEL_NUM, RESP> local_resp;
        resp.read(local_resp);

        #pragma HLS DEPENDENCE variable=count intra false
        
        Update2:for(uint32_t i=0; i<MAX_MODEL_NUM; i++){
        #pragma HLS UNROLL factor=128
            count[i] += local_resp.vec[i];
            next_priors[i] += local_resp.vec[i];
            Update3:for(uint32_t d=0; d<DIM; d++) {
                #pragma HLS UNROLL factor=3
                MEANS tmp1 = local_resp.vec[i]*sample.vec[d];
                VARS tmp2 = local_resp.vec[i]*sample.vec[d];
                tmp2 *= sample.vec[d];
                next_means[i][d] += tmp1;
                next_vars[i][d] += tmp2;
            }
        }
    }
}

void EMCore(DATA _data[DATA_NUM*3], 
    PRIOR priors[MAX_MODEL_NUM],
    MEANS means[MAX_MODEL_NUM][DIM],
    VARS vars[MAX_MODEL_NUM][DIM],
    PRIOR next_priors[MAX_MODEL_NUM],
    MEANS next_means[MAX_MODEL_NUM][DIM],
    VARS next_vars[MAX_MODEL_NUM][DIM],
    ap_uint<9> count[MAX_MODEL_NUM],
    ap_uint<1> func)
{
    #pragma HLS dataflow

    ap_uint<1> function1 = func;
    ap_uint<1> function2 = func;

    hls::stream<Vector<DIM, DATA> > data;
    #pragma HLS STREAM variable=data depth=4
    hls::stream<Vector<DIM, DATA> > data2;
    #pragma HLS STREAM variable=data2 depth=8

    hls::stream<Vector<MAX_MODEL_NUM, PROB> > probs;
    #pragma HLS STREAM variable=probs depth=4
    hls::stream<Vector<MAX_MODEL_NUM, RESP> > resp;
    #pragma HLS STREAM variable=resp depth=4

    GetData(_data, data, data2);
    CalProb(data, probs, priors, means, vars);
    ProcessProb(probs, resp, function1);
    Update(resp, next_priors, next_means, next_vars, count, data2, function2);
    //Update(resp, next_priors, next_means, next_vars, count, data2);
}


void  EM(DATA _data[DATA_NUM*3],
PRIOR priors[MAX_MODEL_NUM],
MEANS means[MAX_MODEL_NUM][DIM],
VARS  vars[MAX_MODEL_NUM][DIM],
ap_uint<1> func) {

    PRIOR next_priors[MAX_MODEL_NUM];
    #pragma HLS ARRAY_PARTITION variable=next_priors block factor=8 dim=1

    MEANS next_means[MAX_MODEL_NUM][DIM];
    #pragma HLS ARRAY_PARTITION variable=next_means block factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=next_means block factor=3 dim=2

    VARS next_vars[MAX_MODEL_NUM][DIM];
    #pragma HLS ARRAY_PARTITION variable=next_vars block factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=next_vars block factor=3 dim=2

    ap_uint<9> count[MAX_MODEL_NUM];
    #pragma HLS ARRAY_PARTITION variable=count cyclic factor=128 dim=1

    reset:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
        next_priors[i] = 0;
        next_means[i][0] = 0; next_means[i][1] = 0; next_means[i][1] = 0;
        next_vars[i][0] = 0; next_vars[i][1] = 0; next_vars[i][1] = 0;
        count[i] = 0;
    }

    EMCore(_data,
    priors, means, vars, 
    next_priors, next_means, next_vars,
    count, func);

    Update_means:for(uint32_t i=0; i<MAX_MODEL_NUM; i++){
        #pragma HLS PIPELINE II=1
        if(func){
            priors[i] = next_priors[i];

            means[i][0] = next_means[i][0];
            means[i][1] = next_means[i][1];
            means[i][2] = next_means[i][2];

            vars[i][0] = next_vars[i][0];
            vars[i][1] = next_vars[i][1];
            vars[i][2] = next_vars[i][2];
        } else {
            if(count[i]>0){
                means[i][0] = next_means[i][0]/count[i];
                means[i][1] = next_means[i][1]/count[i];
                means[i][2] = next_means[i][2]/count[i];
            }
        }
    }
}
