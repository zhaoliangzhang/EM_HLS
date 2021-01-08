#include "em.h"
#include "hls_math.h"

void GetData(DATA _data[MAX_MODEL_NUM*3], hls::stream<DATA> &data, hls::stream<DATA> &data2) {
    for(uint32_t i=0; i<DATA_NUM; i++) {
        for(uint32_t j=0; j<3; j++){
            DATA tmp;
            tmp = _data[i*3+j];
            data.write(tmp);
            data2.write(tmp);
        }
    }
}

void CalProb(hls::stream<DATA> &data, hls::stream<Vector<MAX_MODEL_NUM, PROB> > &probs, PRIOR priors[MAX_MODEL_NUM], MEANS means[MAX_MODEL_NUM][DIM], VARS vars[MAX_MODEL_NUM][DIM]){
    CalPROBF:for(uint32_t n=0; n<DATA_NUM; n++) {
        DATA sample[3];
        for(uint32_t i=0; i<3; i++){
            data.read(sample[i]);
        }
        Vector<MAX_MODEL_NUM, PROB> local_probs;
        CalPROB:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
            //#pragma HLS PIPELINE off
            #pragma UNROLL factor=MAX_MODEL_NUM
            local_probs.vec[i] = 1;
            probability:for (uint32_t d = 0; d < DIM; d++) {
                #pragma HLS UNROLL factor=3
                local_probs.vec[i] *= genhao_er_pai_fenzhiyi * vars[i][d];
                local_probs.vec[i] *= (PROB)(-0.5) * (sample[d] - means[i][d]) * (sample[d] - means[i][d]) * vars[i][d];
            }
            local_probs.vec[i] *= priors[i];
        }
        probs.write(local_probs);
    }
}

void AccumProb(hls::stream<Vector<MAX_MODEL_NUM, PROB> > &probs, hls::stream<Vector<MAX_MODEL_NUM, PROB> > &P) {
    Vector<MAX_MODEL_NUM, PROB> local_probs;
    #pragma HLS ARRAY_PARTITION variable=local_probs block factor=16 dim=1

    Accum1:for(uint32_t n=0; n<DATA_NUM; n++) {
        PROB sum = 0;
        #pragma HLS dependence variable=sum inter false

        probs.read(local_probs);
        Accum2:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
        //#pragma HLS UNROLL factor=MAX_MODEL_NUM
            sum += local_probs.vec[i];
        }
        Accum3:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
        //#pragma HLS UNROLL factor=MAX_MODEL_NUM
            local_probs.vec[i] /= sum;
        }
        P.write(local_probs);
    }
}

void GetMax(hls::stream<Vector<MAX_MODEL_NUM, PROB> > &probs, hls::stream<Vector<MAX_MODEL_NUM, PROB> > &P){
    Vector<MAX_MODEL_NUM, PROB> local_probs;
    #pragma HLS ARRAY_PARTITION variable=local_probs block factor=16 dim=1

    Vector<MAX_MODEL_NUM, PROB> local_P;
    #pragma HLS ARRAY_PARTITION variable=local_P block factor=16 dim=1

    GetMaxF:for(uint32_t n=0; n<DATA_NUM; n++) {
        PROB max = 1.0;
        uint32_t p = 0;
        probs.read(local_probs);
        GetMax1:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
            if(max < local_probs.vec[i]) {
                max = local_probs.vec[i];
                p = i;
            }
        }
        GetMax2:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
            if(i==p){
                local_P.vec[i] = 1.0;
            } else {
                local_P.vec[i] = 0.0;
            }
        }
        P.write(local_P);
    }

}

void ProcessProb(hls::stream<Vector<MAX_MODEL_NUM, PROB> > &probs, hls::stream<Vector<MAX_MODEL_NUM, PROB> > &P, ap_uint<1> func) {
    if(func){
        AccumProb(probs, P);
    } else {
        GetMax(probs, P);
        //AccumProb(probs, P);
    }
}

void Update(hls::stream<Vector<MAX_MODEL_NUM, PROB> > &P, PRIOR next_priors[MAX_MODEL_NUM], MEANS next_means[MAX_MODEL_NUM][DIM], VARS next_vars[MAX_MODEL_NUM][DIM], ap_uint<9> count[MAX_MODEL_NUM], hls::stream<DATA> &data2, ap_uint<1> func){
    UpdateF:for(uint32_t i=0; i<DATA_NUM; i++) {
        DATA sample[3];
        Update1:for(uint32_t i=0; i<3; i++){
            data2.read(sample[i]);
        }

        Vector<MAX_MODEL_NUM, PROB> local_P;
        P.read(local_P);
        Update2:for(uint32_t i=0; i<MAX_MODEL_NUM; i++){
        #pragma UNROLL factor=MAX_MODEL_NUM
            if((func==0)&&(local_P.vec[i]==1)){
                count[i]++;
            }
            local_P.vec[i] += next_priors[i];
            Update3:for(uint32_t d=0; d<DIM; d++) {
                #pragma HLS UNROLL factor=3
                next_means[i][d] += local_P.vec[i]*sample[d];
                next_vars[i][d] += local_P.vec[i]*sample[d]*sample[d];
            }
        }

    }
}

void EMCore(DATA _data[MAX_MODEL_NUM*3], 
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

    hls::stream<DATA> data;
    #pragma HLS STREAM variable=data depth=16
    hls::stream<DATA> data2;
    #pragma HLS STREAM variable=data2 depth=64

    hls::stream<Vector<MAX_MODEL_NUM, PROB> > probs;
    #pragma HLS STREAM variable=probs depth=16
    hls::stream<Vector<MAX_MODEL_NUM, PROB> > P;
    #pragma HLS STREAM variable=P depth=16

    GetData(_data, data, data2);
    CalProb(data, probs, priors, means, vars);
    ProcessProb(probs, P, function1);
    Update(P, next_priors, next_means, next_vars, count, data2, function2);
}


void  EM(DATA _data[MAX_MODEL_NUM*3],
PRIOR priors[MAX_MODEL_NUM],
MEANS means[MAX_MODEL_NUM][DIM],
VARS  vars[MAX_MODEL_NUM][DIM],
ap_uint<1> func) {

    PRIOR next_priors[MAX_MODEL_NUM];
    #pragma HLS ARRAY_PARTITION variable=next_priors block factor=16 dim=1

    MEANS next_means[MAX_MODEL_NUM][DIM];
    #pragma HLS ARRAY_PARTITION variable=next_means block factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=next_means block factor=3 dim=2

    VARS next_vars[MAX_MODEL_NUM][DIM];
    #pragma HLS ARRAY_PARTITION variable=next_vars block factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=next_vars block factor=3 dim=2

    ap_uint<9> count[MAX_MODEL_NUM];
    #pragma HLS ARRAY_PARTITION variable=count cyclic factor=16 dim=1

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