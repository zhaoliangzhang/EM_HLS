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
            #pragma HLS PIPELINE
            local_probs.vec[i] = 1;
            probability:for (uint32_t d = 0; d < DIM; d++) {
                local_probs.vec[i] *= genhao_er_pai_fenzhiyi * vars[i][d];
                local_probs.vec[i] *= exp( (-0.5) * (sample[d] - means[i][d]) * (sample[d] - means[i][d]) * vars[i][d]);
            }
            local_probs.vec[i] *= priors[i];
        }
        probs.write(local_probs);
    }
}

void GetLabel_128(hls::stream<Vector<MAX_MODEL_NUM, PROB> > &probs, hls::stream<Vector<128, PROB> > &dis_128, hls::stream<Vector<128, uint32_t> > &label_128) {
    for(uint32_t n=0; n<DATA_NUM; n++) {
        Vector<MAX_MODEL_NUM, PROB> local_probs;
        Vector<128, PROB> local_dis_128;
        Vector<128, uint32_t> local_label_128;
        probs.read(local_probs);
        for(uint32_t i=0; i<128; i++){
            if(local_probs.vec[i] > local_probs.vec[i+128]){
                local_dis_128.vec[i] = local_probs.vec[i];
                local_label_128.vec[i] = i;
            } else {
                local_dis_128.vec[i] = local_probs.vec[i+128];
                local_label_128.vec[i] = i+128;
            }
        }
        dis_128.write(local_dis_128);
        label_128.write(local_label_128);
    }
}

template<uint32_t size1, uint32_t size2>
void GetLabel(hls::stream<Vector<size1, PROB> > &dis1, hls::stream<Vector<size1, uint32_t> > &label1, hls::stream<Vector<size2, PROB> > &dis2, hls::stream<Vector<size2, uint32_t> > &label2) {
    for(uint32_t n=0; n<DATA_NUM; n++) {
        Vector<size1, PROB> local_dis1;
        Vector<size1, uint32_t> local_label1;
        dis1.read(local_dis1);
        label1.read(local_label1);
        Vector<size2, PROB> local_dis2;
        Vector<size2, uint32_t> local_label2;
        for(uint32_t i=0; i<size2; i++) {
            if(local_dis1.vec[i] > local_dis1.vec[i+size2]) {
                local_dis2.vec[i] = local_dis1.vec[i];
                local_label2.vec[i] = local_label1.vec[i];
            } else {
                local_dis2.vec[i] = local_dis1.vec[i+size2];
                local_label2.vec[i] = local_label1.vec[i+size2];
            }
        }
        dis2.write(local_dis2);
        label2.write(local_label2);
    }
}

void GetLabel_1(hls::stream<Vector<2, PROB> > &dis_2, hls::stream<Vector<2, uint32_t> > &label_2, hls::stream<Vector<MAX_MODEL_NUM, PROB> > &P) {
    for(uint32_t n=0; n<DATA_NUM; n++) {
        Vector<2, PROB> local_dis_2;
        Vector<2, uint32_t> local_label_2;
        dis_2.read(local_dis_2);
        label_2.read(local_label_2);

        uint32_t p;

        Vector<MAX_MODEL_NUM, PROB> local_P;
        if(local_dis_2.vec[0]>local_dis_2.vec[1]) {
            p = local_label_2.vec[0];
        } else {
            p = local_label_2.vec[1];
        }
        for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
            if(i==p) {
                local_P.vec[i] = 1.0;
            } else {
                local_P.vec[i] = 0.0;
            }
        }
        P.write(local_P);
    }
}

void AccumProb(hls::stream<Vector<MAX_MODEL_NUM, PROB> > &probs, hls::stream<Vector<MAX_MODEL_NUM, PROB> > &P) {
    Vector<MAX_MODEL_NUM, PROB> local_probs;
    #pragma HLS ARRAY_PARTITION variable=local_probs block factor=16 dim=1

    Accum1:for(uint32_t n=0; n<DATA_NUM; n++) {
        PROB sum = 0;
        probs.read(local_probs);
        Accum2:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
            sum += local_probs.vec[i];
        }
        Accum3:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
            local_probs.vec[i] /= sum;
        }
        P.write(local_probs);
    }
}

void ProcessProb(hls::stream<Vector<MAX_MODEL_NUM, PROB> > &probs, hls::stream<Vector<MAX_MODEL_NUM, PROB> > &P, ap_uint<1> func) {
    if(func){
        AccumProb(probs, P);
    } else {
        hls::stream<Vector<128, PROB> > dis_128;
        #pragma HLS STREAM variable=dis_128 depth=16
        hls::stream<Vector<64, PROB> > dis_64;
        #pragma HLS STREAM variable=dis_64 depth=16
        hls::stream<Vector<32, PROB> > dis_32;
        #pragma HLS STREAM variable=dis_32 depth=16
        hls::stream<Vector<16, PROB> > dis_16;
        #pragma HLS STREAM variable=dis_16 depth=16
        hls::stream<Vector<8, PROB> > dis_8;
        #pragma HLS STREAM variable=dis_8 depth=16
        hls::stream<Vector<4, PROB> > dis_4;
        #pragma HLS STREAM variable=dis_4 depth=16
        hls::stream<Vector<2, PROB> > dis_2;
        #pragma HLS STREAM variable=dis_2 depth=16
        hls::stream<Vector<128, uint32_t> > label_128;
        #pragma HLS STREAM variable=label_128 depth=16
        hls::stream<Vector<64, uint32_t> > label_64;
        #pragma HLS STREAM variable=label_64 depth=16
        hls::stream<Vector<32, uint32_t> > label_32;
        #pragma HLS STREAM variable=label_32 depth=16
        hls::stream<Vector<16, uint32_t> > label_16;
        #pragma HLS STREAM variable=label_16 depth=16
        hls::stream<Vector<8, uint32_t> > label_8;
        #pragma HLS STREAM variable=label_8 depth=16
        hls::stream<Vector<4, uint32_t> > label_4;
        #pragma HLS STREAM variable=label_4 depth=16
        hls::stream<Vector<2, uint32_t> > label_2;
        #pragma HLS STREAM variable=label_2 depth=16
        
        GetLabel_128(probs, dis_128, label_128);
        GetLabel<128, 64>(dis_128, label_128, dis_64, label_64);
        GetLabel<64, 32>(dis_64, label_64, dis_32, label_32);
        GetLabel<32, 16>(dis_32, label_32, dis_16, label_16);
        GetLabel<16, 8>(dis_16, label_16, dis_8, label_8);
        GetLabel<8, 4>(dis_8, label_8, dis_4, label_4);
        GetLabel<4, 2>(dis_4, label_4, dis_2, label_2);
        GetLabel_1(dis_2, label_2, P);
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

    VARS next_vars[MAX_MODEL_NUM][DIM];
    #pragma HLS ARRAY_PARTITION variable=next_vars block factor=16 dim=1

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