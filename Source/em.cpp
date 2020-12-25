#include "em.h"
#include "hls_math.h"

void GetData(hls::stream<ap_uint<32> > &mm2s, hls::stream<DATA> &data, hls::stream<DATA> &data2) {
    for(uint32_t i=0; i<DATA_NUM; i++) {
        //#pragma HLS PIPELINE
        ap_uint<32> tmp;
        DATA convert;
        for(uint32_t j=0; j<3; j++){
            mm2s.read(tmp);
            convert = *(float *)(&tmp);
            data.write(convert);
            data2.write(convert);
        }
    }
}

void CalProb(hls::stream<DATA> &data, hls::stream<PROB> prob[MAX_MODEL_NUM], PRIOR priors[MAX_MODEL_NUM], MEANS means[MAX_MODEL_NUM][DIM], VARS vars[MAX_MODEL_NUM][DIM]){
    CalPROBF:for(uint32_t n=0; n<DATA_NUM; n++) {
        //#pragma HLS PIPELINE
        DATA sample[3];
        for(uint32_t i=0; i<3; i++){
            data.read(sample[i]);
        }
        CalPROB:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
            #pragma HLS PIPELINE
            PROB tmp_prob = 1;
            probability:for (uint32_t d = 0; d < DIM; d++) {
                tmp_prob *= genhao_er_pai_fenzhiyi * vars[i][d];
                tmp_prob *= exp( (-0.5) * (sample[d] - means[i][d]) * (sample[d] - means[i][d]) * vars[i][d]);
            }
            tmp_prob *= priors[i];
            prob[i].write(tmp_prob);
        }
    }
}

void GetLabel_128(hls::stream<PROB> dis[MAX_MODEL_NUM], hls::stream<PROB> local_dis_128[128], hls::stream<uint32_t> label_128[128]) {
    for(uint32_t n=0; n<DATA_NUM; n++) {
        for(uint32_t i=0; i<128; i++){
            PROB tmp1, tmp2;
            dis[i].read(tmp1);
            dis[i+128].read(tmp2);
            if(tmp1 > tmp2){
                local_dis_128[i].write(tmp1);
                label_128[i].write(i);
            } else {
                local_dis_128[i].write(tmp2);
                label_128[i].write(i+128);
            }
        }
    }
}

void GetLabel_64(hls::stream<PROB> local_dis_128[128], hls::stream<uint32_t> label_128[128], hls::stream<PROB> local_dis_64[64], hls::stream<uint32_t> label_64[64]) {
    for(uint32_t n=0; n<DATA_NUM; n++) {
        for(uint32_t i=0; i<64; i++) {
            PROB tmp, tmp1;
            uint32_t lab, lab1;
            local_dis_128[i].read(tmp);
            local_dis_128[i+64].read(tmp1);
            label_128[i].read(lab);
            label_128[i+64].read(lab1);
            if(tmp > tmp1) {
                local_dis_64[i].write(tmp);
                label_64[i].write(lab);
            } else {
                local_dis_64[i].write(tmp1);
                label_64[i].write(lab1);
            }
        }
    }
}

void GetLabel_32(hls::stream<PROB> local_dis_64[64], hls::stream<uint32_t> label_64[64], hls::stream<PROB> local_dis_32[32], hls::stream<uint32_t> label_32[32]) {
    for(uint32_t n=0; n<DATA_NUM; n++) {
        for(uint32_t i=0; i<32; i++) {
            PROB tmp, tmp1;
            uint32_t lab, lab1;
            local_dis_64[i].read(tmp);
            local_dis_64[i+32].read(tmp1);
            label_64[i].read(lab);
            label_64[i+32].read(lab1);
            if(tmp > tmp1) {
                local_dis_32[i].write(tmp);
                label_32[i].write(lab);
            } else {
                local_dis_32[i].write(tmp1);
                label_32[i].write(lab1);
            }
        }
    }
}

void GetLabel_16(hls::stream<PROB> local_dis_32[32], hls::stream<uint32_t> label_32[32], hls::stream<PROB> local_dis_16[16], hls::stream<uint32_t> label_16[16]) {
    for(uint32_t n=0; n<DATA_NUM; n++) {
        for(uint32_t i=0; i<16; i++) {
            PROB tmp, tmp1;
            uint32_t lab, lab1;
            local_dis_32[i].read(tmp);
            local_dis_32[i+16].read(tmp1);
            label_32[i].read(lab);
            label_32[i+16].read(lab1);
            if(tmp > tmp1) {
                local_dis_16[i].write(tmp);
                label_16[i].write(lab);
            } else {
                local_dis_16[i].write(tmp1);
                label_16[i].write(lab1);
            }
        }
    }
}

void GetLabel_8(hls::stream<PROB> local_dis_16[16], hls::stream<uint32_t> label_16[16], hls::stream<PROB> local_dis_8[8], hls::stream<uint32_t> label_8[8]) {
    for(uint32_t n=0; n<DATA_NUM; n++) {
        for(uint32_t i=0; i<8; i++) {
            PROB tmp, tmp1;
            uint32_t lab, lab1;
            local_dis_16[i].read(tmp);
            local_dis_16[i+8].read(tmp1);
            label_16[i].read(lab);
            label_16[i+8].read(lab1);
            if(tmp > tmp1) {
                local_dis_8[i].write(tmp);
                label_8[i].write(lab);
            } else {
                local_dis_8[i].write(tmp1);
                label_8[i].write(lab1);
            }
        }
    }
}

void GetLabel_4(hls::stream<PROB> local_dis_8[8], hls::stream<uint32_t> label_8[8], hls::stream<PROB> local_dis_4[4], hls::stream<uint32_t> label_4[4]) {
    for(uint32_t n=0; n<DATA_NUM; n++) {
        for(uint32_t i=0; i<4; i++) {
            PROB tmp, tmp1;
            uint32_t lab, lab1;
            local_dis_8[i].read(tmp);
            local_dis_8[i+4].read(tmp1);
            label_8[i].read(lab);
            label_8[i+4].read(lab1);
            if(tmp > tmp1) {
                local_dis_4[i].write(tmp);
                label_4[i].write(lab);
            } else {
                local_dis_4[i].write(tmp1);
                label_4[i].write(lab1);
            }
        }
    }
}

void GetLabel_2(hls::stream<PROB> local_dis_4[4], hls::stream<uint32_t> label_4[4], hls::stream<PROB> local_dis_2[2], hls::stream<uint32_t> label_2[2]) {
    for(uint32_t n=0; n<DATA_NUM; n++) {
        for(uint32_t i=0; i<2; i++) {
            PROB tmp, tmp1;
            uint32_t lab, lab1;
            local_dis_4[i].read(tmp);
            local_dis_4[i+2].read(tmp1);
            label_4[i].read(lab);
            label_4[i+2].read(lab1);
            if(tmp > tmp1) {
                local_dis_2[i].write(tmp);
                label_2[i].write(lab);
            } else {
                local_dis_2[i].write(tmp1);
                label_2[i].write(lab1);
            }
        }
    }
}

void GetLabel_1(hls::stream<PROB> local_dis_2[2], hls::stream<uint32_t> label_2[2], hls::stream<PROB> P[MAX_MODEL_NUM]) {
    for(uint32_t n=0; n<DATA_NUM; n++) {
        uint32_t p;
        PROB local_tmp, local_tmp1;
        local_dis_2[0].read(local_tmp);
        local_dis_2[1].read(local_tmp1);
        if(local_tmp>local_tmp1) {
            label_2[0].read(p);
        } else {
            label_2[1].read(p);
        }
        for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
            if(i==p) {
                P[i].write(1);
            } else {
                P[i].write(0);
            }
        }
    }
}

void AccumProb(hls::stream<PROB> prob[MAX_MODEL_NUM], hls::stream<PROB> P[MAX_MODEL_NUM]) {
    PROB local_prob[MAX_MODEL_NUM];
    #pragma HLS ARRAY_PARTITION variable=local_prob block factor=16 dim=1

    Accum1:for(uint32_t n=0; n<DATA_NUM; n++) {
        PROB sum = 0;
        Accum2:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
            prob[i].read(local_prob[i]);
            sum += local_prob[i];
        }
        Accum3:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
            local_prob[i] /= sum;
            P[i].write(local_prob[i]);
        }
    }
}

void ProcessProb(hls::stream<PROB> probs[MAX_MODEL_NUM], hls::stream<PROB> P[MAX_MODEL_NUM], ap_uint<1> func) {
    if(func){
        AccumProb(probs, P);
    } else {
        hls::stream<PROB> local_dis_128[128];
        #pragma HLS STREAM variable=local_dis_128 depth=2
        hls::stream<PROB> local_dis_64[64];
        #pragma HLS STREAM variable=local_dis_64 depth=2
        hls::stream<PROB> local_dis_32[32];
        #pragma HLS STREAM variable=local_dis_32 depth=2
        hls::stream<PROB> local_dis_16[16];
        #pragma HLS STREAM variable=local_dis_16 depth=2
        hls::stream<PROB> local_dis_8[8];
        #pragma HLS STREAM variable=local_dis_8 depth=2
        hls::stream<PROB> local_dis_4[4];
        #pragma HLS STREAM variable=local_dis_4 depth=2
        hls::stream<PROB> local_dis_2[2];
        #pragma HLS STREAM variable=local_dis_2 depth=2
        hls::stream<uint32_t> label_128[128];
        #pragma HLS STREAM variable=label_128 depth=2
        hls::stream<uint32_t> label_64[64];
        #pragma HLS STREAM variable=label_64 depth=2
        hls::stream<uint32_t> label_32[32];
        #pragma HLS STREAM variable=label_32 depth=2
        hls::stream<uint32_t> label_16[16];
        #pragma HLS STREAM variable=label_16 depth=2
        hls::stream<uint32_t> label_8[8];
        #pragma HLS STREAM variable=label_8 depth=2
        hls::stream<uint32_t> label_4[4];
        #pragma HLS STREAM variable=label_4 depth=2
        hls::stream<uint32_t> label_2[2];
        #pragma HLS STREAM variable=label_2 depth=2
        
        GetLabel_128(probs, local_dis_128, label_128);
        GetLabel_64(local_dis_128, label_128, local_dis_64, label_64);
        GetLabel_32(local_dis_64, label_64, local_dis_32, label_32);
        GetLabel_16(local_dis_32, label_32, local_dis_16, label_16);
        GetLabel_8(local_dis_16, label_16, local_dis_8, label_8);
        GetLabel_4(local_dis_8, label_8, local_dis_4, label_4);
        GetLabel_2(local_dis_4, label_4, local_dis_2, label_2);
        GetLabel_1(local_dis_2, label_2, P);
    }
}

void Update(hls::stream<PROB> P[MAX_MODEL_NUM], PRIOR next_priors[MAX_MODEL_NUM], MEANS next_means[MAX_MODEL_NUM][DIM], VARS next_vars[MAX_MODEL_NUM][DIM], ap_uint<9> count[MAX_MODEL_NUM], hls::stream<DATA> &data2, ap_uint<1> func){
    UpdateF:for(uint32_t i=0; i<DATA_NUM; i++) {
        DATA sample[3];
        Update1:for(uint32_t i=0; i<3; i++){
            data2.read(sample[i]);
        }

        Update2:for(uint32_t i=0; i<MAX_MODEL_NUM; i++){
        //#pragma HLS UNROLL factor=MAX_MODEL_NUM
            PROB tmp;
            P[i].read(tmp);
            if((func==0)&&(tmp==1)){
                count[i]++;
            }
            next_priors[i] += tmp;
            Update3:for(uint32_t d=0; d<DIM; d++) {
                #pragma HLS UNROLL factor=3
                next_means[i][d] += tmp*sample[d];
                next_vars[i][d] += tmp*sample[d]*sample[d];
            }
        }

    }
}

void EMCore(hls::stream<ap_uint<32> > &mm2s, 
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
    #pragma HLS STREAM variable=data2 depth=16

    hls::stream<PROB> probs[MAX_MODEL_NUM];
    #pragma HLS STREAM variable=probs depth=16
    hls::stream<PROB> P[MAX_MODEL_NUM];
    #pragma HLS STREAM variable=P depth=16

    GetData(mm2s, data, data2);
    CalProb(data, probs, priors, means, vars);
    ProcessProb(probs, P, function1);
    Update(P, next_priors, next_means, next_vars, count, data2, function2);
}


void  EM(hls::stream<ap_uint<32> > &mm2s,
PRIOR priors[MAX_MODEL_NUM],
MEANS means[MAX_MODEL_NUM][DIM],
VARS  vars[MAX_MODEL_NUM][DIM],
ap_uint<1> func) {

    /*PRIOR next_vars0[MAX_MODEL_NUM];
    #pragma HLS ARRAY_PARTITION variable=next_vars0 block factor=16 dim=1
    PRIOR next_vars1[MAX_MODEL_NUM];
    #pragma HLS ARRAY_PARTITION variable=next_vars1 block factor=16 dim=1
    PRIOR next_vars2[MAX_MODEL_NUM];
    #pragma HLS ARRAY_PARTITION variable=next_vars2 block factor=16 dim=1
    
    MEANS next_means0[MAX_MODEL_NUM];
    #pragma HLS ARRAY_PARTITION variable=next_means0 block factor=16 dim=1
    MEANS next_means1[MAX_MODEL_NUM];
    #pragma HLS ARRAY_PARTITION variable=next_means1 block factor=16 dim=1
    MEANS next_means2[MAX_MODEL_NUM];
    #pragma HLS ARRAY_PARTITION variable=next_means2 block factor=16 dim=1*/

    PRIOR next_priors[MAX_MODEL_NUM];
    #pragma HLS ARRAY_PARTITION variable=next_priors block factor=16 dim=1

    MEANS next_means[MAX_MODEL_NUM][DIM];
    #pragma HLS ARRAY_PARTITION variable=next_means block factor=16 dim=1

    VARS next_vars[MAX_MODEL_NUM][DIM];
    #pragma HLS ARRAY_PARTITION variable=next_vars block factor=16 dim=1

    ap_uint<9> count[MAX_MODEL_NUM];
    #pragma HLS ARRAY_PARTITION variable=count cyclic factor=16 dim=1


    /*resetnextmeans:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
        next_means0[i] = 0;
        next_means1[i] = 0;
        next_means2[i] = 0;
        count[i] = 0;
    }*/

    reset:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
        next_priors[i] = 0;
        next_means[i][0] = 0; next_means[i][1] = 0; next_means[i][1] = 0;
        next_vars[i][0] = 0; next_vars[i][1] = 0; next_vars[i][1] = 0;
        count[i] = 0;
    }

    EMCore(mm2s,
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