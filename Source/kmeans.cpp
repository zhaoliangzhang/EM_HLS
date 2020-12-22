#include "kmeans.h"
#include "hls_math.h"

void GetData(hls::stream<ap_uint<32> > &mm2s, hls::stream<DIS> &data, hls::stream<DIS> &data2) {
    for(uint32_t i=0; i<DATA_NUM; i++) {
        //#pragma HLS PIPELINE
        ap_uint<32> tmp;
        DIS convert;
        for(uint32_t j=0; j<3; j++){
            mm2s.read(tmp);
            convert = *(float *)(&tmp);
            data.write(convert);
            data2.write(convert);
        }
    }
}

void CalDis(hls::stream<DIS> &data, hls::stream<DIS> dis[MAX_MODEL_NUM], MEANS k_means[MAX_MODEL_NUM][DIM]){
    CalDISF:for(uint32_t n=0; n<DATA_NUM; n++) {
        //#pragma HLS PIPELINE
        DIS sample[3];
        for(uint32_t i=0; i<3; i++){
            data.read(sample[i]);
        }
        CalDIS:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
            #pragma HLS PIPELINE
            DIS tmp_dis = 0;
            DIS axe_dis = 0;
            distance:for (uint32_t d = 0; d < DIM; d++) {
                axe_dis = (sample[d]-k_means[i][d]);
                tmp_dis = fabs(axe_dis);
            }
            dis[i].write(tmp_dis);
        }
    }
}

void GetLabel_128(hls::stream<DIS> dis[MAX_MODEL_NUM], hls::stream<DIS> local_dis_128[128], hls::stream<uint32_t> label_128[128]) {
    for(uint32_t n=0; n<DATA_NUM; n++) {
        for(uint32_t i=0; i<128; i++){
            DIS tmp1, tmp2;
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

void GetLabel_64(hls::stream<DIS> local_dis_128[128], hls::stream<uint32_t> label_128[128], hls::stream<DIS> local_dis_64[64], hls::stream<uint32_t> label_64[64]) {
    for(uint32_t n=0; n<DATA_NUM; n++) {
        for(uint32_t i=0; i<64; i++) {
            DIS tmp, tmp1;
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

void GetLabel_32(hls::stream<DIS> local_dis_64[64], hls::stream<uint32_t> label_64[64], hls::stream<DIS> local_dis_32[32], hls::stream<uint32_t> label_32[32]) {
    for(uint32_t n=0; n<DATA_NUM; n++) {
        for(uint32_t i=0; i<32; i++) {
            DIS tmp, tmp1;
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

void GetLabel_16(hls::stream<DIS> local_dis_32[32], hls::stream<uint32_t> label_32[32], hls::stream<DIS> local_dis_16[16], hls::stream<uint32_t> label_16[16]) {
    for(uint32_t n=0; n<DATA_NUM; n++) {
        for(uint32_t i=0; i<16; i++) {
            DIS tmp, tmp1;
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

void GetLabel_8(hls::stream<DIS> local_dis_16[16], hls::stream<uint32_t> label_16[16], hls::stream<DIS> local_dis_8[8], hls::stream<uint32_t> label_8[8]) {
    for(uint32_t n=0; n<DATA_NUM; n++) {
        for(uint32_t i=0; i<8; i++) {
            DIS tmp, tmp1;
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

void GetLabel_4(hls::stream<DIS> local_dis_8[8], hls::stream<uint32_t> label_8[8], hls::stream<DIS> local_dis_4[4], hls::stream<uint32_t> label_4[4]) {
    for(uint32_t n=0; n<DATA_NUM; n++) {
        for(uint32_t i=0; i<4; i++) {
            DIS tmp, tmp1;
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

void GetLabel_2(hls::stream<DIS> local_dis_4[4], hls::stream<uint32_t> label_4[4], hls::stream<DIS> local_dis_2[2], hls::stream<uint32_t> label_2[2]) {
    for(uint32_t n=0; n<DATA_NUM; n++) {
        for(uint32_t i=0; i<2; i++) {
            DIS tmp, tmp1;
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

void GetLabel_1(hls::stream<DIS> local_dis_2[2], hls::stream<uint32_t> label_2[2], hls::stream<ap_uint<1> > label[MAX_MODEL_NUM]) {
    for(uint32_t n=0; n<DATA_NUM; n++) {
        uint32_t p;
        DIS local_tmp, local_tmp1;
        local_dis_2[0].read(local_tmp);
        local_dis_2[1].read(local_tmp1);
        if(local_tmp>local_tmp1) {
            label_2[0].read(p);
        } else {
            label_2[1].read(p);
        }
        for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
            if(i==p) {
                label[i].write(1);
            } else {
                label[i].write(0);
            }
        }
    }
}

void Update(hls::stream<ap_uint<1> > label[MAX_MODEL_NUM], MEANS next_means0[MAX_MODEL_NUM], MEANS next_means1[MAX_MODEL_NUM], MEANS next_means2[MAX_MODEL_NUM], ap_uint<8> count[MAX_MODEL_NUM], hls::stream<DIS> &data2){
    UpdateF:for(uint32_t i=0; i<DATA_NUM; i++) {
        DIS sample[3];
        for(uint32_t i=0; i<3; i++){
            data2.read(sample[i]);
        }
        update:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
        #pragma HLS PIPELINE
            ap_uint<1> lab;
            label[i].read(lab);
            if(lab){
                count[i]++;
                next_means0[i] += sample[0];
                next_means1[i] += sample[1];
                next_means2[i] += sample[2];
            }
        }
    }
}

void KMeansCore(hls::stream<ap_uint<32> > &mm2s, MEANS k_means[MAX_MODEL_NUM][DIM], 
    MEANS next_means0[MAX_MODEL_NUM], 
    MEANS next_means1[MAX_MODEL_NUM],
    MEANS next_means2[MAX_MODEL_NUM],
    ap_uint<8> count[MAX_MODEL_NUM])
{
    #pragma HLS dataflow

    hls::stream<DIS> data;
    #pragma HLS STREAM variable=data depth=16
    hls::stream<DIS> data2;
    #pragma HLS STREAM variable=data2 depth=16

    hls::stream<DIS> dis[MAX_MODEL_NUM];
    #pragma HLS STREAM variable=dis depth=16
    hls::stream<ap_uint<1> > label[MAX_MODEL_NUM];
    #pragma HLS STREAM variable=label depth=16

    hls::stream<DIS> local_dis_128[128];
    #pragma HLS STREAM variable=local_dis_128 depth=2
    hls::stream<DIS> local_dis_64[64];
    #pragma HLS STREAM variable=local_dis_64 depth=2
    hls::stream<DIS> local_dis_32[32];
    #pragma HLS STREAM variable=local_dis_32 depth=2
    hls::stream<DIS> local_dis_16[16];
    #pragma HLS STREAM variable=local_dis_16 depth=2
    hls::stream<DIS> local_dis_8[8];
    #pragma HLS STREAM variable=local_dis_8 depth=2
    hls::stream<DIS> local_dis_4[4];
    #pragma HLS STREAM variable=local_dis_4 depth=2
    hls::stream<DIS> local_dis_2[2];
    #pragma HLS STREAM variable=local_dis_2 depth=128
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
    #pragma HLS STREAM variable=label_2 depth=128

    GetData(mm2s, data, data2);
    CalDis(data, dis, k_means);
    GetLabel_128(dis, local_dis_128, label_128);
    GetLabel_64(local_dis_128, label_128, local_dis_64, label_64);
    GetLabel_32(local_dis_64, label_64, local_dis_32, label_32);
    GetLabel_16(local_dis_32, label_32, local_dis_16, label_16);
    GetLabel_8(local_dis_16, label_16, local_dis_8, label_8);
    GetLabel_4(local_dis_8, label_8, local_dis_4, label_4);
    GetLabel_2(local_dis_4, label_4, local_dis_2, label_2);
    GetLabel_1(local_dis_2, label_2, label);
    Update(label, next_means0, next_means1, next_means2, count, data2);
}


void  KMeans(hls::stream<ap_uint<32> > &mm2s,
MEANS k_means[MAX_MODEL_NUM][DIM]) {

    MEANS next_means0[MAX_MODEL_NUM];
    #pragma HLS ARRAY_PARTITION variable=next_means0 block factor=16 dim=1
    MEANS next_means1[MAX_MODEL_NUM];
    #pragma HLS ARRAY_PARTITION variable=next_means1 block factor=16 dim=1
    MEANS next_means2[MAX_MODEL_NUM];
    #pragma HLS ARRAY_PARTITION variable=next_means2 block factor=16 dim=1

    ap_uint<8> count[MAX_MODEL_NUM];
    #pragma HLS ARRAY_PARTITION variable=count cyclic factor=16 dim=1


    resetnextmeans:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
        next_means0[i] = 0;
        next_means1[i] = 0;
        next_means2[i] = 0;
        count[i] = 0;
    }

    KMeansCore(mm2s, k_means, next_means0, next_means1, next_means2, count);

    Update_means:for(uint32_t i=0; i<MAX_MODEL_NUM; i++){
        #pragma HLS PIPELINE II=1
        if(count[i]>0){
            k_means[i][0] = next_means0[i]/count[i];
            k_means[i][1] = next_means1[i]/count[i];
            k_means[i][2] = next_means2[i]/count[i];
        }
    }
}