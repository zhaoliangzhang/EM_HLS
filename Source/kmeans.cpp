#include "kmeans.h"
#include "hls_math.h"

void GetData(hls::stream<ap_uint<32> > &mm2s, hls::stream<DIS> &data, hls::stream<DIS> &data2) {
#pragma HLS dataflow
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
#pragma HLS dataflow
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

void GetLabel(hls::stream<DIS> dis[MAX_MODEL_NUM], hls::stream<ap_uint<1> > label[MAX_MODEL_NUM]){
#pragma HLS dataflow
    DIS local_dis[MAX_MODEL_NUM];
    #pragma HLS ARRAY_PARTITION variable=local_dis cyclic factor=16 dim=1 partition

    DIS local_dis_128[128];
    DIS local_dis_64[64];
    DIS local_dis_32[32];
    DIS local_dis_16[16];
    DIS local_dis_8[8];
    DIS local_dis_4[4];
    DIS local_dis_2[2];

    uint32_t label_128[128];
    uint32_t label_64[64];
    uint32_t label_32[32];
    uint32_t label_16[16];
    uint32_t label_8[8];
    uint32_t label_4[4];
    uint32_t label_2[2];

    Getlab:for(uint32_t n=0; n<DATA_NUM; n++) {
    //#pragma HLS PIPELINE
        update:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
            dis[i].read(local_dis[i]);
        }

        for(uint32_t i=0; i<128; i++){
            if(local_dis[i] > local_dis[128+i]){
                local_dis_128[i] = local_dis[i];
                label_128[i] = i;
            } else {
                local_dis_128[i] = local_dis[i+128];
                label_128[i] = i+128;
            }
        }
        
        for(uint32_t i=0; i<64; i++) {
            if(local_dis_128[i] > local_dis_128[i+64]) {
                local_dis_64[i] = local_dis_128[i];
                label_64[i] = label_128[i];
            } else {
                local_dis_64[i] = local_dis_128[i+64];
                label_64[i] = label_128[i+64];
            }
        }

        for(uint32_t i=0; i<32; i++) {
            if(local_dis_64[i] > local_dis_64[i+32]) {
                local_dis_32[i] = local_dis_64[i];
                label_32[i] = label_64[i];
            } else {
                local_dis_32[i] = local_dis_64[i+32];
                label_32[i] = label_64[i+32];
            }
        }

        for(uint32_t i=0; i<16; i++) {
            if(local_dis_32[i] > local_dis_32[i+16]) {
                local_dis_16[i] = local_dis_32[i];
                label_16[i] = label_32[i];
            } else {
                local_dis_16[i] = local_dis_32[i+16];
                label_16[i] = label_32[i+16];
            }
        }

        for(uint32_t i=0; i<8; i++) {
            if(local_dis_16[i] > local_dis_16[i+8]) {
                local_dis_8[i] = local_dis_16[i];
                label_8[i] = label_16[i];
            } else {
                local_dis_8[i] = local_dis_16[i+8];
                label_8[i] = label_16[i+8];
            }
        }

        for(uint32_t i=0; i<4; i++) {
            if(local_dis_8[i] > local_dis_8[i+4]) {
                local_dis_4[i] = local_dis_8[i];
                label_4[i] = label_8[i];
            } else {
                local_dis_4[i] = local_dis_8[i+4];
                label_4[i] = label_8[i+4];
            }
        }

        for(uint32_t i=0; i<2; i++) {
            if(local_dis_4[i] > local_dis_4[i+2]) {
                local_dis_2[i] = local_dis_4[i];
                label_2[i] = label_4[i];
            } else {
                local_dis_2[i] = local_dis_4[i+2];
                label_2[i] = label_4[i+2];
            }
        }

        uint32_t p;
        if(local_dis_2[0]>local_dis_2[1]) {
            p = label_2[0];
        } else {
            p = label_2[1];
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
#pragma HLS dataflow
    for(uint32_t i=0; i<DATA_NUM; i++) {
    //#pragma HLS PIPELINE
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


void  KMeans(hls::stream<ap_uint<32> > &mm2s,
MEANS k_means[MAX_MODEL_NUM][DIM],
uint32_t cnt_in) {

    ap_uint<32> iterNum = 0;
    MEANS next_means0[MAX_MODEL_NUM];
    #pragma HLS ARRAY_PARTITION variable=next_means0 block factor=16 dim=1 partition
    MEANS next_means1[MAX_MODEL_NUM];
    #pragma HLS ARRAY_PARTITION variable=next_means1 block factor=16 dim=1 partition
    MEANS next_means2[MAX_MODEL_NUM];
    #pragma HLS ARRAY_PARTITION variable=next_means2 block factor=16 dim=1 partition
    ap_uint<8> count[MAX_MODEL_NUM];
    #pragma HLS ARRAY_PARTITION variable=counts cyclic factor=16 dim=1 partition

    //#pragma HLS DATAFLOW

    resetnextmeans:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
        next_means0[i] = 0;
        next_means1[i] = 0;
        next_means2[i] = 0;
        count[i] = 0;
    }

    /*Full:for(uint32_t i=0; i<DATA_NUM; i++) {
        #pragma HLS PIPELINE II=3
        DATA sample[3];
        readdata:for(uint32_t j=0; j<3; j++) {
            ap_uint<32> tmp;
            mm2s.read(tmp);
            sample[j] = *(float *)(&tmp);
        }
        CalDis(sample, dis, k_means);
        GetLabel(dis, label);
        Update(label, next_means0, next_means1, next_means2, count, sample);
    }*/

    hls::stream<DIS> data;
    #pragma HLS STREAM variable=data depth=768
    hls::stream<DIS> data2;
    #pragma HLS STREAM variable=data2 depth=768
    hls::stream<DIS> dis[MAX_MODEL_NUM];
    #pragma HLS STREAM variable=dis depth=128
    hls::stream<ap_uint<1> > label[MAX_MODEL_NUM];
    #pragma HLS STREAM variable=label depth=128

    GetData(mm2s, data, data2);
    CalDis(data, dis, k_means);
    GetLabel(dis, label);
    Update(label, next_means0, next_means1, next_means2, count, data2);

    Update_means:for(uint32_t i=0; i<MAX_MODEL_NUM; i++){
        #pragma HLS PIPELINE II=1
        if(count[i]>0){
            k_means[i][0] = next_means0[i]/count[i];
            k_means[i][1] = next_means1[i]/count[i];
            k_means[i][2] = next_means2[i]/count[i];
        }
    }
}

void ToRAM(MEANS k_means[MAX_MODEL_NUM][DIM],
ap_uint<32> ram[MAX_MODEL_NUM*DIM]) {
    output_label0:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
        output_label1:for(uint32_t j=0; j<DIM; j++) {
            ram[i*DIM+j] = k_means[i][j];
        }
    }
}