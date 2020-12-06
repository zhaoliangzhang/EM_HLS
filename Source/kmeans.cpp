#include "kmeans.h"
#include "hls_math.h"


void CalDis(DATA* sample, hls::stream<DIS> &dis, MEANS k_means[MAX_MODEL_NUM][DIM]){
    
    CalDIS:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=MAX_MODEL_NUM
        DIS tmp_dis=0;
        distance:for (uint32_t d = 0; d < DIM; d++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS UNROLL factor=DIM
            tmp_dis += (sample[d]-k_means[i][d]) * (sample[d]-k_means[i][d]);
        }
        tmp_dis = sqrt(tmp_dis);
        dis.write(tmp_dis);
    }
}

void GetLabel(hls::stream<DIS> &dis, hls::stream<uint32_t> &label){
    DIS max = 0;
    uint32_t p = 0;
    Label:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=MAX_MODEL_NUM
        DIS tmp;
        dis.read(tmp);
        if(tmp>max){
            max = tmp;
            p = i;
        }
    }
    LabelStream:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=MAX_MODEL_NUM
        if(i == p){
            label.write(1);
        } else {
            label.write(0);
        }
    }
}

void Update(hls::stream<uint32_t> &label, MEANS next_means[MAX_MODEL_NUM][DIM], ap_uint<8> count[MAX_MODEL_NUM], DATA sample[DIM]){
    #pragma HLS ARRAY_PARTITION variable=next_means block factor=8 dim=1 partition
    float lab[MAX_MODEL_NUM];
    onehot:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=MAX_MODEL_NUM
        uint32_t tmp;
        label.read(tmp);
        lab[i] = *(float *)(&tmp);
    }
    update:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=MAX_MODEL_NUM
        updateinner:for(uint32_t j=0; j<DIM; j++) {
            #pragma HLS UNROLL factor=DIM
            next_means[i][j] += sample[j]*lab[i];
        }
        if(lab[i]){
            count[i]++;
        }
    }
}


void  KMeans(hls::stream<ap_uint<32> > &mm2s,
MEANS k_means[MAX_MODEL_NUM][DIM],
uint32_t cnt_in) {
    #pragma HLS ARRAY_PARTITION variable=k_means block factor=8 dim=1 partition

    ap_uint<32> iterNum = 0;
    ap_uint<8> count[MAX_MODEL_NUM];
    MEANS next_means[MAX_MODEL_NUM][DIM];

    //#pragma HLS DATAFLOW
    hls::stream<DIS> dis;
    #pragma HLS STREAM variable=dis depth=MAX_MODEL_NUM
    hls::stream<uint32_t> label;
    #pragma HLS STREAM variable=label depth=MAX_MODEL_NUM


    resetnextmeans:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
        #pragma HLS UNROLL factor=MAX_MODEL_NUM
        next_means[i][0] = 0;
        next_means[i][1] = 0;
        next_means[i][2] = 0;
        count[i] = 0;
    }

    DIS local_dis[MAX_MODEL_NUM];

    Full:for(uint32_t i=0; i<DATA_NUM; i++) {
        DATA sample[3];
        readdata:for(uint32_t j=0; j<3; j++) {
            ap_uint<32> tmp;
            mm2s.read(tmp);
            sample[j] = *(float *)(&tmp);
        }
        CalDis(sample, dis, k_means);
        GetLabel(dis, label);
        Update(label, next_means, count, sample);
    }

    for(uint32_t i=0; i<MAX_MODEL_NUM; i++){
        #pragma HLS PIPELINE
        #pragma HLS UNROLL factor=MAX_MODEL_NUM
        if(count[i]>0){
            k_means[i][0] = next_means[i][0]/count[i];
            k_means[i][1] = next_means[i][1]/count[i];
            k_means[i][2] = next_means[i][2]/count[i];
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