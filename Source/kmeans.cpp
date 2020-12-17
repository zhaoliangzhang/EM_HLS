#include "kmeans.h"
#include "hls_math.h"


//void CalDis(DATA* sample, hls::stream<DIS> &dis, MEANS k_means[MAX_MODEL_NUM][DIM]){
void CalDis(DATA sample[3], hls::stream<DIS> dis[MAX_MODEL_NUM], MEANS k_means[MAX_MODEL_NUM][DIM]){
    CalDIS:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
        #pragma HLS PIPELINE II=2
        DIS tmp_dis = 0;
        DIS axe_dis = 0;
        distance:for (uint32_t d = 0; d < DIM; d++) {
            axe_dis = (sample[d]-k_means[i][d]);
            tmp_dis = fabs(axe_dis);
        }
        dis[i].write(tmp_dis);
    }
}


/*void compare(DIS* ){
    
}*/
//void GetLabel(hls::stream<DIS> &dis, hls::stream<uint32_t> &label){
void GetLabel(hls::stream<DIS> dis[MAX_MODEL_NUM], hls::stream<DIS> label[MAX_MODEL_NUM]){
    DIS local_dis[MAX_MODEL_NUM];
    #pragma HLS ARRAY_PARTITION variable=local_dis cyclic factor=256 dim=1 partition
    update:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
        dis[i].read(local_dis[i]);
    }

    DIS labels[MAX_MODEL_NUM];

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

    
    /*hls::stream<DIS> local_dis_128[128];
    #pragma HLS STREAM variable=local_dis_128 depth=16
    hls::stream<DIS> local_dis_64[64];
    #pragma HLS STREAM variable=local_dis_64 depth=16
    hls::stream<DIS> local_dis_32[32];
    #pragma HLS STREAM variable=local_dis_32 depth=16
    hls::stream<DIS> local_dis_16[16];
    #pragma HLS STREAM variable=local_dis_16 depth=16
    hls::stream<DIS> local_dis_8[8];
    #pragma HLS STREAM variable=local_dis_8 depth=16
    hls::stream<DIS> local_dis_4[4];
    #pragma HLS STREAM variable=local_dis_4 depth=16
    hls::stream<DIS> local_dis_2[2];
    #pragma HLS STREAM variable=local_dis_2 depth=16

    hls::stream<uint32_t> label_128[128];
    #pragma HLS STREAM variable=label_128 depth=16
    hls::stream<uint32_t> label_64[64];
    #pragma HLS STREAM variable=label_64 depth=16
    hls::stream<uint32_t> label_32[32];
    #pragma HLS STREAM variable=label_32 depth=16
    hls::stream<uint32_t> label_16[16];
    #pragma HLS STREAM variable=label_16 depth=16
    hls::stream<uint32_t> label_8[8];
    #pragma HLS STREAM variable=label_8 depth=16
    hls::stream<uint32_t> label_4[4];
    #pragma HLS STREAM variable=label_4 depth=16
    hls::stream<uint32_t> label_2[2];
    #pragma HLS STREAM variable=label_2 depth=16
    
    for(uint32_t i=0; i<128; i++){
        if(local_dis[i] > local_dis[128+i]){
            local_dis_128[i].write(local_dis[i]);
            label_128[i].write(i);
        } else {
            local_dis_128[i].write(local_dis[i+128]);
            label_128[i].write(i+128);
        }
    }

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

    uint32_t p;

    DIS local_tmp, local_tmp1;
    local_dis_2[0].read(local_tmp);
    local_dis_2[1].read(local_tmp1);
    if(local_tmp>local_tmp1) {
        label_2[0].read(p);
    } else {
        label_2[1].read(p);
    }*/

    for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
        if(i==p) {
            label[i].write(1);
        } else {
            label[i].write(0);
        }
    }

}

void Update(hls::stream<DIS> label[MAX_MODEL_NUM], MEANS next_means0[MAX_MODEL_NUM], MEANS next_means1[MAX_MODEL_NUM], MEANS next_means2[MAX_MODEL_NUM], ap_uint<8> count[MAX_MODEL_NUM], DATA sample[DIM]){
    update:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
        #pragma HLS PIPELINE II=1
        DIS lab;
        label[i].read(lab);
        next_means0[i] += sample[0]*lab;
        next_means1[i] += sample[1]*lab;
        next_means2[i] += sample[2]*lab;
        if(lab){
            count[i]++;
        }
    }
}


void  KMeans(hls::stream<ap_uint<32> > &mm2s,
MEANS k_means[MAX_MODEL_NUM][DIM],
uint32_t cnt_in) {

    ap_uint<32> iterNum = 0;
    MEANS next_means0[MAX_MODEL_NUM];
    #pragma HLS ARRAY_PARTITION variable=next_means0 block factor=256 dim=1 partition
    MEANS next_means1[MAX_MODEL_NUM];
    #pragma HLS ARRAY_PARTITION variable=next_means1 block factor=256 dim=1 partition
    MEANS next_means2[MAX_MODEL_NUM];
    #pragma HLS ARRAY_PARTITION variable=next_means2 block factor=256 dim=1 partition
    ap_uint<8> count[MAX_MODEL_NUM];
    #pragma HLS ARRAY_PARTITION variable=counts cyclic factor=256 dim=1 partition

    //#pragma HLS DATAFLOW
    hls::stream<DIS> dis[MAX_MODEL_NUM];
    #pragma HLS STREAM variable=dis depth=MAX_MODEL_NUM
    hls::stream<DIS> label[MAX_MODEL_NUM];
    //hls::stream<uint32_t> label;
    #pragma HLS STREAM variable=label depth=MAX_MODEL_NUM


    resetnextmeans:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
        next_means0[i] = 0;
        next_means1[i] = 0;
        next_means2[i] = 0;
        count[i] = 0;
    }

    //DIS local_dis[MAX_MODEL_NUM];
    //#pragma HLS ARRAY_PARTITION variable=local_dis cyclic factor=8 dim=1 partition

    Full:for(uint32_t i=0; i<DATA_NUM; i++) {
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
    }

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