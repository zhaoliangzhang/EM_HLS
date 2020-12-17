#include "kmeans.h"
#include "hls_math.h"


void  KMeans(hls::stream<ap_uint<32> > &mm2s,
MEANS k_means[MAX_MODEL_NUM][DIM],
uint32_t cnt_in) {

    ap_uint<32> iterNum = 0;
    ap_uint<8> count[MAX_MODEL_NUM];
    /*MEANS next_means0[MAX_MODEL_NUM];
    MEANS next_means1[MAX_MODEL_NUM];
    MEANS next_means2[MAX_MODEL_NUM];
    #pragma HLS ARRAY_PARTITION variable=next_means0 block factor=32 dim=1 partition
    #pragma HLS ARRAY_PARTITION variable=next_means1 block factor=32 dim=1 partition
    #pragma HLS ARRAY_PARTITION variable=next_means2 block factor=32 dim=1 partition*/
    #pragma HLS ARRAY_PARTITION variable=counts cyclic factor=8 dim=1 partition

    //#pragma HLS DATAFLOW
    hls::stream<DIS> dis;
    #pragma HLS STREAM variable=dis depth=256
    hls::stream<DIS> label_stream[MAX_MODEL_NUM];
    #pragma HLS STREAM variable=label_stream depth=16

    hls::stream<MEANS> next_means0[MAX_MODEL_NUM];
    #pragma HLS STREAM variable=next_means0 depth=16
    hls::stream<MEANS> next_means1[MAX_MODEL_NUM];
    #pragma HLS STREAM variable=next_means1 depth=16
    hls::stream<MEANS> next_means2[MAX_MODEL_NUM];
    #pragma HLS STREAM variable=next_means2 depth=16

    #pragma HLS ARRAY_PARTITION variable=next_means0 block factor=32 dim=1 partition
    #pragma HLS ARRAY_PARTITION variable=next_means1 block factor=32 dim=1 partition
    #pragma HLS ARRAY_PARTITION variable=next_means2 block factor=32 dim=1 partition

    MEANS label[MAX_MODEL_NUM];
    #pragma HLS ARRAY_PARTITION variable=counts cyclic factor=8 dim=1 partition


    /*resetnextmeans:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
        next_means0[i] = 0;
        next_means1[i] = 0;
        next_means2[i] = 0;
        count[i] = 0;
    }*/

    DIS local_dis[MAX_MODEL_NUM];
    #pragma HLS ARRAY_PARTITION variable=local_dis cyclic factor=8 dim=1 partition

    Full:for(uint32_t i=0; i<DATA_NUM; i++) {
        #pragma HLS PIPELINE II=3
        DATA sample[3];
        readdata:for(uint32_t j=0; j<3; j++) {
            ap_uint<32> tmp;
            mm2s.read(tmp);
            sample[j] = *(float *)(&tmp);
        }

        //Calculate distance
        CalDIS:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
            DIS tmp_dis = 0;
            DIS axe_dis = 0;
            distance:for (uint32_t d = 0; d < DIM; d++) {
                axe_dis = (sample[d]-k_means[i][d]);
                tmp_dis += axe_dis*axe_dis;
            }
            local_dis[i] = tmp_dis;
        }
        
        /*for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
            dis.write(local_dis[i]);
        }*/

        DIS max = 0;
        uint32_t p = 0;
        Label:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
            /*DIS tmp_dis;
            dis.read(tmp_dis);*/
            if(local_dis[i]>max){
                max = local_dis[i];
                p = i;
            }
        }
        LabelStream:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
            if(i == p){
                label_stream[i].write(1);
            } else {
                label_stream[i].write(0);
            }
        }

        update:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
            DIS lab;
            label_stream.read(lab);
            next_means0[i] += sample[0]*lab;
            next_means1[i] += sample[1]*lab;
            next_means2[i] += sample[2]*lab;
            if(lab){
                count[i]++;
            }
        }
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