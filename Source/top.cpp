#include "top.h"


void top(DATA _data[DATA_NUM*3],
MEANS _k_means[MAX_MODEL_NUM*3]
) {
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE m_axi depth=6000 port=_data offset=slave bundle=in_data
#pragma HLS INTERFACE m_axi depth=256*3 port=_k_means offset=slave bundle=in_means

    hls::stream<ap_uint<32> > mm2s;
    #pragma HLS STREAM variable=mm2s depth=16
    MEANS mean_buffer[MAX_MODEL_NUM][DIM];
    #pragma HLS ARRAY_PARTITION variable=mean_buffer block factor=16 dim=1

    for(int i=0; i<DATA_NUM; i++) {
        for(int j=0; j<3; j++) {
            float tmp;
            tmp = _data[i*3+j];
            ap_uint<32> x;
            x = *(ap_uint<32> *)&tmp;
            mm2s.write(x)
        }
    }

    for(int i=0; i<MAX_MODEL_NUM; i++) {
        #pragma HLS PIPELINE
        mean_buffer[i][0] = _k_means[i*3];
        mean_buffer[i][1] = _k_means[i*3+1];
        mean_buffer[i][2] = _k_means[i*3+2];
    }

    KMeans(mm2s, mean_buffer);

    for(int i=0; i<MAX_MODEL_NUM; i++) {
        #pragma HLS PIPELINE
        _k_means[i*3] = mean_buffer[i][0];
        _k_means[i*3+1] = mean_buffer[i][1];
        _k_means[i*3+2] = mean_buffer[i][2];
    }

}
