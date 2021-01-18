#include "top.h"


void top(float _data[DATA_NUM],
PRIOR _priors[MAX_MODEL_NUM],
MEANS _means[MAX_MODEL_NUM*3],
VARS  _vars[MAX_MODEL_NUM*3],
ap_uint<1> func
) {
#pragma HLS INTERFACE s_axilite port=return bundle=hls_ctrl
#pragma HLS INTERFACE s_axilite port=func bundle=func_ctrl
#pragma HLS INTERFACE m_axi depth=DATA_NUM+1 port=_data offset=slave bundle=in_data
#pragma HLS INTERFACE m_axi depth=MAX_MODEL_NUM port=_priors offset=slave bundle=in_priors
#pragma HLS INTERFACE m_axi depth=MAX_MODEL_NUM*3 port=_means offset=slave bundle=in_means
#pragma HLS INTERFACE m_axi depth=MAX_MODEL_NUM*3 port=_vars offset=slave bundle=in_vars

    PRIOR prior_buffer[MAX_MODEL_NUM];
    #pragma HLS ARRAY_PARTITION variable=prior_buffer block factor=8 dim=1

    MEANS mean_buffer[MAX_MODEL_NUM][DIM];
    #pragma HLS ARRAY_PARTITION variable=mean_buffer block factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=mean_buffer block factor=3 dim=2

    VARS var_buffer[MAX_MODEL_NUM][DIM];
    #pragma HLS ARRAY_PARTITION variable=var_buffer block factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=var_buffer block factor=3 dim=2

    for(int i=0; i<MAX_MODEL_NUM; i++) {
        #pragma HLS PIPELINE
        prior_buffer[i] = _priors[i];
    }

    for(int i=0; i<MAX_MODEL_NUM; i++) {
        #pragma HLS PIPELINE
        mean_buffer[i][0] = _means[i*3];
        mean_buffer[i][1] = _means[i*3+1];
        mean_buffer[i][2] = _means[i*3+2];
    }

    for(int i=0; i<MAX_MODEL_NUM; i++) {
        #pragma HLS PIPELINE
        var_buffer[i][0] = _vars[i*3];
        var_buffer[i][1] = _vars[i*3+1];
        var_buffer[i][2] = _vars[i*3+2];
    }

    EM(_data, prior_buffer, mean_buffer, var_buffer, func);

    for(int i=0; i<MAX_MODEL_NUM; i++) {
        _priors[i] = prior_buffer[i];
    }
    for(int i=0; i<MAX_MODEL_NUM; i++) {
    #pragma HLS PIPELINE off
        _means[i*3] = mean_buffer[i][0];
        _means[i*3+1] = mean_buffer[i][1];
        _means[i*3+2] = mean_buffer[i][2];
    }
    for(int i=0; i<MAX_MODEL_NUM; i++) {
    #pragma HLS PIPELINE off
        _vars[i*3] = var_buffer[i][0];
        _vars[i*3+1] = var_buffer[i][1];
        _vars[i*3+2] = var_buffer[i][2];
    }

}
