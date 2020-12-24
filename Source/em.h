#ifndef _EM_H_
#define _EM_H_

#include "config.h"
#include "hls_stream.h"

void EM(hls::stream<ap_uint<32> > &mm2s,
PRIOR priors[MAX_MODEL_NUM],
MEANS means[MAX_MODEL_NUM][DIM],
VARS  vars[MAX_MODEL_NUM][DIM],
ap_uint<1> func
);

#endif