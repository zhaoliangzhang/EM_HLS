#ifndef _KMEANS_H_
#define _KMEANS_H_

#include "config.h"
#include "hls_stream.h"

void KMeans(hls::stream<ap_uint<32> > &mm2s,
MEANS k_means[MAX_MODEL_NUM][DIM]
);

#endif