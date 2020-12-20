#ifndef _TOP_H_
#define _TOP_H_

#include "config.h"
#include "kmeans.h"
#include "ap_axi_sdata.h"
#include "datamover_cmd.h"
#include "hls_stream.h"

    /**
    * @brief Top wrapper of kmeans module
    * @param[in]    mm2s              input 32-bit MM2S stream
    * @param[out]   mm2s_cmd          MM2S stream command
    * @param[in]    addr_in           input point cloud address
    * @param[in]    addr_means_in     input point means address
    * @param[out]   pram              output means ram port 
    * @param[in]    cnt_in            number of input points
    * @param[out]   cnt_out           number of output means
    */
    void top(DATA _data[DATA_NUM*3],
    MEANS _k_means[MAX_MODEL_NUM*3]
    );



#endif
