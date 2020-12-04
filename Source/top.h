#ifndef _TOP_H_
#define _TOP_H_

#include "config.h"
#include "kmeans.h"
#include "ap_axi_sdata.h"
#include "datamover_cmd.h"

namespace efc {
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
    void top(hls::stream<ap_uint<32> > &mm2s,
    hls::stream<ap_uint<CMD_W>> &mm2s_cmd,
    hls::stream<ap_uint<32> > &mm2s_means,
    hls::stream<ap_uint<CMD_W> > &mm2s_means_cmd,
    ap_uint<ADDR_W> addr_in,
    ap_uint<ADDR_W> addr_means_in,
    ap_uint<32> pram[MAX_MODEL_NUM*DIM],
    uint32_t cnt_in,
    uint32_t &cnt_out
    );
    
}


#endif