#include "top.h"


namespace efc {

    void top(hls::stream<ap_uint <32> > &mm2s,
    hls::stream<ap_uint<CMD_W>> &mm2s_cmd,
    hls::stream<MEANS> &mm2s_means,
    hls::stream<ap_uint<CMD_W> > &mm2s_means_cmd,
    ap_uint<ADDR_W> addr_in,
    ap_uint<ADDR_W> addr_means_in,
    ap_uint<32> pram[MAX_MODEL_NUM*DIM],
    uint32_t cnt_in,
    uint32_t &cnt_out
    ) {
        #pragma HLS INTERFACE s_axilite register port=return bundle=ctrl
        #pragma HLS INTERFACE s_axilite register port=addr_in bundle=ctrl
        #pragma HLS INTERFACE s_axilite register port=cnt_in bundle=ctrl

        #pragma HLS INTERFACE bram port=ram depth=32768 storage_type=ram_1p
        #pragma HLS INTERFACE ap_none port=cnt_out
        #pragma HLS INTERFACE axis register both port=mm2s
        #pragma HLS INTERFACE axis register both port=mm2s_cmd
        #pragma HLS INTERFACE axis register both port=mm2s_means
        #pragma HLS INTERFACE axis register both port=mm2s_means_cmd

    MEANS k_means[MAX_MODEL_NUM][DIM];

    DatamoverCmd<ADDR_W, BTT_W> cmd_in(addr_in, ap_uint<BTT_W>(cnt_in * 12));
    DatamoverCmd<ADDR_W, BTT_W> cmd_means_in(addr_means_in, ap_uint<BTT_W>(MAX_MODEL_NUM * 12));
    mm2s_cmd.write(cmd_in.word());
    mm2s_means_cmd.write(cmd_means_in.word());

    KMeansInit:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
        unsigned int tmp[3];
        mm2s_means.read(tmp[i][0]); k_means[i][0] = *(float *)(&(tmp[i][0]));
        mm2s_means.read(tmp[i][1]); k_means[i][1] = *(float *)(&(tmp[i][1]));
        mm2s_means.read(tmp[i][2]); k_means[i][2] = *(float *)(&(tmp[i][2]));
    }

    KMeans(mm2s, k_means, cnt_in);
    ToRAM(k_means, ram);
    }
}