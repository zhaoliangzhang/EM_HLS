#include <iostream>
#include "top.h"
#include "test.h"

using namespace std;
using namespace efc;

int main(){

    std::vector<DATA> data;

    fstream fp;
	fp.open("../PointCloud6.csv");

    hls::stream<ap_uint<32> > data_str;
    hls::stream<ap_uint<32> > means_str;
    ap_uint<32> pram = new ap_uint<32>[MAX_MODEL_NUM*DIM];
    uint32_t cnt_out;
    hls::stream<ap_uint<CMD_W> > mm2s_cmd;
    hls::stream<ap_uint<CMD_W> > mm2s_means_cmd;
    ap_uint<ADDR_W> addr_in = 0;
    ap_uint<ADDR_W> addr_means_in = DATA_NUM*DIM*4+4;
    ap_uint<CMD_W> cmd;
    ap_uint<CMD_W> means_cmd;

    FileToData(fp, data);
    DataToStream(data, data_str);
    DataToMstream(data, means_str);

    top(data_str,
    mm2s_cmd,
    means_str,
    mm2s_means_cmd,
    addr_in,
    addr_means_in,
    pram,
    DATA_NUM,
    cnt_out)

    mm2s_cmd.read(cmd);
    mm2s_means_cmd.read(means_cmd);

    fp.close();
    delete [] pram;
    return 0;

}