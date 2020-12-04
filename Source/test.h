#ifndef _TEST_H_
#define _TEST_H_

#include <iostream>
#include <string>
#include <vector>
#include "hls_stream.h"
#include "config.h"

namespace efc {

    void FileToData(fstream fp, std::vector<DATA> &data);
    void DataToStream(std::vector<DATA> &data, hls::stream<ap_uint<32> > data_str);
    void DataToMstream(std::vector<DATA> &data, hls::stream<ap_uint<32> > means_str);

}

#endif