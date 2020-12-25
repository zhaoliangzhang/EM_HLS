#ifndef _TEST_H_
#define _TEST_H_

#include <iostream>
#include <fstream>
#include <vector>
#include "hls_stream.h"
#include "config.h"

void FileToData(DATA* data);
void DataToStream(std::vector<DATA> &data, hls::stream<ap_uint<32> > &data_str);
void DataToMstream(std::vector<DATA> &data, hls::stream<MEANS> &means_str);

#endif