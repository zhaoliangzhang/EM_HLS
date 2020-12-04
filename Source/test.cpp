#include "test.h"

DATA string_to_num(string str) {
	int i=0,len=str.length();
    DATA sum=0;
	if(str[0] == '-'){
		i++;
	}
    while(i<len){
        if(str[i]=='.') break;
        sum=sum*10+str[i]-'0';
        ++i;
    }
    ++i;
    DATA t=1,d=1;
    while(i<len){
        d*=0.1;
        t=str[i]-'0';
        sum+=t*d;
        ++i;
    }
	if(str[0] == '-'){
		sum *= -1;
	}
    return sum;
}

void FileToData(fstream fp, std::vetor<DATA> &data) {
    string tmp;
    uint32_t split_index[6];
    fp>>tmp;
    uint32_t index = 0;
    for(uint32_t i=0; i<180000; i++) {
        fp>>tmp;
        if(i%18 == 0) {
            uint32_t k = 0;
            for(uint32_t j=0; j<tmp.length(); j++) {
                if(tmp[j] == ',') {
                    split_index[k] = j;
                    k +=1;
                }
            }
            data.pushback(string_to_num(tmp.substr(split_index[0]+1,(split_index[1]-split_index[0]-1))));
            data.pushback(string_to_num(tmp.substr(split_index[1]+1,(split_index[2]-split_index[1]-1))));
			data.pushback(string_to_num(tmp.substr(split_index[2]+1,(split_index[3]-split_index[2]-1))));
            index += 1;
        }
    }
}

void DataToStream(std::vector<DATA> &data, hls::stream<ap_uint<32> > data_str) {
    for(auto d:data) {
        ap_uint<32> tmp;
        tmp = *(uint32_t *)(&d);
        data_str.write(tmp);
    }
}

void DataToMstream(std::vector<DATA> &data, hls::stream<ap_uint<32> > means_str) {
    for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
        uint32_t select = i*DATA_NUM / MAX_MODEL_NUM;
        for(uint32_t j=0; j<3; j++){
            ap_uint<32> tmp;
            tmp = *(uint32_t *)(&(data[select*3+j]));
            means_str.write(tmp);
        }
    }
}