#include "kmeans.h"
#include "hls_math.h"

DIS CalcDistance(const DATA* x, const MEANS* u) {
	DIS temp = 0;
	distance:for (uint32_t d = 0; d < DIM; d++) {
		temp += (x[d] - u[d]) * (x[d] - u[d]);
	}
	return sqrt(temp);
}

void CalDis(DATA* sample, hls::stream<DIS> dis, MEANS** k_means){
    
    CalDIS:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
        DIS tmp_dis;
        tmp_dis = CalcDistance(sample, k_means[i]);
        dis.write(tmp_dis);
    }
}

void GetLabel(hls::stream<DIS> dis, hls::stream<uint32_t> label){
    DIS max = 0;
    uint32_t p = 0;
    Label:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
        DIS tmp;
        dis.read(tmp);
        if(tmp>max){
            max = tmp;
            p = i;
        }
    }
    label.write(p);
}

void Update(hls::stream<uint32_t> label, MEANS** next_means, ap_uint<8>* count, DATA* sample){
    update:for(uint32_t i=0; i<DIM; i++) {
        next_means[label][i] += sample[i];
    }
    count[label]++;
}


void  KMeans(hls::stream<ap_uint<32> > &mm2s,
MEANS k_means[MAX_MODEL_NUM][DIM],
uint32_t cnt_in) {

    ap_uint<32> iterNum = 0;
    ap_uint<8> count[MAX_MODEL_NUM];
    MEANS next_means[MAX_MODEL_NUM][DIM];

    #pragma HLS DATAFLOW
    
    hls::stream<DIS> dis;
    hls::stream<uint32_t> label;


    for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
        next_means[i][0] = 0;
        next_means[i][1] = 0;
        next_means[i][2] = 0;
        count[i] = 0;
    }

    DIS local_dis[MAX_MODEL_NUM];

    for(uint32_t i=0; i<cnt_in; i++) {
        DATA sample[3];
        readdata:for(uint32_t i=0; i<3; i++) {
            unsigned int tmp;
            mm2s.read(tmp);
            sample[i] = *(float *)(&tmp);
        }
        CalDis(sample, dis, k_means, local_dis);
        GetLabel(dis, label);
        Update(label, next_means, count, sample);
    }

    for(uint32_t i=0; i<MAX_MODEL_NUM; i++){
        if(count[i]>0){
            k_means[i][0] = next_means[i][0]/count[i];
            k_means[i][1] = next_means[i][1]/count[i];
            k_means[i][2] = next_means[i][2]/count[i];
        }
    }
}

void ToRAM(MEANS k_means[MAX_MODEL_NUM][DIM],
ap_uint<32> ram[MAX_MODEL_NUM*DIM]) {
    output_label0:for(uint32_t i=0; i<MAX_MODEL_NUM; i++) {
        output_label1:for(uint32_t j=0; j<DIM; j++) {
            ram[i*DIM+j] = k_means[i][j];
        }
    }
}