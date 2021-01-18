#include <iostream>
#include "top.h"
#include "test.h"
#include <string>

using namespace std;

int main(){

    float data[90000];
    //FileToData(data);

    PRIOR priors[MAX_MODEL_NUM];
    MEANS means[MAX_MODEL_NUM*3];
    VARS vars[MAX_MODEL_NUM*3];
    ap_uint<1> func=1;


    for(int i=0; i<MAX_MODEL_NUM; i++){
        priors[i] = 0.1;

        means[i*3] = 0;
        means[i*3+1] = 0;
        means[i*3+2] = 0;

        vars[i*3] = 0;
        vars[i*3+1] = 0;
        vars[i*3+2] = 0;
    }
    top(data, priors, means, vars, func);

    return 0;
}
