#include <iostream>
#include "top.h"
#include "test.h"
#include <string>

using namespace std;

int main(){

    DATA data[6000];
    FileToData(data);

    MEANS means[MAX_MODEL_NUM*3];
    for(int i=0; i<256; i++){
        means[i*3] = 0;
        means[i*3+1] = 0;
        means[i*3+2] = 0;
    }
    top(data, means);

    return 0;

}