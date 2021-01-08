#ifndef _CONFIG_H_
#define _CONFIG_H_

#include <stdint.h>
#include "ap_int.h"

/*#define PRIOR float
#define MEANS float
#define VARS float
#define DATA float
#define PROB float*/

#define PRIOR ap_ufixed<16, 8, AP_RND, AP_SAT>
#define MEANS ap_fixed<16, 8, AP_RND, AP_SAT>
#define VARS ap_fixed<16, 8, AP_RND, AP_SAT>
#define DATA ap_fixed<16, 8, AP_RND, AP_SAT>
#define PROB ap_fixed<16, 8, AP_RND, AP_SAT>

const uint32_t  DIM     = 3;
const uint8_t   ADDR_W  = 40;
const uint8_t   BTT_W   = 23;
const uint8_t   CMD_W   = ADDR_W + BTT_W + 17;
const uint32_t  MAX_MODEL_NUM = 128;
const uint32_t  DATA_NUM = 1000;

const PROB  genhao_er_pai_fenzhiyi = 0.56419;

template<uint32_t size, class T>
struct Vector{
    T vec[size];
};

#endif