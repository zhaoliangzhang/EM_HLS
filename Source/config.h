#ifndef _CONFIG_H_
#define _CONFIG_H_

#include <stdint.h>
#include "ap_int.h"

#define PRIOR float
#define MEANS float
#define VARS float
#define DATA float
#define PROB float

const uint32_t  DIM     = 3;
const uint8_t   ADDR_W  = 40;
const uint8_t   BTT_W   = 23;
const uint8_t   CMD_W   = ADDR_W + BTT_W + 17;
const uint32_t  MAX_MODEL_NUM = 256;
const uint32_t  DATA_NUM = 10;

const PROB  genhao_er_pai_fenzhiyi = 0.56419;

#endif