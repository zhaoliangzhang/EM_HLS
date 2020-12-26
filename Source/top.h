#ifndef _TOP_H_
#define _TOP_H_

#include "config.h"
#include "em.h"
#include "ap_axi_sdata.h"
#include "datamover_cmd.h"
#include "hls_stream.h"

    /**
    * @brief Top wrapper of EM module
    * @param[in]    _data           input data
    * @param[in]    _priors         priors of GMM
    * @param[in]    _means          means of GMM
    * @param[in]    _vars           variances of GMM
    * @param[in]    func            0 for kmeans, 1 for GMM
    */
    void top(DATA _data[DATA_NUM*3],
    PRIOR _priors[MAX_MODEL_NUM],
    MEANS _means[MAX_MODEL_NUM*3],
    VARS  _vars[MAX_MODEL_NUM*3],
    ap_uint<1> func,
    int &stat
    );



#endif
