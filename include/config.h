/*
@author:chenzhengqiang
@date:2019-01-23
*/

#ifndef _OPENCL_SALIENCY_CONFIG_H_
#define _OPENCL_SALIENCY_CONFIG_H_

namespace config
{
    /*
    @note:sample image's rows is 160, cols 1280
    */
    static const int IMAGE_ROWS = 640;
    static const int IMAGE_COLS = 640;
    static const int IMAGE_PIXELS = IMAGE_ROWS * IMAGE_COLS;

    static const int CL_REDUCTION_GLOBAL_WORK_SIZE = 640;
    static const int CL_REDUCTION_LOCAL_WORK_SIZE = 64;
    static const int CL_REDUCTION_GROUPS = 640 / 64;

    /*
    @note:using 32 blocks to calculate image's sum with reduction algorithm
   */
    static const int REDUCE_BLOCKS =  (640 / 64) * 4;
}

#endif
