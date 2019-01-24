/*
 * Defect-Detection native API header
 *
 * Copyright (c) 2018-2028 chenzhengqiang 642346572@qq.com 
 * All rights reserved since 2018-12-14
 *
 * Redistribution and use in source and binary forms, with or without modifica-
 * tion, are permitted provided that the following conditions are met:
 *
 *   1.  Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimer.
 *
 *   2.  Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MER-
 * CHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO
 * EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPE-
 * CIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTH-
 * ERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Alternatively, the contents of this file may be used under the terms of
 * the GNU General Public License ("GPL") version 2 or any later version,
 * in which case the provisions of the GPL are applicable instead of
 * the above. If you wish to allow the use of your version of this file
 * only under the terms of the GPL and not to allow others to use your
 * version of this file under the BSD license, indicate your decision
 * by deleting the provisions above and replace them with the notice
 * and other provisions required by the GPL. If you do not delete the
 * provisions above, a recipient may use your version of this file under
 * either the BSD or the GPL.
 */


#ifndef _BACKER_OPENCL_SALIENCY_BUFFER_H_
#define _BACKER_OPENCL_SALIENCY_BUFFER_H_

#include "config.h"
#include "opencl_common.h"

struct CLBuffer
{
    unsigned char  bimg_buffer[config::IMAGE_PIXELS];
    unsigned char  gimg_buffer[config::IMAGE_PIXELS];
    unsigned char  rimg_buffer[config::IMAGE_PIXELS];
    unsigned int reduce_l[config::REDUCE_BLOCKS];
    unsigned int reduce_a[config::REDUCE_BLOCKS];
    unsigned int reduce_b[config::REDUCE_BLOCKS];
    cl_image_format climg_format;
    cl_image_desc climg_desc;
    cl_mem cl_bimg;
    cl_mem cl_gimg;
    cl_mem cl_rimg;
    cl_mem cl_Limg;
    cl_mem cl_Aimg;
    cl_mem cl_Bimg;

    cl_mem loutput_buffer;
    cl_mem aoutput_buffer;
    cl_mem boutput_buffer;

    cl_sampler cl_sampler_;
    cl_program cl_bgr2lab_program;
    cl_kernel cl_bgr2lab_kernel;
    cl_program cl_reduction_program;
    cl_kernel cl_reduction_kernel;
    size_t cl_bgr2lab_origin[3];
    size_t cl_bgr2lab_region[3];
    size_t cl_bgr2lab_global_work_size[2];
    size_t cl_reduction_global_work_size[1];
    size_t cl_reduction_local_work_size[1];
    size_t cl_groups;
    
};
#endif
