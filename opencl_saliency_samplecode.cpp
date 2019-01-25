/*
 * defect detect with opencl's sample code
 *
 * Copyright (c) 2018-2028 chenzhengqiang 642346572@qq.com 
 * All rights reserved since 2019-01-18
 *
 * Redistribution and use in source and binary forms, with or without modifica-
 * tIOn, are permitted provided that the following conditions are met:
 *
 *   1.  Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimer.
 *
 *   2.  RedistributIOns in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentatIOn and/or other materials provided with the distributIOn.
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
 * the GNU General Public License ("GPL") versIOn 2 or any later versIOn,
 * in which case the provisIOns of the GPL are applicable instead of
 * the above. If you wish to allow the use of your versIOn of this file
 * only under the terms of the GPL and not to allow others to use your
 * versIOn of this file under the BSD license, indicate your decisIOn
 * by deleting the provisIOns above and replace them with the notice
 * and other provisIOns required by the GPL. If you do not delete the
 * provisIOns above, a recipient may use your versIOn of this file under
 * either the BSD or the GPL.
 */


#include "config.h"
#include "buffer.h"
#include "utils.h"
#include "opencl_common.h"
#include "opencv_common.h"

#include <utility>
#include <map>
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <unistd.h>

CLBuffer CL_BUFFER;


cv::Mat opencl_calc_saliency_with_ft_algorithm(cv::Mat & sample_image, cl_context & context, cl_command_queue & queue);

int main(int argc, char ** argv)
{
    if (argc != 2)
    {
        std::cerr<<"you must specify one input image"<<std::endl;
        return EXIT_FAILURE;
    }

    Mat sample_image = imread(argv[1]);
    if(sample_image.rows > config::IMAGE_ROWS || sample_image.cols > config::IMAGE_COLS)
    {
        std::cerr<<"imput image's rows or cols must less than or equal to which you confiured";
        return EXIT_FAILURE; 
    }

    cl_device_id        device;
    cl_context          context;
    cl_command_queue    queue;
    bool ok = init_gpu_cl(device, context, queue);
    if(!ok)
    {
        std::cerr<<"open gpu device with opencl failed."<<std::endl;
        return EXIT_FAILURE;
    }

    ok = init_global_clbuffer(CL_BUFFER, device, context);
    if (!ok)
    {
        std::cerr<<"initialize global opencl buffer failed."<<std::endl;
        return EXIT_FAILURE;
    }

    cv::Mat saliency_image = opencl_calc_saliency_with_ft_algorithm(sample_image, context, queue);
    imwrite("result.jpg", saliency_image);
    free_global_clbuffer(CL_BUFFER);

    return EXIT_SUCCESS;
}

cv::Mat  opencl_calc_saliency_with_ft_algorithm(cv::Mat & sample_image, cl_context & context, cl_command_queue & queue)
{
    Mat channels[3];
    split(sample_image, channels);

    for (int index = 0; index < config::IMAGE_PIXELS; ++index)
    {
        CL_BUFFER.bimg_buffer[index] = channels[0].data[index];
        CL_BUFFER.gimg_buffer[index] = channels[1].data[index];
        CL_BUFFER.rimg_buffer[index] = channels[2].data[index];
    }

    clSetKernelArg(CL_BUFFER.cl_saliency_kernel, 0, sizeof(cl_mem), &(CL_BUFFER.cl_bimg));
    clSetKernelArg(CL_BUFFER.cl_saliency_kernel, 1, sizeof(cl_mem), &(CL_BUFFER.cl_gimg));
    clSetKernelArg(CL_BUFFER.cl_saliency_kernel, 2, sizeof(cl_mem), &(CL_BUFFER.cl_rimg));
    clSetKernelArg(CL_BUFFER.cl_saliency_kernel, 3, sizeof(cl_mem), &(CL_BUFFER.cl_Limg));
    clSetKernelArg(CL_BUFFER.cl_saliency_kernel, 4, sizeof(cl_mem), &(CL_BUFFER.cl_Aimg));
    clSetKernelArg(CL_BUFFER.cl_saliency_kernel, 5, sizeof(cl_mem), &(CL_BUFFER.cl_Bimg));
    clSetKernelArg(CL_BUFFER.cl_saliency_kernel, 6, sizeof(cl_sampler), &(CL_BUFFER.cl_sampler_));
    
    clEnqueueWriteImage(queue, CL_BUFFER.cl_bimg, CL_TRUE, CL_BUFFER.cl_saliency_origin, CL_BUFFER.cl_saliency_region, 0, 0, CL_BUFFER.bimg_buffer, 0, 0, 0);
    clEnqueueWriteImage(queue, CL_BUFFER.cl_gimg, CL_TRUE, CL_BUFFER.cl_saliency_origin, CL_BUFFER.cl_saliency_region, 0, 0, CL_BUFFER.gimg_buffer, 0, 0, 0);
    clEnqueueWriteImage(queue, CL_BUFFER.cl_rimg, CL_TRUE, CL_BUFFER.cl_saliency_origin, CL_BUFFER.cl_saliency_region, 0, 0, CL_BUFFER.rimg_buffer, 0, 0, 0);
    clEnqueueNDRangeKernel(queue, CL_BUFFER.cl_saliency_kernel, 2, 0, CL_BUFFER.cl_saliency_global_work_size, 0, 0, 0, 0);
    clEnqueueReadImage(queue, CL_BUFFER.cl_Limg, CL_TRUE, CL_BUFFER.cl_saliency_origin, CL_BUFFER.cl_saliency_region, 0, 0, CL_BUFFER.bimg_buffer, 0, NULL, NULL);
    clEnqueueReadImage(queue, CL_BUFFER.cl_Aimg, CL_TRUE, CL_BUFFER.cl_saliency_origin, CL_BUFFER.cl_saliency_region, 0, 0, CL_BUFFER.gimg_buffer, 0, NULL, NULL);
    clEnqueueReadImage(queue, CL_BUFFER.cl_Bimg, CL_TRUE, CL_BUFFER.cl_saliency_origin, CL_BUFFER.cl_saliency_region, 0, 0, CL_BUFFER.rimg_buffer, 0, NULL, NULL);

    cl_mem linput_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, config::IMAGE_PIXELS, CL_BUFFER.bimg_buffer, NULL);
    cl_mem ainput_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, config::IMAGE_PIXELS, CL_BUFFER.gimg_buffer, NULL);
    cl_mem binput_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, config::IMAGE_PIXELS,  CL_BUFFER.rimg_buffer, NULL);

    clSetKernelArg(CL_BUFFER.cl_reduction_kernel, 0, sizeof(cl_mem), (void *)&linput_buffer);
    clSetKernelArg(CL_BUFFER.cl_reduction_kernel, 1, sizeof(cl_mem), (void *)&(CL_BUFFER.loutput_buffer));
    clSetKernelArg(CL_BUFFER.cl_reduction_kernel, 2, sizeof(cl_mem), (void *)&ainput_buffer);
    clSetKernelArg(CL_BUFFER.cl_reduction_kernel, 3, sizeof(cl_mem), (void *)&(CL_BUFFER.aoutput_buffer));
    clSetKernelArg(CL_BUFFER.cl_reduction_kernel, 4, sizeof(cl_mem), (void *)&binput_buffer);
    clSetKernelArg(CL_BUFFER.cl_reduction_kernel, 5, sizeof(cl_mem), (void *)&(CL_BUFFER.boutput_buffer));
    clSetKernelArg(CL_BUFFER.cl_reduction_kernel, 6, sizeof(int), &config::IMAGE_PIXELS);

    clEnqueueNDRangeKernel(queue, CL_BUFFER.cl_reduction_kernel, 1, NULL, CL_BUFFER.cl_reduction_global_work_size, CL_BUFFER.cl_reduction_local_work_size, 0, 0, 0);
    size_t reduce_buffer_bytes = CL_BUFFER.cl_groups * 4 * sizeof(unsigned int);
    clEnqueueReadBuffer(queue, CL_BUFFER.loutput_buffer, CL_TRUE, 0, reduce_buffer_bytes, CL_BUFFER.reduce_l, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, CL_BUFFER.aoutput_buffer, CL_TRUE, 0, reduce_buffer_bytes, CL_BUFFER.reduce_a, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, CL_BUFFER.boutput_buffer, CL_TRUE, 0, reduce_buffer_bytes, CL_BUFFER.reduce_b, 0, NULL, NULL);
    
    unsigned int lsum = 0;
    unsigned int asum = 0;
    unsigned int bsum = 0;

    for (int i = 0; i < config::REDUCE_BLOCKS; i++)
    {
        lsum += CL_BUFFER.reduce_l[i];
        asum += CL_BUFFER.reduce_a[i];
        bsum += CL_BUFFER.reduce_b[i];
    }

    float L_mean = (float) lsum / config::IMAGE_PIXELS;
    float A_mean = (float) asum / config::IMAGE_PIXELS;
    float B_mean = (float) bsum / config::IMAGE_PIXELS;
    
    for (int i = 0; i < config::IMAGE_PIXELS; ++i)
    {
        float bdiff = (CL_BUFFER.bimg_buffer[i] > L_mean) ? CL_BUFFER.bimg_buffer[i] - L_mean : L_mean - CL_BUFFER.bimg_buffer[i];
        float gdiff = (CL_BUFFER.gimg_buffer[i] > A_mean) ? CL_BUFFER.gimg_buffer[i] - A_mean : A_mean - CL_BUFFER.gimg_buffer[i];
        float rdiff = (CL_BUFFER.rimg_buffer[i] > B_mean) ? CL_BUFFER.rimg_buffer[i] - B_mean : B_mean - CL_BUFFER.rimg_buffer[i];
        CL_BUFFER.bimg_buffer[i] = static_cast<unsigned char>(bdiff * bdiff + gdiff*gdiff + rdiff * rdiff);
    }

    memcpy(channels[0].data, CL_BUFFER.bimg_buffer, config::IMAGE_PIXELS);
    cv::GaussianBlur(channels[0], channels[0], Size(7,7),0,0);
    return channels[0];
}
