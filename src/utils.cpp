/*
 * Utils source file
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
 
#include "buffer.h"
#include "utils.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>

static const char *SALIENCY_KERNEL_FILE_PATH = "./kernels/saliency.cl";
static const char *REDUCTION_KERNEL_FILE_PATH = "./kernels/reduction.cl";

int read_kernel_from_file(const char* kernel_file_path, char **kernel_buffer)
{
    int file_size = -1;
    FILE * fp = fopen(kernel_file_path, "rb");
    if (!fp)
    {
        std::cerr<<"open file failed:"<<kernel_file_path<<std::endl;
        return -1; 
    }

    if (fseek(fp, 0, SEEK_END) != 0)
    {
        std::cerr<<"fseek file failed:"<<kernel_file_path<<std::endl;
        return -1;
    }

    if ((file_size = ftell(fp)) < 0)
    {
        std::cerr<<"ftell file failed:"<<kernel_file_path<<std::endl;
        return -1;
    }

    rewind(fp);
    if ((*kernel_buffer = (char *)malloc(file_size + 1)) == NULL)
    {
        std::cerr<<"memory allocated error"<<std::endl;
        return -1; 
    }
    
    fread((void*)*kernel_buffer, 1, file_size, fp);
    fclose(fp);
    (*kernel_buffer)[file_size] = '\0';
    return file_size;
}

bool init_gpu_cl(cl_device_id & device, cl_context & context, cl_command_queue & queue)
{
    cl_int                 err;
    cl_device_type      device_type = CL_DEVICE_TYPE_GPU;
    cl_platform_id platform = NULL;
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS)
    {
        return false;
    }

    err = clGetDeviceIDs(platform, device_type, 1, &device, NULL);
    if (err != CL_SUCCESS)
    {
        return false;
    }

    context = clCreateContext(0, 1, &device, NULL, NULL, &err);
    if (!context || err != CL_SUCCESS)
    {
        return false;
    }

    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (!queue || err != CL_SUCCESS)
    {
        return false;
    }
    return true;
}

bool init_global_clbuffer(CLBuffer & buffer, cl_device_id & device, cl_context & context)
{
    cl_int error;
    buffer.climg_format.image_channel_order = CL_R;
    buffer.climg_format.image_channel_data_type = CL_UNORM_INT8;
    buffer.climg_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    buffer.climg_desc.image_width = config::IMAGE_COLS;
    buffer.climg_desc.image_height = config::IMAGE_ROWS;
    buffer.climg_desc.image_depth = 0;
    buffer.climg_desc.image_array_size = 0;
    buffer.climg_desc.image_row_pitch = 0;
    buffer.climg_desc.image_slice_pitch = 0;
    buffer.climg_desc.num_mip_levels = 0;
    buffer.climg_desc.num_samples = 0;
    buffer.climg_desc.buffer = 0;
    buffer.cl_bimg = clCreateImage(context, CL_MEM_READ_ONLY, &(buffer.climg_format), &(buffer.climg_desc), NULL, &error);
    buffer.cl_gimg = clCreateImage(context, CL_MEM_READ_ONLY, &(buffer.climg_format), &(buffer.climg_desc), NULL, &error);
    buffer.cl_rimg = clCreateImage(context, CL_MEM_READ_ONLY, &(buffer.climg_format), &(buffer.climg_desc), NULL, &error);

    buffer.cl_Limg = clCreateImage(context, CL_MEM_WRITE_ONLY, &(buffer.climg_format), &(buffer.climg_desc), NULL, &error);
    buffer.cl_Aimg = clCreateImage(context, CL_MEM_WRITE_ONLY, &(buffer.climg_format), &(buffer.climg_desc), NULL, &error);
    buffer.cl_Bimg = clCreateImage(context, CL_MEM_WRITE_ONLY, &(buffer.climg_format), &(buffer.climg_desc), NULL, &error);

    buffer.cl_sampler_ = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &error);
    if (error != CL_SUCCESS)
    {
        std::cerr<<"clCreateSampler() failed. error code:"<<error<<std::endl;
        return false;
    }
    
    
    char* saliency_kernel_source = 0;
    int saliency_kernel_size = read_kernel_from_file(SALIENCY_KERNEL_FILE_PATH, &saliency_kernel_source);
    if (saliency_kernel_size <=0)
    {
        return false;
    }

    buffer.cl_saliency_program = clCreateProgramWithSource(context, 1, (const char **)&saliency_kernel_source, (const size_t *)&saliency_kernel_size, &error);
    if (error != CL_SUCCESS)
    {
        std::cerr<<"clCreateProgramWithSource() failed. error code:"<<error<<std::endl;
        free(saliency_kernel_source);
        return false;
    }
    clBuildProgram(buffer.cl_saliency_program, 1, &device, 0, 0, 0);
    buffer.cl_saliency_kernel = clCreateKernel(buffer.cl_saliency_program, "saliency", &error);
    if (error != CL_SUCCESS)
    {
        std::cerr<<"clCreateKernel() failed. error code:"<<error<<std::endl;
        free(saliency_kernel_source);
        return false;
    }

    char* reduction_kernel_source = 0;
    int reduction_kernel_size = read_kernel_from_file(REDUCTION_KERNEL_FILE_PATH, &reduction_kernel_source);
    if(reduction_kernel_size <=0)
    {
        free(saliency_kernel_source);
        return false; 
    }

    buffer.cl_reduction_program =  clCreateProgramWithSource(context, 1, (const char **)&reduction_kernel_source, (const size_t *) &reduction_kernel_size, &error);
    if (error != CL_SUCCESS)
    {
        std::cerr<<"clCreateProgramWithSource() failed. error code:"<<error<<std::endl;
        free(saliency_kernel_source);
        free(reduction_kernel_source);
        return false;
    }
    clBuildProgram(buffer.cl_reduction_program, 1, &device, 0, 0, 0);
    buffer.cl_reduction_kernel = clCreateKernel(buffer.cl_reduction_program, "reduction", &error);
    if (error != CL_SUCCESS)
    {
        std::cerr<<"clCreateKernel() failed. error code:"<<error<<std::endl;
        free(saliency_kernel_source);
        free(reduction_kernel_source);
        return false;
    }
    
    buffer.cl_saliency_origin[0] = 0;
    buffer.cl_saliency_origin[1] = 0;
    buffer.cl_saliency_origin[2] = 0;
    buffer.cl_saliency_region[0] = config::IMAGE_COLS;
    buffer.cl_saliency_region[1] = config::IMAGE_ROWS;
    buffer.cl_saliency_region[2] = 1; 
    buffer.cl_saliency_global_work_size[0] = config::IMAGE_COLS;
    buffer.cl_saliency_global_work_size[1] = config::IMAGE_ROWS;
    
    buffer.cl_reduction_global_work_size[0] = 512; 
    buffer.cl_reduction_local_work_size[0] = 64; 
    buffer.cl_groups = 8;
    buffer.loutput_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, buffer.cl_groups * 4 * sizeof(unsigned int), NULL, NULL);
    buffer.aoutput_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, buffer.cl_groups * 4 * sizeof(unsigned int), NULL, NULL);
    buffer.boutput_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, buffer.cl_groups * 4 * sizeof(unsigned int), NULL, NULL);
    free(saliency_kernel_source);
    free(reduction_kernel_source);
    return true;

}
