__kernel void reduction(__global uchar4* Limg, __global uint4* L_reduce, 
                        __global uchar4* Aimg, __global uint4* A_reduce, 
                        __global uchar4* Bimg, __global uint4* B_reduce,
                        int image_pixels)
{
    image_pixels = image_pixels / 4;
    unsigned int tid = get_local_id(0);
    unsigned int local_size = get_local_size(0);
    unsigned int global_size = get_global_size(0);

    uint4 L_pixels = (uint4) { 0, 0, 0, 0 };
    uint4 A_pixels = (uint4) { 0, 0, 0, 0 };
    uint4 B_pixels = (uint4) { 0, 0, 0, 0 };
       
    __local uint4 L_cache[64];
    __local uint4 A_cache[64];
    __local uint4 B_cache[64];

    unsigned int i = get_global_id(0);
    while (i < image_pixels)
    {
        L_pixels += convert_uint4(Limg[i]);
        A_pixels += convert_uint4(Aimg[i]);
        B_pixels += convert_uint4(Bimg[i]);
        i += global_size;
    }

    L_cache[tid] = L_pixels;
    A_cache[tid] = A_pixels;
    B_cache[tid] = B_pixels;
    barrier(CLK_LOCAL_MEM_FENCE);

    // do reduction in shared mem
    for (unsigned int s = local_size >> 1; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            L_cache[tid] += L_cache[tid + s];
            A_cache[tid] += A_cache[tid + s];
            B_cache[tid] += B_cache[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0)
    {
        L_reduce[get_group_id(0)] = L_cache[0];
        A_reduce[get_group_id(0)] = A_cache[0];
        B_reduce[get_group_id(0)] = B_cache[0];
    }
}
