__kernel void bgr2lab(__read_only image2d_t Bimg_, __read_only image2d_t Gimg_, __read_only image2d_t Rimg_, 
                       __write_only image2d_t Limg, __write_only image2d_t Aimg, __write_only image2d_t Bimg,
                       sampler_t sampler)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
   
    float4 sB = read_imagef(Bimg_, sampler, (int2)(x, y));
    float4 sG = read_imagef(Gimg_, sampler, (int2)(x, y));
    float4 sR = read_imagef(Rimg_, sampler, (int2)(x, y));
    
    float B = sB.x/255.0f;
    float G = sG.x/255.0f;
    float R = sR.x/255.0f;

    float b = 0.0f;
    float g = 0.0f;
    float r = 0.0f;

    if(R <= 0.04045f) r = R/12.92f;
    else              r = pow((R+0.055f)/1.055f, 2.4f);
    if(G <= 0.04045f) g = G/12.92f;
    else              g = pow((G+0.055f)/1.055f, 2.4f);
    if(B <= 0.04045f) b = B/12.92f;
    else	      b = pow((B+0.055f)/1.055f, 2.4f);
    
    float X = r*0.4124564f + g*0.3575761f + b*0.1804375f;
    float Y = r*0.2126729f + g*0.7151522f + b*0.0721750f;
    float Z = r*0.0193339f + g*0.1191920f + b*0.9503041f;
    
    float epsilon = 0.008856f;
    float kappa   = 903.3f;
    
    float Xr = 0.950456f;
    float Yr = 1.0f;
    float Zr = 1.088754f;

    float xr = X/Xr;
    float yr = Y/Yr;
    float zr = Z/Zr;

    float fx = 0.0f;
    float fy = 0.0f;
    float fz = 0.0f;
    
    if(xr > epsilon) fx = pow(xr, 1.0f/3.0f);
    else fx = (kappa*xr + 16.0f)/116.0f;
    if(yr > epsilon) fy = pow(yr, 1.0f/3.0f);
    else fy = (kappa*yr + 16.0f)/116.0f;
    if(zr > epsilon) fz = pow(zr, 1.0f/3.0f);
    else fz = (kappa*zr + 16.0f)/116.0f;

    sR.x =  yr > epsilon ? (116.0*fy-16.0):(yr*kappa);
    sG.x =  500.0f * (fx-fy);
    sB.x =  200.0f * (fy-fz);

    write_imagef(Limg, (int2)(x, y), sR);
    write_imagef(Aimg, (int2)(x, y), sG);
    write_imagef(Bimg, (int2)(x, y), sB);
}

