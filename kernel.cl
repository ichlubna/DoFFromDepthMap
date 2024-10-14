__kernel void kernelMain(__read_only image2d_t inputImage, __read_write image2d_t temp, __write_only image2d_t outputImage, __read_only image2d_t inputDepth, float depthLimit) 
{
    const sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
    const sampler_t depthSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    int2 coords = (int2)(get_global_id(0), get_global_id(1)); 
	float4 depth = read_imagef(inputDepth, depthSampler, coords);
	//uint4 pixel = read_imageui(inputImage, imageSampler, coords);
    float blurStrength = depth[0]/depthLimit;
    uint t = round(blurStrength*255);
    uint4 pixel = (uint4)(t,t,t,255);
	write_imageui(outputImage, coords, pixel);
}
