float getBlurStrength(float depth, float focus, float bounds, float depthLimit) 
{
    return fmax(0.0f,(fabs(depth-focus)-bounds)/depthLimit);
}

__kernel void kernelMain(__read_only image2d_t inputImage, __write_only image2d_t outputImage, __read_only image2d_t inputDepth, float depthLimit, int strength, float focus, float bounds) 
{
    const sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
    const sampler_t depthSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    int2 coords = (int2)(get_global_id(0), get_global_id(1)); 
	float4 depth = read_imagef(inputDepth, depthSampler, coords);
    float blurStrength = getBlurStrength(depth[0], focus, bounds, depthLimit);

    uint kernelSize = round((strength*2)*blurStrength);
    if (kernelSize%2 == 0)
         kernelSize++;
    const int kernelHalf = kernelSize/2;

    float4 originalPixel = convert_float4(read_imageui(inputImage, imageSampler, coords));
    float weightSum = 1.0f;
    float4 pixelSum = originalPixel;
    for (int x = -kernelHalf; x <= kernelHalf; x++)
    for (int y = -kernelHalf; y <= kernelHalf; y++)
    {
        int2 newCoords = coords + (int2)(x, y);
	    float4 sampleDepth = read_imagef(inputDepth, depthSampler, newCoords);
        float sampleStrength = getBlurStrength(sampleDepth[0], focus, bounds, depthLimit);
        float kernelWeight = 1.0f-(distance(convert_float2(coords), convert_float2(newCoords))/kernelSize);
        float weight = kernelWeight;
        if (sampleStrength < 0.1f)
            weight *= sampleStrength;   
        weightSum += weight;
        pixelSum += convert_float4(read_imageui(inputImage, imageSampler, newCoords))*weight;
    }
    uint4 pixel = convert_uint4(round(pixelSum/weightSum));
	write_imageui(outputImage, coords, pixel);
}
