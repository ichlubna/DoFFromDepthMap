#include <CL/opencl.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <exception>
#include "arguments/arguments.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define TINYEXR_USE_STB_ZLIB 1
#define TINYEXR_USE_MINIZ 0
#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

class Params
{
    public:
    std::string inputImage;
    std::string outputImage;
    std::string depthMap;
    float focus;
    float focusBounds;
    int strength;
};

void process(Params params)
{
    std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0)
        throw std::runtime_error("No OpenCL platforms available");
    cl::Context context(CL_DEVICE_TYPE_DEFAULT);
    std::ifstream file("kernel.cl");
    std::stringstream kernelContent;
    kernelContent << file.rdbuf();
    file.close();
    cl::Program program(context, kernelContent.str(), true);
    cl::CommandQueue queue(context);
   
    int imageWidth, imageHeight, imageChannels;
    unsigned char *imageData = stbi_load(params.inputImage.c_str(), &imageWidth, &imageHeight, &imageChannels, 0);
    if (imageData == nullptr)
        throw std::runtime_error("Failed to load image");
    int depthWidth, depthHeight;
    const char* err = nullptr;
    /*
    float* depthData;
    if(LoadEXR(&depthData, &depthWidth, &depthHeight, params.depthMap.c_str(), &err) != TINYEXR_SUCCESS)
    {
        FreeEXRErrorMessage(err);
        throw std::runtime_error("Failed to load depth map");
    }
    if(imageWidth != depthWidth || imageHeight != depthHeight)
        throw std::runtime_error("Image and depth map must be the same size");
    */

    EXRHeader exrHeader;
    EXRVersion exrVersion;
    InitEXRHeader(&exrHeader);
    ParseEXRHeaderFromFile(&exrHeader, &exrVersion, params.depthMap.c_str(), &err);
    for (int i = 0; i < exrHeader.num_channels; i++)
        if (exrHeader.pixel_types[i] == TINYEXR_PIXELTYPE_HALF) 
            exrHeader.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
    EXRImage exr_image;
    InitEXRImage(&exr_image);
    if(LoadEXRImageFromFile(&exr_image, &exrHeader,params.depthMap.c_str(), &err) != TINYEXR_SUCCESS)
    {
        FreeEXRErrorMessage(err);
        throw std::runtime_error("Failed to load depth map");
    }
    std::vector<float> depthData(exr_image.width * exr_image.height);
    for(size_t i = 0; i < exr_image.width * exr_image.height * exr_image.num_channels; i += exr_image.num_channels)
        depthData[i] = exr_image.images[0][i];

    const cl::ImageFormat imageFormat(CL_RGB, CL_UNSIGNED_INT8);
	cl::Image2D inputImageGPU(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imageFormat, imageWidth, imageHeight, 0, imageData);
	cl::Image2D tempImageGPU(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, imageFormat, imageWidth, imageHeight, 0, nullptr);
	cl::Image2D outputImageGPU(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, imageFormat, imageWidth, imageHeight, 0, nullptr);
    const cl::ImageFormat depthFormat(CL_R, CL_FLOAT);
	cl::Image2D inputDepthGPU(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, depthFormat, imageWidth, imageHeight, 0, depthData.data());

    stbi_image_free(imageData);

    auto kernel = cl::compatibility::make_kernel<>(program, "kernel");
    kernel(cl::EnqueueArgs(queue, cl::NDRange(imageWidth, imageHeight)));
    queue.finish();

    cl::array<size_t, 3> origin{0, 0, 0};
    cl::array<size_t, 3> size{static_cast<size_t>(imageWidth), static_cast<size_t>(imageHeight), 1};
    std::vector<unsigned char> outData(imageWidth * imageHeight * imageChannels);
    queue.enqueueReadImage(outputImageGPU, CL_TRUE, origin, size, 0, 0, outData.data());
    stbi_write_png(params.outputImage.c_str(), imageWidth, imageHeight, imageChannels, outData.data(), imageWidth * imageChannels);
}

int main(int argc, char *argv[])
{
    std::string helpText =  "This program takes a depth map, an image and focusing values and simulates a depth of field\n"
                            "--help, -h Prints this help\n"
                            "-i input image\n"
                            "-o output image\n"
                            "-d input depth map\n"
                            "-f foucus distance in the same units as values in the depth map\n"
                            "-b focus bounds - how much around the focus distance is to stay focused \n"
                            "-s blur distance in pixels \n";
    Arguments args(argc, argv);
    if(args.printHelpIfPresent(helpText))
        return 0;
    if(argc < 2)
    {
        std::cerr << "Use --help" << std::endl;
        return 0;
    }

    Params params;
    params.inputImage = static_cast<std::string>(args["-i"]);
    params.outputImage = static_cast<std::string>(args["-o"]);
    params.depthMap = static_cast<std::string>(args["-d"]);
    params.focus = static_cast<float>(args["-f"]);
    params.focusBounds = static_cast<float>(args["-b"]);
    params.strength = static_cast<int>(args["-s"]);

    try
    {
        process(params);
    }

    catch(const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
