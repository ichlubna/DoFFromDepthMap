#include <CL/opencl.hpp>
#include <stdexcept>
#include <string>
#include <fstream>
#include <sstream>
#include <exception>
#include "arguments/arguments.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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
  
    std::cerr << "Loading input image" << std::endl;
    int imageWidth, imageHeight, imageChannels;
    int imageChannelsGPU = 4;
    unsigned char *imageData = stbi_load(params.inputImage.c_str(), &imageWidth, &imageHeight, &imageChannels, imageChannelsGPU);
    if (imageData == nullptr)
        throw std::runtime_error("Failed to load image");
    const char* err = nullptr;
   
    int depthWidth, depthHeight, depthChannels;
    float *depthData = stbi_loadf(params.depthMap.c_str(), &depthWidth, &depthHeight, &depthChannels, 1);
    if (imageData == nullptr)
        throw std::runtime_error("Failed to load depth map");
    
    float maxDepth = 0;
    for(int i = 0; i < depthWidth * depthHeight; i++)
    {
        if(depthData[i] > maxDepth)
            maxDepth = depthData[i];
    }
    float depthDifference = maxDepth-params.focus;    
    float depthLimit = (depthDifference > params.focus) ? depthDifference : params.focus; 

    std::cerr << "Allocating GPU memory" << std::endl;
    const cl::ImageFormat imageFormat(CL_RGBA, CL_UNSIGNED_INT8);
	cl::Image2D inputImageGPU(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imageFormat, imageWidth, imageHeight, 0, imageData);
	cl::Image2D tempImageGPU(context, CL_MEM_READ_WRITE, imageFormat, imageWidth, imageHeight, 0, nullptr);
	cl::Image2D outputImageGPU(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, imageFormat, imageWidth, imageHeight, 0, nullptr);
    const cl::ImageFormat depthFormat(CL_R, CL_FLOAT);
	cl::Image2D inputDepthGPU(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, depthFormat, imageWidth, imageHeight, 0, depthData);

    stbi_image_free(imageData);

    std::cerr << "Processing on GPU" << std::endl;
    auto kernel = cl::compatibility::make_kernel<cl::Image2D&, cl::Image2D&, cl::Image2D&, cl::Image2D&, float>(program, "kernelMain"); 
    cl_int buildErr = CL_SUCCESS; 
    auto buildInfo = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
    for (auto &pair : buildInfo)
        if(!pair.second.empty() && !std::all_of(pair.second.begin(),pair.second.end(),isspace))
            std::cerr << pair.second << std::endl;
    cl::EnqueueArgs kernelArgs(queue, cl::NDRange(imageWidth, imageHeight));
    kernel(kernelArgs, inputImageGPU, tempImageGPU, outputImageGPU, inputDepthGPU, depthLimit);

    std::cerr << "Storing the result" << std::endl;
    cl::array<size_t, 3> origin{0, 0, 0};
    cl::array<size_t, 3> size{static_cast<size_t>(imageWidth), static_cast<size_t>(imageHeight), 1};
    std::vector<unsigned char> outData;
    outData.resize(imageWidth * imageHeight * imageChannelsGPU);
    if(queue.enqueueReadImage(outputImageGPU, CL_TRUE, origin, size, 0, 0, outData.data()) != CL_SUCCESS)
        throw std::runtime_error("Cannot download the result");
    stbi_write_png(params.outputImage.c_str(), imageWidth, imageHeight, imageChannelsGPU, outData.data(), imageWidth * imageChannelsGPU);
}

int main(int argc, char *argv[])
{
    std::string helpText =  "This program takes a depth map, an image and focusing values and simulates a depth of field\n"
                            "--help, -h Prints this help\n"
                            "-i input image - 8-BIT RGBA\n"
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
