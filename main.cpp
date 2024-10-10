#include <CL/opencl.hpp>
#include <string>
#include "arguments/arguments.hpp"

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
    auto inputImage = static_cast<std::string>(args["-i"]);
    auto outputImage = static_cast<std::string>(args["-o"]);
    auto depthMap = static_cast<std::string>(args["-d"]);
    auto focus = static_cast<float>(args["-f"]);
    auto focusBounds = static_cast<float>(args["-b"]);
    auto strength = static_cast<int>(args["-s"]);

    cl::Context context(CL_DEVICE_TYPE_DEFAULT);

}
