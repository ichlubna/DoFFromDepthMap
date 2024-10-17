# Depth-of-field from depth map
This program can be used to compute the depth of field from a depth map. The input is an image and a corresponding depth map plus focusing parameters. The result is the image with applied depth-of-field blur according to the depth and focusing. Use `--help` for the description of the parameters. The program uses OpenCL for GPU acceleration. Make sure to have the `kernel.cl` code in the same directory as the binary.

Example:
```
./DoFFromDepthMap -i inputImage.png -d depthMap.hdr -s 9 -f 0.5 -b 0.1 -o blurred.png
```
