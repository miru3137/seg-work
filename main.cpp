#include "segm/MaskRCNN/MaskRCNN.h"
#include "segm/MySegmentation.cuh"
#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    // print message
    std::cout << "Hello, world!" << std::endl;

    // read images
    auto rgb = cv::imread("../data/test-rgb.jpg");
    auto gray = cv::imread("../data/test-depth.png", 0);

    // convert to depth data
    cv::Mat_<float> depth;
    gray.convertTo(depth, CV_32F);

    // set frame data
    FrameData frameData;
    frameData.rgb = rgb;
    frameData.depth = depth;

    // run Mask-RCNN
    MaskRCNN mrcnn;
    mrcnn.executeSequential(frameData);

    // set camera intrinsic
    CameraModel instrinsic(574.334300f, 574.334300f, 320.000000f, 240.000000f);

    // run segmentation
    MySegmentation myseg(640, 480, instrinsic);
    auto mask = myseg.performSegmentation(frameData);

    // show images
    cv::imshow("RGB", frameData.rgb);
    cv::imshow("Depth", frameData.depth / 450.f);

    // wait until ESC input
    while (cv::waitKey() != 27);
    cv::destroyAllWindows();

    // exit program
    return 0;
}

extern "C"
void say_hello()
{
    // print message
    std::cout << "Hello, from segwork!\n";
}
