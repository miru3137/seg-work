#include "segm/MaskRCNN/MaskRCNN.h"
#include "segm/MySegmentation.cuh"
#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    // print message
    std::cout << "Hello, world!" << std::endl;

    // read images
    auto rgb = cv::imread("../data/01-rgb.jpg");
    auto gray = cv::imread("../data/01-depth.png", 0);

    // convert to color & depth map
    cv::Mat color(rgb.rows, rgb.cols, CV_8UC4);
    cv::Mat depth(gray.rows, gray.cols, CV_32FC1);
    rgb.convertTo(color, CV_8UC4);
    gray.convertTo(depth, CV_32FC1);

    // set frame data
    FrameData frame;
    frame.rgb = color;
    frame.depth = depth;

    // run Mask-RCNN
    MaskRCNN mrcnn;
    mrcnn.executeSequential(frame);

    // set camera intrinsic
    CameraModel instrinsic(574.f, 574.f, 320.f, 240.f);

    // run segmentation
    MySegmentation myseg(frame.depth.cols, frame.depth.rows, instrinsic);
    auto mask = myseg.performSegmentation(frame);

    // show images
    cv::imshow("RGB", frame.rgb);
    cv::imshow("Depth", frame.depth / 450.f);

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
