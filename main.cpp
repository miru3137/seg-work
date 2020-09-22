#include "segm/MaskRCNN/MaskRCNN.h"
#include "segm/MySegmentation.cuh"
#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    // print message
    std::cout << "Hello, world!" << std::endl;

    // read images
    auto rgb = cv::imread("../data/02-rgb.jpg", cv::IMREAD_ANYCOLOR);
    auto gray = cv::imread("../data/02-depth.png", cv::IMREAD_ANYDEPTH);

    // convert to color & depth map
    cv::Mat color(rgb.rows, rgb.cols, CV_8UC4);
    cv::Mat depth(gray.rows, gray.cols, CV_32FC1);
    rgb.convertTo(color, CV_8UC4);
    gray.convertTo(depth, CV_32FC1);

    // set frame data
    FrameData frame;
    frame.rgb = color;
    frame.depth = depth / 1000.f;

    // run Mask-RCNN
    MaskRCNN mrcnn;
    mrcnn.executeSequential(frame);

    // set camera intrinsic
    CameraModel instrinsic(574.f, 574.f, 320.f, 240.f);

    // run segmentation
    MySegmentation myseg(frame.depth.cols, frame.depth.rows, instrinsic);
    auto mask = myseg.performSegmentation(frame);

    // display images
    cv::imshow("RGB", frame.rgb);
    cv::imshow("Depth", frame.depth / 100.f);

    // color palette
    const unsigned char colors[31][3] = {
        {0, 0, 0},     {0, 0, 255},     {255, 0, 0},   {0, 255, 0},     {255, 26, 184},  {255, 211, 0},   {0, 131, 246},  {0, 140, 70},
        {167, 96, 61}, {79, 0, 105},    {0, 255, 246}, {61, 123, 140},  {237, 167, 255}, {211, 255, 149}, {184, 79, 255}, {228, 26, 87},
        {131, 131, 0}, {0, 255, 149},   {96, 0, 43},   {246, 131, 17},  {202, 255, 0},   {43, 61, 0},     {0, 52, 193},   {255, 202, 131},
        {0, 43, 96},   {158, 114, 140}, {79, 184, 17}, {158, 193, 255}, {149, 158, 123}, {255, 123, 175}, {158, 8, 0}};
    auto getColor = [&colors](unsigned index) -> cv::Vec3b {
        return (index == 255) ? cv::Vec3b(255, 255, 255) : (cv::Vec3b)colors[index % 31];
    }; // color pick function

    // display mask result before process (use only Mask-RCNN)
    cv::Mat before(frame.rgb.rows, frame.rgb.cols, CV_8UC3);
    for (unsigned i = 0; i < frame.rgb.total(); ++i) {
        before.at<cv::Vec3b>(i) = getColor(frame.mask.data[i]);
        before.at<cv::Vec3b>(i) = 0.5 * before.at<cv::Vec3b>(i) + 0.5 * frame.rgb.at<cv::Vec3b>(i);
    }
    cv::imshow("Mask Before", before);

    // display mask result after process (combine with geometry segmentation)
    cv::Mat after(frame.rgb.rows, frame.rgb.cols, CV_8UC3);
    for (unsigned i = 0; i < frame.rgb.total(); ++i) {
        after.at<cv::Vec3b>(i) = getColor(mask.data[i]);
        after.at<cv::Vec3b>(i) = 0.5 * after.at<cv::Vec3b>(i) + 0.5 * frame.rgb.at<cv::Vec3b>(i);
    }
    cv::imshow("Mask After", after);

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
