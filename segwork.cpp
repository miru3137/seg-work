#include "MySegmentation.cuh"
#include <iostream>
 
extern "C"
void say_hello()
{
    CameraModel model;
    MySegmentation myseg(640, 480, model);

    std::cout << "Hello, from segwork!\n";
}
