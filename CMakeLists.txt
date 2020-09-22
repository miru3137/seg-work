cmake_minimum_required(VERSION 3.0.0)
project(segwork VERSION 0.1.0 LANGUAGES CXX CUDA)

include(CTest)
enable_testing()

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/modules")

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PythonLibs)
find_package(Freenect2 REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${FREENECT2_INCLUDE_DIRS})

file(GLOB srcs *.cpp)
file(GLOB cuda cuda/*.cu)
file(GLOB cuda_cont cuda/containers/*.cpp)
file(GLOB segm segm/*.cu)
file(GLOB segm_rcnn segm/MaskRCNN/*.cpp)

add_library(segwork SHARED ${srcs} ${cuda} ${cuda_cont} ${segm} ${segm_rcnn})
add_executable(segtest ${srcs} ${cuda} ${cuda_cont} ${segm} ${segm_rcnn})

target_link_libraries(segwork ${CUDA_LIBRARIES} ${OpenCV_LIBS} ${PYTHON_LIBRARIES})
target_link_libraries(segtest ${CUDA_LIBRARIES} ${OpenCV_LIBS} ${PYTHON_LIBRARIES} ${FREENECT2_LIBRARY})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
