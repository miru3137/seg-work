cmake_minimum_required(VERSION 3.0.0)
project(segwork VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB srcs *.cpp *.cu)
file(GLOB cuda Cuda/*.cu)

add_library(segwork SHARED ${srcs} ${cuda})

target_link_libraries(segwork ${CUDA_LIBRARIES} ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
