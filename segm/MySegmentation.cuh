/*
 * This file is part of https://github.com/martinruenz/maskfusion
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>
 */

#pragma once

#include "../cuda/containers/device_array.hpp"
#include "../cuda/types.cuh"
#include "../util/BoundingBox.h"
#include "../util/FrameData.h"

#include <list>
#include <queue>
#include <map>
#include <memory>

class MySegmentation {
public:

 // Parameters
  float bilatSigmaDepth = 3;
  float bilatSigmaColor = 8;
  float bilatSigmaLocation = 2;
  int bilatSigmaRadius = 2;

  float nonstaticThreshold = 0.4;
  float threshold = 0.1;
  float weightConvexity = 1;
  float weightDistance = 1;
  int morphEdgeIterations = 3;
  int morphEdgeRadius = 1;
  int morphMaskIterations = 3;
  int morphMaskRadius = 1;

  bool removeEdges = true; // These two are exclusive (otherwise wasting computations)
  bool removeEdgeIslands = false;
  const int minMappedComponentSize;

  int personClassID = 255;

 public:
  MySegmentation(int w,
                 int h,
                 const CameraModel& cameraIntrinsics);
  virtual ~MySegmentation();


  cv::Mat performSegmentation(FrameData& frame);

  void computeLookups(FrameData& frame);

 private:
  // CPU buffers for internal use
  cv::Mat cv8UC1Buffer;
  cv::Mat cvLabelComps;
  cv::Mat cvLabelEdges;
  unsigned char maskToID[256];
  cv::Mat semanticIgnoreMap;

  // Buffers for internal use
  DeviceArray2D<float> floatEdgeMap;
  DeviceArray2D<float> floatBuffer;
  DeviceArray2D<unsigned char> binaryEdgeMap;
  DeviceArray2D<unsigned char> ucharBuffer;

  // 3D-data used for tracking is filtered too heavily, hence recompute here
  DeviceArray2D<float> vertexMap;
  DeviceArray2D<float> normalMap;
  DeviceArray2D<float> depthMapMetric;
  DeviceArray2D<float> depthMapMetricFiltered;
  DeviceArray2D<uchar4> rgb;
  CameraModel cameraIntrinsics;
};
