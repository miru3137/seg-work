/*
 * This file is part of ElasticFusion.
 *
 * Copyright (C) 2015 Imperial College London
 * 
 * The use of the code within this file and all code within files that 
 * make up the software that is ElasticFusion is permitted for 
 * non-commercial purposes only.  The full terms and conditions that 
 * apply to the code within this file are detailed within the LICENSE.txt 
 * file and at <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/elastic-fusion/elastic-fusion-license/> 
 * unless explicitly stated.  By downloading this file you agree to 
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then 
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Author: Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
 */

#include "cudafuncs.cuh"
#include "convenience.cuh"
#include "operators.cuh"

// Generate a vertex map 'vmap' based on the depth map 'depth' and camera parameters
__global__ void computeVmapKernel(const PtrStepSz<float> depth, PtrStep<float> vmap, float fx_inv, float fy_inv, float cx, float cy, float depthCutoff)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;

    if(u < depth.cols && v < depth.rows)
    {
        float z = depth.ptr (v)[u];

        if(z > 0.0f && z < depthCutoff)
        {
            float vx = z * (u - cx) * fx_inv;
            float vy = z * (v - cy) * fy_inv;
            float vz = z;

            vmap.ptr (v                 )[u] = vx;
            vmap.ptr (v + depth.rows    )[u] = vy;
            vmap.ptr (v + depth.rows * 2)[u] = vz;
        }
        else
        {
            vmap.ptr (v)[u] = __int_as_float(0x7fffffff); /*CUDART_NAN_F*/
            vmap.ptr (v + depth.rows * 2)[u] = 0;
        }
    }
}

void createVMap(const CameraModel& intr, const DeviceArray2D<float> & depth, DeviceArray2D<float> & vmap, const float depthCutoff)
{
    vmap.create (depth.rows () * 3, depth.cols ());

    dim3 block (32, 8);
    dim3 grid (1, 1, 1);
    grid.x = getGridDim (depth.cols (), block.x);
    grid.y = getGridDim (depth.rows (), block.y);

    float fx = intr.fx, cx = intr.cx;
    float fy = intr.fy, cy = intr.cy;

    computeVmapKernel<<<grid, block>>>(depth, vmap, 1.f / fx, 1.f / fy, cx, cy, depthCutoff);
    cudaSafeCall (cudaGetLastError ());
    cudaSafeCall (cudaDeviceSynchronize ());
}

__global__ void computeNmapKernel(int rows, int cols, const PtrStep<float> vmap, PtrStep<float> nmap)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;

    if (u >= cols || v >= rows)
        return;

    if (u == cols - 1 || v == rows - 1)
    {
        nmap.ptr (v)[u] = __int_as_float(0x7fffffff); /*CUDART_NAN_F*/
        return;
    }

    float3 v00, v01, v10;
    v00.x = vmap.ptr (v  )[u];
    v01.x = vmap.ptr (v  )[u + 1];
    v10.x = vmap.ptr (v + 1)[u];

    if (!isnan (v00.x) && !isnan (v01.x) && !isnan (v10.x))
    {
        v00.y = vmap.ptr (v + rows)[u];
        v01.y = vmap.ptr (v + rows)[u + 1];
        v10.y = vmap.ptr (v + 1 + rows)[u];

        v00.z = vmap.ptr (v + 2 * rows)[u];
        v01.z = vmap.ptr (v + 2 * rows)[u + 1];
        v10.z = vmap.ptr (v + 1 + 2 * rows)[u];

        float3 r = normalized (cross (v01 - v00, v10 - v00));

        nmap.ptr (v       )[u] = r.x;
        nmap.ptr (v + rows)[u] = r.y;
        nmap.ptr (v + 2 * rows)[u] = r.z;
    }
    else
        nmap.ptr (v)[u] = __int_as_float(0x7fffffff); /*CUDART_NAN_F*/
}

void createNMap(const DeviceArray2D<float>& vmap, DeviceArray2D<float>& nmap)
{
    nmap.create (vmap.rows (), vmap.cols ());

    int rows = vmap.rows () / 3;
    int cols = vmap.cols ();

    dim3 block (32, 8);
    dim3 grid (1, 1, 1);
    grid.x = getGridDim (cols, block.x);
    grid.y = getGridDim (rows, block.y);

    computeNmapKernel<<<grid, block>>>(rows, cols, vmap, nmap);
    cudaSafeCall (cudaGetLastError ());
    cudaSafeCall (cudaDeviceSynchronize ());
}
