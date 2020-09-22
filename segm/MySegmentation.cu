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

#include "MySegmentation.cuh"
#include "../cuda/segmentation.cuh"
#include "../cuda/cudafuncs.cuh"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <memory>
#include <algorithm>

MySegmentation::MySegmentation(int w, int h, const CameraModel& cameraIntrinsics) :
    minMappedComponentSize(160) {

    floatEdgeMap.create(h, w);
    floatBuffer.create(h, w);

    ucharBuffer.create(h, w);
    binaryEdgeMap.create(h, w);

    cv8UC1Buffer.create(h, w, CV_8UC1);
    cvLabelComps.create(h, w, CV_32S);
    cvLabelEdges.create(h, w, CV_32S);
    semanticIgnoreMap = cv::Mat::zeros(h, w, CV_8UC1);

    this->cameraIntrinsics = cameraIntrinsics;
    vertexMap.create(h*3,w);
    normalMap.create(h*3,w);
    depthMapMetric.create(h,w);
    depthMapMetricFiltered.create(h,w);
    rgb.create(h,w);
}

MySegmentation::~MySegmentation(){}

cv::Mat MySegmentation::performSegmentation(FrameData& frame)
{    
    cv::Mat result;
    const int& width = frame.depth.cols;
    const int& height = frame.depth.rows;
    const size_t total = frame.depth.total();
    result = cv::Mat::zeros(height, width, CV_8UC1);
    const int nMasks = int(frame.classIDs.size());
    const float maxRelSizeNew = 0.4;
    const float minRelSizeNew = 0.07;
    const size_t minNewMaskPixels = minRelSizeNew * total;
    const size_t maxNewMaskPixels = maxRelSizeNew * total;

    // Prepare data (vertex/depth/... maps)
    
    computeLookups(frame);
    computeGeometricSegmentationMap(vertexMap, normalMap, floatEdgeMap, weightDistance, weightConvexity);
    
    // Perform geometric segmentation

    
    DeviceArray2D<float>& edgeMap = floatEdgeMap;
    
    thresholdMap(edgeMap, binaryEdgeMap, threshold);
    morphGeometricSegmentationMap(binaryEdgeMap,ucharBuffer, morphEdgeRadius, morphEdgeIterations);
    invertMap(binaryEdgeMap,ucharBuffer);
    ucharBuffer.download(cv8UC1Buffer.data, ucharBuffer.cols());

#if true // FIXME: segmentation debugging

    cv::imshow("Segmentation", cv8UC1Buffer);

#endif

    // Build use ignore map
    if(nMasks)
    {
        for(size_t i=0; i<total; i++)
        {
            if(frame.classIDs[frame.mask.data[i]] == personClassID)
            {
                semanticIgnoreMap.data[i] = 255;
                cv8UC1Buffer.data[i] = 0;
            }
            else
            {
                semanticIgnoreMap.data[i] = 0;
            }
        }
        //cv::compare(frame.mask, cv::Scalar(...), semanticIgnoreMap, CV_CMP_EQ);
    }
    else
    {
        for(size_t i=0; i<total; i++)
        {
            if(semanticIgnoreMap.data[i])
                cv8UC1Buffer.data[i] = 0;
        }
    }

    // Run connected-components on segmented map
    cv::Mat statsComp, centroidsComp;
    int nComponents = cv::connectedComponentsWithStats(cv8UC1Buffer, cvLabelComps, statsComp, centroidsComp, 4);
    

    // Todo, this can be faster! (GPU?)
    if(removeEdges)
    {
        const bool remove_small_components = true;
        const int small_components_threshold = 50;
        const int removeEdgeIterations = 5;
        
        
        auto checkNeighbor = [&, this](int y, int x, int& n, float d) {
            n = this->cvLabelComps.at<int>(y,x);
            if (n != 0 && std::fabs(frame.depth.at<float>(y,x)-d) < 0.008 && statsComp.at<int>(n, 4) > small_components_threshold)
            {
                return true;
            }
            return false;
        };
        for (int i = 0; i < removeEdgeIterations; ++i)
        {
            cv::Mat r;
            cvLabelComps.copyTo(r);
            for (int y = 1; y < height-1; ++y) // TODO reduce index computations here
            {
                for (int x = 1; x < width-1; ++x)
                {
                    int& c = r.at<int>(y,x);
//                    statsComp.at<int>(c, 4);
                    float d = frame.depth.at<float>(y,x);

                    if(c==0 || (remove_small_components && statsComp.at<int>(c, 4) < small_components_threshold))
                    {
                        int c2;
                        if(checkNeighbor(y-1,x-1,c2,d)) { c = c2; continue; }
                        if(checkNeighbor(y-1,x,c2,d)) { c = c2; continue; }
                        if(checkNeighbor(y-1,x+1,c2,d)) { c = c2; continue; }
                        if(checkNeighbor(y,x-1,c2,d)) { c = c2; continue; }
                        if(checkNeighbor(y,x+1,c2,d)) { c = c2; continue; }
                        if(checkNeighbor(y+1,x-1,c2,d)) { c = c2; continue; }
                        if(checkNeighbor(y+1,x,c2,d)) { c = c2; continue; }
                        if(checkNeighbor(y+1,x+1,c2,d)) { c = c2; continue; }
                    }
                }
            }
            cvLabelComps = r;
        }
    }

    // Assign mask to each component
    // Replace edges and persons with 255
//    mapComponentToMask[0] = 255; // Edges

    // Group components that belong to the same mask
    
    std::vector<int> mapComponentToMask(nComponents, 0); // By default, components are mapped to background (maskid==0)
    std::vector<int> maskComponentPixels(nMasks, 0); // Number of pixels per mask
    std::vector<BoundingBox> maskComponentBoxes(nMasks);
    cv::Mat compMaskOverlap(nComponents,nMasks,CV_32SC1, cv::Scalar(0));

    if(nMasks)
    {

        // Compute component-mask overlap
        for (size_t i = 0; i < total; ++i)
        {
            const unsigned char& mask_val = frame.mask.data[i];
            const int& comp_val = cvLabelComps.at<int>(i);
            //assert(frame.classIDs.size() > mask_val);
            //if(mask_val != 255)
            compMaskOverlap.at<int>(comp_val,mask_val)++;
        }

        // Compute mapping
        const float overlap_threshold = 0.65;
        for (int c = 1; c < nComponents; ++c)
        {
            int& csize = statsComp.at<int>(c, 4);
            if(csize > minMappedComponentSize)
            {
                int t = overlap_threshold * csize;
                for (int m = 1; m < nMasks; ++m)
                {
                    if(compMaskOverlap.at<int>(c,m) > t)
                    {
                        mapComponentToMask[c] = m;
                        maskComponentPixels[m] += statsComp.at<int>(c, 4);
                        maskComponentBoxes[m].mergeLeftTopWidthHeight(statsComp.at<int>(c, 0),
                                                                      statsComp.at<int>(c, 1),
                                                                      statsComp.at<int>(c, 2),
                                                                      statsComp.at<int>(c, 3));
                    }
                }
            }
            else
            {
                // Map tiny component to ignored
                //mapComponentToMask[c] = 255;

                // Map tiny component to background
                mapComponentToMask[c] = 0;
            }
        }
    }

    for (size_t i = 0; i < total; ++i)
        result.data[i] = mapComponentToMask[cvLabelComps.at<int>(i)];
    
    // FIX HACK
    for(size_t i=0; i<total; i++)
        if(semanticIgnoreMap.data[i])
            result.data[i] = 255;
    
    if(removeEdgeIslands && nMasks)
    {
        // Remove "edge islands" within masks
        cv::threshold(result, cv8UC1Buffer, 254, 255, cv::THRESH_TOZERO); // THRESH_BINARY is equivalent here
        cv::Mat statsEdgeComp, centroidsEdgeComp;
        int nEdgeComp = cv::connectedComponentsWithStats(cv8UC1Buffer, cvLabelEdges, statsEdgeComp, centroidsEdgeComp, 4);
        //cv::imshow("edge labels", mapLabelToColorImage(cvLabelEdges));



        for (int ec = 1; ec < nEdgeComp; ++ec)
        {
            for (int m = 1; m < nMasks; ++m)
            {
                BoundingBox bb = BoundingBox::fromLeftTopWidthHeight(statsEdgeComp.at<int>(ec,0),
                                                                     statsEdgeComp.at<int>(ec,1),
                                                                     statsEdgeComp.at<int>(ec,2),
                                                                     statsEdgeComp.at<int>(ec,3));
                if(maskComponentBoxes[m].includes(bb))
                {
                    //std::cout << "mask " << m << " fully contains edge-component " << ec << std::endl;
                    int x1 = std::max(bb.left+1,1);
                    int x2 = std::min(bb.right, width-2);
                    int y1 = std::max(bb.top+1, 1);
                    int y2 = std::min(bb.bottom, height-2);
                    bool doBreak = false;
                    for (int y = y1; y <= y2; ++y) {
                        for (int x = x1; x <= x2; ++x) {
                            const int& le = cvLabelEdges.at<int>(y,x-1); // TODO this can be a bit faster
                            const int& te = cvLabelEdges.at<int>(y-1,x);
                            const int& ce = cvLabelEdges.at<int>(y,x);
                            const unsigned char& lm = result.at<unsigned char>(y,x-1);
                            const unsigned char& tm = result.at<unsigned char>(y-1,x);
                            const unsigned char& cm = result.at<unsigned char>(y,x);
                            if( (le!=ec && ce==ec && lm!=m) ||
                                    (le==ec && ce!=ec && cm!=m) ||
                                    (te!=ec && ce==ec && tm!=m) ||
                                    (te==ec && ce!=ec && cm!=m)) {
                                doBreak = true;
                                break;
                            }
                        }
                        if(doBreak) break;
                    }
                    if(doBreak) break;

                    // This can only happen once, replace component
                    for (int y = bb.top; y <= bb.bottom; ++y) {
                        for (int x = bb.left; x <= bb.right; ++x) {
                            if (cvLabelEdges.at<int>(y,x)==ec){
                                result.at<unsigned char>(y,x) = m;
                                //islands.at<unsigned char>(y,x) = 255;
                            }
                        }
                    }
                }
            }
        }
    }
    
    return result;
}

void MySegmentation::computeLookups(FrameData& frame)
{
    // Copy OpenGL depth texture for CUDA use
    // textureDepthMetric->cudaMap();
    // cudaArray* depthTexturePtr = textureDepthMetric->getCudaArray();
    // cudaMemcpy2DFromArray(depthMapMetric.ptr(0), depthMapMetric.step(), depthTexturePtr, 0, 0, depthMapMetric.colsBytes(), depthMapMetric.rows(),
    //                       cudaMemcpyDeviceToDevice);
    // textureDepthMetric->cudaUnmap();

    // textureRGB->cudaMap();
    // cudaArray* rgbTexturePtr = textureRGB->getCudaArray();
    // cudaMemcpy2DFromArray(rgb.ptr(0), rgb.step(), rgbTexturePtr, 0, 0, rgb.colsBytes(), rgb.rows(), cudaMemcpyDeviceToDevice);
    // textureRGB->cudaUnmap();
    rgb.upload(frame.rgb.data, frame.rgb.cols * sizeof(unsigned char) * 4, frame.rgb.rows, frame.rgb.cols);
    depthMapMetric.upload(frame.depth.data, frame.depth.cols * sizeof(float), frame.depth.rows, frame.depth.cols);

    // Custom filter for depth map
    bilateralFilter(rgb, depthMapMetric, depthMapMetricFiltered, bilatSigmaRadius, 0, bilatSigmaDepth, bilatSigmaColor, bilatSigmaLocation);
    //    cudaArray* debugMapPtr = debugMap->getCudaArray();
    //    cudaMemcpy2DToArray(debugMapPtr, 0, 0, depthMapMetricFiltered.ptr(0), depthMapMetricFiltered.step(), depthMapMetricFiltered.colsBytes(), depthMapMetricFiltered.rows(), cudaMemcpyDeviceToDevice);

    // Generate buffers for vertex and normal maps
    createVMap(cameraIntrinsics, depthMapMetricFiltered, vertexMap, 999.0f);
    createNMap(vertexMap, normalMap);

#if true // FIXME: normal map debugging

    // get normal map result
    cv::Mat sample(frame.depth.rows, frame.depth.cols, CV_32FC3);
    normalMap.download(sample.data, sample.cols * sizeof(float));

    // match normal map with OpenCV format
    cv::Mat relocte(frame.depth.rows, frame.depth.cols, CV_32FC3);
    int step = frame.depth.rows * frame.depth.cols;
    for (int i = 0; i < step; i++)
    {
        ((float*)relocte.data)[3 * i + 0] = ((float*)sample.data)[i + step * 2];
        ((float*)relocte.data)[3 * i + 1] = ((float*)sample.data)[i + step * 1];
        ((float*)relocte.data)[3 * i + 2] = ((float*)sample.data)[i + step * 0];
    }
    
    // display normal map image
    cv::imshow("Normal Map", relocte * 0.5f + 0.5f);

#endif

}
