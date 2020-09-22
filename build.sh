#!/bin/bash

# get dependencies
cd dep

# pip dependencies
pip3 install pip --upgrade
pip3 install tensorflow
pip3 install scikit-image
pip3 install keras
pip3 install IPython
pip3 install h5py
pip3 install cython
pip3 install imgaug
pip3 install opencv-python
pip3 install pytoml

# COCO
git clone https://github.com/waleedka/coco.git
cd coco/PythonAPI
make && make install
cd ../..

# Mask-RCNN
cd Mask_RCNN
mkdir -p data && cd data
wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
cd ../..

# end dependencies
cd ..

# build process
mkdir -p build && cd build
cmake ..
make -j9

# run test program
./segtest
