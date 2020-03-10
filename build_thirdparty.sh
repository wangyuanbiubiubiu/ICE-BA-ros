#!/bin/sh
cd thirdparty/DBoW3/
mkdir build
cd build
cmake ..
make -j4
cd ../../opengv/
mkdir build
cd build
cmake ..
make -j4
