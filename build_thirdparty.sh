#!/bin/sh
BUILD_TYPE=Release
cd thirdparty/DBoW3/
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..
make -j4
cd ../../opengv/
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..
make -j4
