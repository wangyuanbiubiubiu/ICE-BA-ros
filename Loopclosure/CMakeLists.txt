cmake_minimum_required(VERSION 2.8.11)
project(loop_closure)

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
#set(CMAKE_BUILD_TYPE "Debug")
#set(CMAKE_CXX_FLAGS "-std=c++11 -O0 -Wall -g -march=native")
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../cmake)

find_package(Glog REQUIRED)
find_package(GFlags REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen REQUIRED)
#DBoW3
set(DBoW3_LIBS ${PROJECT_SOURCE_DIR}/../thirdparty/DBoW3/build/src/libDBoW3${CMAKE_SHARED_LIBRARY_SUFFIX})

#opengv
set(opengv_LIBS ${PROJECT_SOURCE_DIR}/../thirdparty/opengv/build/devel/lib/libopengv${CMAKE_SHARED_LIBRARY_SUFFIX})

set(HEADERS
        LoopClosing.h
        KeyFrame.h
        FeatureMatcher.h
        PnpPoseEstimator.h
        )

set(SOURCES
        LoopClosing.cpp
        KeyFrame.cpp
        FeatureMatcher.cpp
        PnpPoseEstimator.cpp)

add_library(LC SHARED
        ${SOURCES}
        )
target_include_directories(LC PUBLIC
        ${GFLAGS_INCLUDE_DIR}
        ${EIGEN_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIR}
        ${YAMLCPP_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/../thirdparty/DBoW3/
        ${PROJECT_SOURCE_DIR}/../thirdparty/opengv/include/
        )
target_link_libraries(LC
        ${GFLAGS_LIBRARIES}
        ${OpenCV_LIBS}
        ${YAMLCPP_LIBRARIES}
        IBA
        glog
        ${DBoW3_LIBS}
        ${opengv_LIBS}
        )
