cmake_minimum_required(VERSION 2.8.11)
project(vinsmonoinit)

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
#set(CMAKE_BUILD_TYPE "Debug")
#set(CMAKE_CXX_FLAGS "-std=c++11 -O0 -Wall -g -march=native")
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../cmake)

find_package(Glog REQUIRED)
find_package(GFlags REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen REQUIRED)
find_package(Ceres REQUIRED)

set(HEADERS
        estimator/estimator.h
        estimator/feature_manager.h
        factor/integration_base.h
        initial/initial_sfm.h
        initial/initial_alignment.h
        initial/solve_5pts.h
        utility/utility.h
        )

set(SOURCES
        estimator/estimator.cpp
        estimator/feature_manager.cpp
        initial/initial_sfm.cpp
        initial/initial_aligment.cpp
        initial/solve_5pts.cpp
        utility/utility.cpp)

add_library(MONOinit SHARED
        ${SOURCES}
        )
target_include_directories(MONOinit PUBLIC
        ${GFLAGS_INCLUDE_DIR}
        ${EIGEN_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIR}
        ${YAMLCPP_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CERES_INCLUDE_DIRS}
        )
target_link_libraries(MONOinit
        ${GFLAGS_LIBRARIES}
        ${OpenCV_LIBS}
        ${YAMLCPP_LIBRARIES}
        glog
        ${CERES_LIBRARIES}
        )
