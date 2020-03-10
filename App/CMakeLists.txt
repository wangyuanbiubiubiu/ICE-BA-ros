cmake_minimum_required(VERSION 2.8.11)
project(EuRoC_test)


#set(CMAKE_BUILD_TYPE "Debug")
#set(CMAKE_CXX_FLAGS "-std=c++11 -O0 -Wall -g -march=native")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../bin)

set(USE_ROS true)   #true or false
# =========================
# ros
# =========================
if(USE_ROS)
 find_package(catkin REQUIRED COMPONENTS
         roscpp
         std_msgs
         geometry_msgs
         nav_msgs
         tf
         cv_bridge
         )
 include_directories(
         ${catkin_INCLUDE_DIRS}
 )
endif(USE_ROS)

# =========================
# run front-end + back-end with ROS
# =========================
if(USE_ROS)
 add_executable(ice_ba_ros
         ../App/ICE-BA-ROS.cpp
         ../ros_visualization/CameraPoseVisualization.cpp ../ros_visualization/CameraPoseVisualization.h ../ros_visualization/visualization.cpp ../ros_visualization/visualization.h)
 target_link_libraries(ice_ba_ros
         IBA
         OF
         LC
         ${catkin_LIBRARIES}
         )
 if (CVD_FOUND)
  target_link_libraries(ice_ba_ros
          IBAVis
          )
 endif(CVD_FOUND)
endif(USE_ROS)


# =========================
# run front-end + back-end with EuRoC dataset
# =========================
add_executable(ice_ba
    ../App/EuRoC_test.cpp
)
target_link_libraries(ice_ba
    IBA
    OF
    LC
)
if (CVD_FOUND)
 target_link_libraries(ice_ba
     IBAVis
)
endif(CVD_FOUND)


# =========================
# run back-end only with .dat file
# =========================
add_executable(back_end
    ../App/run_backend.cpp
)
target_include_directories(back_end PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GLOBAL_IBA_INCLUDE_DIRS}
    ${EIGEN_INCLUDE_DIR}
)
target_link_libraries(back_end
    IBA
    OF
)
if (CVD_FOUND)
 target_include_directories(back_end PUBLIC
     ${GLOBAL_VISUALIZATION_INCLUDE_DIR}
 )
 target_link_libraries(back_end
     IBAVis
 )
endif (CVD_FOUND)
