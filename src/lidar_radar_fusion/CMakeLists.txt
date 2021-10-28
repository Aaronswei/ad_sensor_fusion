cmake_minimum_required(VERSION 3.10)
project(LidarRadarFusion)
# set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

# if(NOT CMAKE_BUILD_TYPE)
#     set(CMAKE_BUILD_TYPE release CACHE "Choose the type of build" FORCE)
# endif()

# OpenCV
set(OpenCV_DIR "/home/kavin/3rdparty/share/OpenCV")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "libraries: ${OpenCV_LIBS}")

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
message(STATUS "EIGEN3_VERSION: ${EIGEN3_VERSION}")

# PCL
find_package(PCL COMPONENTS segmentation surface common io registration sample_consensus filters visualization REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
message(STATUS "PCL_INCLUDE_DIRS: ${PCL_INCLUDE_DIRS}")
message(STATUS "PCL_LIBRARIES: ${PCL_LIBRARIES}")
message(STATUS "PCL_VERSION: ${PCL_VERSION}")

# GFlags
find_package(gflags REQUIRED)
message(STATUS "GFLAGS_LIBRARIES: ${GFLAGS_LIBRARIES}")

# GLog
find_package(glog REQUIRED)
include_directories(SYSTEM ${GLOG_INCLUDE_DIRS})
link_directories(/usr/local/lib)
message(STATUS "GLOG_LIBRARIES: ${GLOG_LIBRARIES}")

include_directories(./include/)
add_executable(${PROJECT_NAME} src/main.cpp src/lidar_radar_fusion.cpp
                                            src/lidar_radar_fusion_task.cpp
                                            src/lidar_radar_fusion_app.cpp
                                            src/utils.cpp
                                            src/ekf.cpp
                                            src/ekfInterface.cpp
                                            src/kalmanFilter.cpp
                                            src/tools.cpp)
target_link_libraries( ${PROJECT_NAME}  boost_system ${PCL_LIBRARIES} ${OpenCV_LIBS} glog::glog ${GFLAGS_LIBRARIES})




