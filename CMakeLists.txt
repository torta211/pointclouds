CMAKE_MINIMUM_REQUIRED(VERSION 3.11)
project( pointclouds C CXX )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package( Eigen3 REQUIRED )

add_executable(pointclouds src/main.cpp)
target_link_libraries(pointclouds 
  Eigen3::Eigen
)