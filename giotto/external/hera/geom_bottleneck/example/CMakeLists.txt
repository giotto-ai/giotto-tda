cmake_minimum_required(VERSION 2.8.9)

# Default to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# Add path to ANN header files
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../bottleneck/include")

set(CXX_FLAGS ${CMAKE_CXX_FLAGS_RELEASE})

if(NOT WIN32)
    set(CMAKE_CXX_FLAGS "-O3 -DNDEBUG -DBOOST_DISABLE_ASSERTS")
    add_definitions(-std=c++11)
endif(NOT WIN32)


if(UNIX)
    # specify full path to ANN
    set_property(TARGET bottleneck PROPERTY IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/../bottleneck/build/libbottleneck.a")
endif(UNIX)

add_executable (bottleneck_dist bottleneck_dist.cpp)

target_link_libraries(bottleneck_dist PUBLIC bottleneck)

