project                     (wasserstein)
cmake_minimum_required      (VERSION 3.5.1)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include (GenerateExportHeader)

include(TestBigEndian)
test_big_endian(BIG_ENDIAN)
if(BIG_ENDIAN)
  add_definitions(-DBIGENDIAN)
endif()

# Default to Release

if                          (NOT CMAKE_BUILD_TYPE)
    set                     (CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif                       (NOT CMAKE_BUILD_TYPE)

# Boost
find_package                (Boost)
include_directories         (${CMAKE_CURRENT_SOURCE_DIR}/include
                             SYSTEM ${Boost_INCLUDE_DIR})

if(NOT WIN32)
    add_definitions(-std=c++14)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -D_GLIBCXX_DEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 ")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} -O2 -g -ggdb")
endif(NOT WIN32)

file(GLOB WS_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

#add_library(wasserstein ${WS_SOURCES})

#if (WIN32)
    #GENERATE_EXPORT_HEADER(wasserstein
               #BASE_NAME wasserstein
               #EXPORT_MACRO_NAME wasserstein_EXPORT
               #EXPORT_FILE_NAME wasserstein_export.h
               #STATIC_DEFINE wasserstein_BUILT_AS_STATIC)
#endif(WIN32)

find_package (Threads)
set (libraries ${libraries}     ${CMAKE_THREAD_LIBS_INIT})

add_executable(wasserstein_dist ${CMAKE_CURRENT_SOURCE_DIR}/example/wasserstein_dist.cpp ${WS_HEADERS} include/hera_infinity.h)
target_link_libraries(wasserstein_dist PUBLIC ${libraries})

add_executable(wasserstein_dist_dipha ${CMAKE_CURRENT_SOURCE_DIR}/example/wasserstein_dist_dipha.cpp ${WS_HEADERS} include/hera_infinity.h)
target_link_libraries(wasserstein_dist_dipha PUBLIC ${libraries})

# pure geometric version, arbitrary dimension
add_executable(wasserstein_dist_point_cloud ${CMAKE_CURRENT_SOURCE_DIR}/example/wasserstein_dist_point_cloud.cpp ${WS_HEADERS} include/hera_infinity.h)
target_link_libraries(wasserstein_dist_point_cloud PUBLIC ${libraries})

# Tests
add_executable(wasserstein_test  ${CMAKE_CURRENT_SOURCE_DIR}/tests/tests_main.cpp ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_hera_wasserstein.cpp ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_hera_wasserstein_pure_geom.cpp  include/hera_infinity.h tests/tests_reader.h)
#add_executable(wasserstein_test EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_hera_wasserstein.cpp)
target_link_libraries(wasserstein_test PUBLIC ${libraries})
