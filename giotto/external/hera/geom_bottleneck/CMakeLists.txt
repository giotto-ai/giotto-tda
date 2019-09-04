project (geom_bottleneck)
cmake_minimum_required (VERSION 3.5.1)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# C++ 11 required
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release

if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif (NOT CMAKE_BUILD_TYPE)

if(NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -D_GLIBCXX_DEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 ")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} -O2 -g -ggdb")
endif(NOT WIN32)

include_directories         (${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB WS_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)



find_package (Threads)
set (libraries ${libraries}     ${CMAKE_THREAD_LIBS_INIT})

add_executable(bottleneck_dist ${CMAKE_CURRENT_SOURCE_DIR}/example/bottleneck_dist.cpp ${WS_HEADERS})
target_link_libraries(bottleneck_dist PUBLIC ${libraries})

add_executable(bottleneck_test  ${CMAKE_CURRENT_SOURCE_DIR}/tests/tests_main.cpp ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_hera_bottleneck.cpp)
target_link_libraries(bottleneck_test PUBLIC ${libraries})
