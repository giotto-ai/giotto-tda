cmake_minimum_required(VERSION 3.9)
project(giotto_bindings LANGUAGES CXX)

find_package(pybind11 REQUIRED)
set(BINDINGS_DIR "giotto/externals/bindings")

find_package(Boost 1.56 REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

find_package(OpenMP)

#######################################################################
#                               Ripser                                #
#######################################################################

pybind11_add_module(giotto_ripser "${BINDINGS_DIR}/ripser_bindings.cpp")
if(OpenMP_FOUND)
  target_link_libraries(giotto_ripser PRIVATE OpenMP::OpenMP_CXX)
endif()
target_compile_options(giotto_ripser PUBLIC -O3 -D_hypot=hypot)
target_compile_options(giotto_ripser PUBLIC $<$<CONFIG:DEBUG>:-O2 -ggdb -D_GLIBCXX_DEBUG>)
# Cannot have two inlined namespace !
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(giotto_ripser PUBLIC -march=native -D_GLIBCXX_PARALLEL)
  target_link_libraries(giotto_ripser PRIVATE)
endif()
target_compile_definitions(giotto_ripser PRIVATE USE_COEFFICIENTS=1 ASSEMBLE_REDUCTION_MATRIX=1)