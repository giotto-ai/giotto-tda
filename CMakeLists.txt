cmake_minimum_required(VERSION 3.0)
project(giotto_bindings)

add_subdirectory(giotto/externals/pybind11)
set(BINDINGS_DIR "giotto/externals/bindings")

find_package(Boost 1.56 REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

pybind11_add_module(giotto_ripser "${BINDINGS_DIR}/ripser_bindings.cpp")
set_target_properties(giotto_ripser PROPERTIES CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -D_hypot=hypot
                                                                  -march=native -fopenmp -D_GLIBCXX_PARALLEL")
set_target_properties(giotto_ripser PROPERTIES CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")
target_compile_definitions(giotto_ripser PRIVATE USE_COEFFICIENTS=1 ASSEMBLE_REDUCTION_MATRIX=1)

pybind11_add_module(giotto_wasserstein "${BINDINGS_DIR}/wasserstein_bindings.cpp")
target_link_libraries(giotto_wasserstein LINK_PUBLIC ${Boost_LIBRARIES})
target_compile_options(giotto_wasserstein PUBLIC -Wall -O3)
target_compile_options(giotto_wasserstein PUBLIC $<$<CONFIG:DEBUG>:-O2 -ggdb -D_GLIBCXX_DEBUG>)
target_compile_definitions(giotto_wasserstein PRIVATE BOOST_RESULT_OF_USE_DECLTYPE=1 BOOST_ALL_NO_LIB=1 BOOST_SYSTEM_NO_DEPRECATED=1)

pybind11_add_module(giotto_bottleneck "${BINDINGS_DIR}/bottleneck_bindings.cpp")
target_link_libraries(giotto_bottleneck LINK_PUBLIC ${Boost_LIBRARIES})
target_compile_options(giotto_bottleneck PUBLIC -Wall -O3)
target_compile_options(giotto_bottleneck PUBLIC $<$<CONFIG:DEBUG>:-O2 -ggdb -D_GLIBCXX_DEBUG>)
target_compile_definitions(giotto_bottleneck PRIVATE BOOST_RESULT_OF_USE_DECLTYPE=1 BOOST_ALL_NO_LIB=1 BOOST_SYSTEM_NO_DEPRECATED=1)

set(GUDHI_SRC_DIR "giotto/externals/gudhi-devel/src")

pybind11_add_module(giotto_cubical_complex "${BINDINGS_DIR}/cubical_complex_bindings.cpp")
target_link_libraries(giotto_cubical_complex LINK_PUBLIC ${Boost_LIBRARIES})
target_include_directories(giotto_cubical_complex PRIVATE "${GUDHI_SRC_DIR}/Bitmap_cubical_complex/include")
target_compile_options(giotto_cubical_complex PUBLIC -Ofast -shared -pthread -fPIC -fwrapv -Wall -fno-strict-aliasing -frounding-math)
target_compile_options(giotto_cubical_complex PUBLIC $<$<CONFIG:DEBUG>:-O2 -ggdb -D_GLIBCXX_DEBUG>)

# Cannot have two inlined namespace !
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(giotto_cubical_complex PUBLIC -march=native -fopenmp -D_GLIBCXX_PARALLEL)
endif()

target_compile_definitions(giotto_cubical_complex PRIVATE BOOST_RESULT_OF_USE_DECLTYPE=1 BOOST_ALL_NO_LIB=1 BOOST_SYSTEM_NO_DEPRECATED=1)

pybind11_add_module(giotto_persistent_cohomology "${BINDINGS_DIR}/persistent_cohomology_bindings.cpp")
target_link_libraries(giotto_persistent_cohomology LINK_PUBLIC ${Boost_LIBRARIES})
target_include_directories(giotto_persistent_cohomology PRIVATE "${GUDHI_SRC_DIR}/Persistent_cohomology/include")
target_include_directories(giotto_persistent_cohomology PRIVATE "${GUDHI_SRC_DIR}/common/include")
target_include_directories(giotto_persistent_cohomology PRIVATE "${GUDHI_SRC_DIR}/Bitmap_cubical_complex/include")
target_compile_options(giotto_persistent_cohomology PUBLIC -Ofast -shared -pthread -fPIC -fwrapv -Wall -fno-strict-aliasing
                                                                  -frounding-math)
target_compile_options(giotto_persistent_cohomology PUBLIC $<$<CONFIG:DEBUG>:-O2 -ggdb -D_GLIBCXX_DEBUG>)

# Cannot have two inlined namespace !
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(giotto_persistent_cohomology PUBLIC -march=native -fopenmp -D_GLIBCXX_PARALLEL)
endif()
target_compile_definitions(giotto_persistent_cohomology PRIVATE BOOST_RESULT_OF_USE_DECLTYPE=1 BOOST_ALL_NO_LIB=1 BOOST_SYSTEM_NO_DEPRECATED=1)

pybind11_add_module(giotto_simplex_tree "${BINDINGS_DIR}/simplex_tree_bindings.cpp")
target_link_libraries(giotto_simplex_tree LINK_PUBLIC ${Boost_LIBRARIES})
target_include_directories(giotto_simplex_tree PRIVATE "${GUDHI_SRC_DIR}/Simplex_tree/include")
target_include_directories(giotto_simplex_tree PRIVATE "${GUDHI_SRC_DIR}/common/include")
target_include_directories(giotto_simplex_tree PRIVATE "${GUDHI_SRC_DIR}/Cech_complex/include")
target_include_directories(giotto_simplex_tree PRIVATE "${GUDHI_SRC_DIR}/Persistent_cohomology/include")
target_include_directories(giotto_simplex_tree PRIVATE "${GUDHI_SRC_DIR}/Rips_complex/include")
target_include_directories(giotto_simplex_tree PRIVATE "${GUDHI_SRC_DIR}/Subsampling/include")
target_compile_options(giotto_simplex_tree PUBLIC -Ofast -shared -pthread -fPIC
                        -fwrapv -Wall -fno-strict-aliasing -frounding-math)
target_compile_options(giotto_simplex_tree PUBLIC $<$<CONFIG:DEBUG>:-O2 -ggdb -D_GLIBCXX_DEBUG>)
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(giotto_simplex_tree PUBLIC -march=native -fopenmp -D_GLIBCXX_PARALLEL)
endif()
target_compile_definitions(giotto_simplex_tree PRIVATE BOOST_RESULT_OF_USE_DECLTYPE=1 BOOST_ALL_NO_LIB=1 BOOST_SYSTEM_NO_DEPRECATED=1)
