cmake_minimum_required(VERSION 3.0)
project(giotto_bindings)

add_subdirectory(giotto/externals/pybind11)
set(BINDINGS_DIR "giotto/externals/bindings")

find_package(Boost 1.56 REQUIRED)
include_directories(${Boost_INCLUDE_DIR})


pybind11_add_module(giotto_ripser "${BINDINGS_DIR}/ripser_bindings.cpp")
set_target_properties(giotto_ripser PROPERTIES CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -D_hypot=hypot
                                                                  -march=native -fopenmp -D_GLIBCXX_PARALLEL")
set_target_properties(giotto_ripser PROPERTIES CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")
target_compile_definitions(giotto_ripser PRIVATE USE_COEFFICIENTS=1 ASSEMBLE_REDUCTION_MATRIX=1)

pybind11_add_module(giotto_wasserstein "${BINDINGS_DIR}/wasserstein_bindings.cpp")
target_link_libraries(giotto_wasserstein LINK_PUBLIC ${Boost_LIBRARIES})
set_target_properties(giotto_wasserstein PROPERTIES CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set_target_properties(giotto_wasserstein PROPERTIES CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE} -O3 ")
set_target_properties(giotto_wasserstein PROPERTIES CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -D_GLIBCXX_DEBUG")
set_target_properties(giotto_wasserstein PROPERTIES CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} -O2 -g -ggdb")

target_compile_definitions(giotto_wasserstein PRIVATE BOOST_RESULT_OF_USE_DECLTYPE=1 BOOST_ALL_NO_LIB=1 BOOST_SYSTEM_NO_DEPRECATED=1)

pybind11_add_module(giotto_bottleneck "${BINDINGS_DIR}/bottleneck_bindings.cpp")
target_link_libraries(giotto_bottleneck LINK_PUBLIC ${Boost_LIBRARIES})
set_target_properties(giotto_bottleneck PROPERTIES CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set_target_properties(giotto_bottleneck PROPERTIES CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE} -O3 ")
set_target_properties(giotto_bottleneck PROPERTIES CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -D_GLIBCXX_DEBUG")
set_target_properties(giotto_bottleneck PROPERTIES CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} -O2 -g -ggdb")

target_compile_definitions(giotto_bottleneck PRIVATE BOOST_RESULT_OF_USE_DECLTYPE=1 BOOST_ALL_NO_LIB=1 BOOST_SYSTEM_NO_DEPRECATED=1)

set(GUDHI_SRC_DIR "giotto/externals/gudhi-devel/src")

pybind11_add_module(giotto_cubical_complex "${BINDINGS_DIR}/cubical_complex_bindings.cpp")
target_link_libraries(giotto_cubical_complex LINK_PUBLIC ${Boost_LIBRARIES})
target_include_directories(giotto_cubical_complex PRIVATE "${GUDHI_SRC_DIR}/Bitmap_cubical_complex/include")
set_target_properties(giotto_cubical_complex PROPERTIES CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -shared -pthread -fPIC -fwrapv -Wall -fno-strict-aliasing
                                                                  -march=native -fopenmp -D_GLIBCXX_PARALLEL -frounding-math")
target_compile_definitions(giotto_cubical_complex PRIVATE BOOST_RESULT_OF_USE_DECLTYPE=1 BOOST_ALL_NO_LIB=1 BOOST_SYSTEM_NO_DEPRECATED=1)

pybind11_add_module(giotto_persistent_cohomology "${BINDINGS_DIR}/persistent_cohomology_bindings.cpp")
target_link_libraries(giotto_persistent_cohomology LINK_PUBLIC ${Boost_LIBRARIES})
target_include_directories(giotto_persistent_cohomology PRIVATE "${GUDHI_SRC_DIR}/Persistent_cohomology/include")
target_include_directories(giotto_persistent_cohomology PRIVATE "${GUDHI_SRC_DIR}/common/include")
target_include_directories(giotto_persistent_cohomology PRIVATE "${GUDHI_SRC_DIR}/Bitmap_cubical_complex/include")
set_target_properties(giotto_persistent_cohomology PROPERTIES CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -shared -pthread -fPIC -fwrapv -Wall -fno-strict-aliasing
                                                                  -march=native -fopenmp -D_GLIBCXX_PARALLEL -frounding-math")
target_compile_definitions(giotto_persistent_cohomology PRIVATE BOOST_RESULT_OF_USE_DECLTYPE=1 BOOST_ALL_NO_LIB=1 BOOST_SYSTEM_NO_DEPRECATED=1)

pybind11_add_module(giotto_simplex_tree "${BINDINGS_DIR}/simplex_tree_bindings.cpp")
target_link_libraries(giotto_simplex_tree LINK_PUBLIC ${Boost_LIBRARIES})
target_include_directories(giotto_simplex_tree PRIVATE "${GUDHI_SRC_DIR}/Simplex_tree/include")
target_include_directories(giotto_simplex_tree PRIVATE "${GUDHI_SRC_DIR}/common/include")
target_include_directories(giotto_simplex_tree PRIVATE "${GUDHI_SRC_DIR}/Cech_complex/include")
target_include_directories(giotto_simplex_tree PRIVATE "${GUDHI_SRC_DIR}/Persistent_cohomology/include")
target_include_directories(giotto_simplex_tree PRIVATE "${GUDHI_SRC_DIR}/Rips_complex/include")
target_include_directories(giotto_simplex_tree PRIVATE "${GUDHI_SRC_DIR}/Subsampling/include")
set_target_properties(giotto_simplex_tree PROPERTIES CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -shared -pthread -fPIC -fwrapv -Wall -fno-strict-aliasing
                                                                  -march=native -fopenmp -D_GLIBCXX_PARALLEL -frounding-math")
target_compile_definitions(giotto_simplex_tree PRIVATE BOOST_RESULT_OF_USE_DECLTYPE=1 BOOST_ALL_NO_LIB=1 BOOST_SYSTEM_NO_DEPRECATED=1)
