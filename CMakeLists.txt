cmake_minimum_required(VERSION 3.9)
project(gtda_bindings LANGUAGES CXX)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/gtda/externals/pybind11)
set(BINDINGS_DIR "gtda/externals/bindings")

include(cmake/HelperBoost.cmake)
include_directories(${Boost_INCLUDE_DIR})

find_package(OpenMP)

set(RIPSER_SRC_DIR "gtda/externals/ripser")
set(GUDHI_SRC_DIR "gtda/externals/gudhi-devel/src")

#######################################################################
#                               Ripser                                #
#######################################################################

pybind11_add_module(gtda_ripser "${BINDINGS_DIR}/ripser_bindings.cpp")
if(OpenMP_FOUND)
  target_link_libraries(gtda_ripser PRIVATE OpenMP::OpenMP_CXX)
endif()
target_compile_definitions(gtda_ripser PRIVATE ASSEMBLE_REDUCTION_MATRIX=1)

target_include_directories(gtda_ripser PRIVATE "${RIPSER_SRC_DIR}/ripser")

if(MSVC)
    target_compile_options(gtda_ripser PUBLIC $<$<CONFIG:RELEASE>: /Wall /O2>)
    target_compile_options(gtda_ripser PUBLIC $<$<CONFIG:DEBUG>:/O1 /DEBUG:FULL /Zi /Zo>)
else()
    target_compile_options(gtda_ripser PUBLIC $<$<CONFIG:RELEASE>: -O3>)
    target_compile_options(gtda_ripser PUBLIC $<$<CONFIG:DEBUG>: -O2 -ggdb -D_GLIBCXX_DEBUG>)
endif()

#######################################################################
#                     Ripser - Coefficient enable                     #
#######################################################################

pybind11_add_module(gtda_ripser_coeff "${BINDINGS_DIR}/ripser_bindings.cpp")
if(OpenMP_FOUND)
  target_link_libraries(gtda_ripser_coeff PRIVATE OpenMP::OpenMP_CXX)
endif()
target_compile_definitions(gtda_ripser_coeff PRIVATE USE_COEFFICIENTS=1 ASSEMBLE_REDUCTION_MATRIX=1)

target_include_directories(gtda_ripser_coeff PRIVATE "${RIPSER_SRC_DIR}/ripser")

if(MSVC)
    target_compile_options(gtda_ripser_coeff PUBLIC $<$<CONFIG:RELEASE>: /Wall /O2>)
    target_compile_options(gtda_ripser_coeff PUBLIC $<$<CONFIG:DEBUG>:/O1 /DEBUG:FULL /Zi /Zo>)
else()
    target_compile_options(gtda_ripser_coeff PUBLIC $<$<CONFIG:RELEASE>: -O3>)
    target_compile_options(gtda_ripser_coeff PUBLIC $<$<CONFIG:DEBUG>: -O2 -ggdb -D_GLIBCXX_DEBUG>)
endif()

#######################################################################
#                             Wasserstein                             #
#######################################################################

pybind11_add_module(gtda_wasserstein "${BINDINGS_DIR}/wasserstein_bindings.cpp")
target_link_libraries(gtda_wasserstein LINK_PUBLIC ${Boost_LIBRARIES})
target_compile_definitions(gtda_wasserstein PRIVATE BOOST_RESULT_OF_USE_DECLTYPE=1 BOOST_ALL_NO_LIB=1 BOOST_SYSTEM_NO_DEPRECATED=1)

if(MSVC)
    target_compile_options(gtda_wasserstein PUBLIC $<$<CONFIG:RELEASE>: /Wall /O2>)
    target_compile_options(gtda_wasserstein PUBLIC $<$<CONFIG:DEBUG>:/O1 /DEBUG:FULL /Zi /Zo>)
else()
    target_compile_options(gtda_wasserstein PUBLIC $<$<CONFIG:RELEASE>: -Wall -O3>)
    target_compile_options(gtda_wasserstein PUBLIC $<$<CONFIG:DEBUG>:-O2 -ggdb -D_GLIBCXX_DEBUG>)
endif()

#######################################################################
#                             Bottleneck                              #
#######################################################################

pybind11_add_module(gtda_bottleneck "${BINDINGS_DIR}/bottleneck_bindings.cpp")
target_link_libraries(gtda_bottleneck LINK_PUBLIC ${Boost_LIBRARIES})
target_compile_definitions(gtda_bottleneck PRIVATE BOOST_RESULT_OF_USE_DECLTYPE=1 BOOST_ALL_NO_LIB=1 BOOST_SYSTEM_NO_DEPRECATED=1)

if(MSVC)
    target_compile_options(gtda_bottleneck PUBLIC $<$<CONFIG:RELEASE>: /Wall /O2>)
    target_compile_options(gtda_bottleneck PUBLIC $<$<CONFIG:DEBUG>:/O1 /DEBUG:FULL /Zi /Zo>)
else()
    target_compile_options(gtda_bottleneck PUBLIC $<$<CONFIG:RELEASE>: -Wall -O3>)
    target_compile_options(gtda_bottleneck PUBLIC $<$<CONFIG:DEBUG>:-O2 -ggdb -D_GLIBCXX_DEBUG>)
endif()

#######################################################################
#                           Cubical Complex                           #
#######################################################################

pybind11_add_module(gtda_cubical_complex "${BINDINGS_DIR}/cubical_complex_bindings.cpp")

if(OpenMP_FOUND)
  target_link_libraries(gtda_cubical_complex PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(gtda_cubical_complex LINK_PUBLIC ${Boost_LIBRARIES})
target_compile_definitions(gtda_cubical_complex PRIVATE BOOST_RESULT_OF_USE_DECLTYPE=1 BOOST_ALL_NO_LIB=1 BOOST_SYSTEM_NO_DEPRECATED=1)

target_include_directories(gtda_cubical_complex PRIVATE "${GUDHI_SRC_DIR}/Bitmap_cubical_complex/include")
target_include_directories(gtda_cubical_complex PRIVATE "${GUDHI_SRC_DIR}/python/include")

if(MSVC)
    target_compile_options(gtda_cubical_complex PUBLIC $<$<CONFIG:RELEASE>: /O2 /Wall /fp:strict>)
    target_compile_options(gtda_cubical_complex PUBLIC $<$<CONFIG:DEBUG>:/O1 /DEBUG:FULL /Zi /Zo>)
else()
    target_compile_options(gtda_cubical_complex PUBLIC $<$<CONFIG:RELEASE>: -Ofast -shared -pthread -fPIC -fwrapv -Wall -fno-strict-aliasing -frounding-math>)
    target_compile_options(gtda_cubical_complex PUBLIC $<$<CONFIG:DEBUG>:-O2 -ggdb -D_GLIBCXX_DEBUG>)
endif()

#######################################################################
#                        Persistent Cohomology                        #
#######################################################################

pybind11_add_module(gtda_persistent_cohomology "${BINDINGS_DIR}/persistent_cohomology_bindings.cpp")

if(OpenMP_FOUND)
  target_link_libraries(gtda_persistent_cohomology PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(gtda_persistent_cohomology LINK_PUBLIC ${Boost_LIBRARIES})
target_compile_definitions(gtda_persistent_cohomology PRIVATE BOOST_RESULT_OF_USE_DECLTYPE=1 BOOST_ALL_NO_LIB=1 BOOST_SYSTEM_NO_DEPRECATED=1)

target_include_directories(gtda_persistent_cohomology PRIVATE "${GUDHI_SRC_DIR}/Persistent_cohomology/include")
target_include_directories(gtda_persistent_cohomology PRIVATE "${GUDHI_SRC_DIR}/common/include")
target_include_directories(gtda_persistent_cohomology PRIVATE "${GUDHI_SRC_DIR}/Bitmap_cubical_complex/include")
target_include_directories(gtda_persistent_cohomology PRIVATE "${GUDHI_SRC_DIR}/python/include")

if(MSVC)
    target_compile_options(gtda_persistent_cohomology PUBLIC $<$<CONFIG:RELEASE>: /O2 /Wall /fp:strict>)
    target_compile_options(gtda_persistent_cohomology PUBLIC $<$<CONFIG:DEBUG>:/O1 /DEBUG:FULL /Zi /Zo>)
else()
    target_compile_options(gtda_persistent_cohomology PUBLIC $<$<CONFIG:RELEASE>: -Ofast -shared -pthread -fPIC -fwrapv -Wall -fno-strict-aliasing -frounding-math>)
    target_compile_options(gtda_persistent_cohomology PUBLIC $<$<CONFIG:DEBUG>:-O2 -ggdb -D_GLIBCXX_DEBUG>)
endif()

#######################################################################
#                            Simplex Tree                             #
#######################################################################

pybind11_add_module(gtda_simplex_tree "${BINDINGS_DIR}/simplex_tree_bindings.cpp")

if(OpenMP_FOUND)
    target_link_libraries(gtda_simplex_tree PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(gtda_simplex_tree LINK_PUBLIC ${Boost_LIBRARIES})
target_compile_definitions(gtda_simplex_tree PRIVATE BOOST_RESULT_OF_USE_DECLTYPE=1 BOOST_ALL_NO_LIB=1 BOOST_SYSTEM_NO_DEPRECATED=1)

target_include_directories(gtda_simplex_tree PRIVATE "${GUDHI_SRC_DIR}/Simplex_tree/include")
target_include_directories(gtda_simplex_tree PRIVATE "${GUDHI_SRC_DIR}/common/include")
target_include_directories(gtda_simplex_tree PRIVATE "${GUDHI_SRC_DIR}/Cech_complex/include")
target_include_directories(gtda_simplex_tree PRIVATE "${GUDHI_SRC_DIR}/Persistent_cohomology/include")
target_include_directories(gtda_simplex_tree PRIVATE "${GUDHI_SRC_DIR}/Subsampling/include")
target_include_directories(gtda_simplex_tree PRIVATE "${GUDHI_SRC_DIR}/python/include")

if(MSVC)
    target_compile_options(gtda_simplex_tree PUBLIC $<$<CONFIG:RELEASE>: /O2 /Wall /fp:strict>)
    target_compile_options(gtda_simplex_tree PUBLIC $<$<CONFIG:DEBUG>:/O1 /DEBUG:FULL /Zi /Zo>)
else()
    target_compile_options(gtda_simplex_tree PUBLIC $<$<CONFIG:RELEASE>: -Ofast -shared -pthread -fPIC -fwrapv -Wall -fno-strict-aliasing -frounding-math>)
    target_compile_options(gtda_simplex_tree PUBLIC $<$<CONFIG:DEBUG>:-O2 -ggdb -D_GLIBCXX_DEBUG>)
endif()

#######################################################################
#                      Periodic Cubical Complex                       #
#######################################################################

pybind11_add_module(gtda_periodic_cubical_complex "${BINDINGS_DIR}/periodic_cubical_complex_bindings.cpp")

if(OpenMP_FOUND)
    target_link_libraries(gtda_periodic_cubical_complex PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(gtda_periodic_cubical_complex LINK_PUBLIC ${Boost_LIBRARIES})
target_compile_definitions(gtda_periodic_cubical_complex PRIVATE BOOST_RESULT_OF_USE_DECLTYPE=1 BOOST_ALL_NO_LIB=1 BOOST_SYSTEM_NO_DEPRECATED=1)

target_include_directories(gtda_periodic_cubical_complex PRIVATE "${GUDHI_SRC_DIR}/Bitmap_cubical_complex/include")
target_include_directories(gtda_periodic_cubical_complex PRIVATE "${GUDHI_SRC_DIR}/Persistent_cohomology/include")
target_include_directories(gtda_periodic_cubical_complex PRIVATE "${GUDHI_SRC_DIR}/python/include")
target_include_directories(gtda_periodic_cubical_complex PRIVATE "${GUDHI_SRC_DIR}/common/include")

if(MSVC)
    target_compile_options(gtda_periodic_cubical_complex PUBLIC $<$<CONFIG:RELEASE>: /O2 /Wall /fp:strict>)
    target_compile_options(gtda_periodic_cubical_complex PUBLIC $<$<CONFIG:DEBUG>:/O1 /DEBUG:FULL /Zi /Zo>)
else()
    target_compile_options(gtda_periodic_cubical_complex PUBLIC $<$<CONFIG:RELEASE>: -Ofast -shared -pthread -fPIC -fwrapv -Wall -fno-strict-aliasing -frounding-math>)
    target_compile_options(gtda_periodic_cubical_complex PUBLIC $<$<CONFIG:DEBUG>:-O2 -ggdb -D_GLIBCXX_DEBUG>)
endif()

#######################################################################
#                           Witness Complex                           #
#######################################################################

pybind11_add_module(gtda_witness_complex "${BINDINGS_DIR}/witness_complex_bindings.cpp")

if(OpenMP_FOUND)
    target_link_libraries(gtda_witness_complex PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(gtda_witness_complex LINK_PUBLIC ${Boost_LIBRARIES})
target_compile_definitions(gtda_witness_complex PRIVATE BOOST_RESULT_OF_USE_DECLTYPE=1 BOOST_ALL_NO_LIB=1 BOOST_SYSTEM_NO_DEPRECATED=1)

target_include_directories(gtda_witness_complex PRIVATE "${GUDHI_SRC_DIR}/Witness_complex/include")
target_include_directories(gtda_witness_complex PRIVATE "${GUDHI_SRC_DIR}/Simplex_tree/include")
target_include_directories(gtda_witness_complex PRIVATE "${GUDHI_SRC_DIR}/Cech_complex/include")
target_include_directories(gtda_witness_complex PRIVATE "${GUDHI_SRC_DIR}/Persistent_cohomology/include")
target_include_directories(gtda_witness_complex PRIVATE "${GUDHI_SRC_DIR}/python/include")
target_include_directories(gtda_witness_complex PRIVATE "${GUDHI_SRC_DIR}/common/include")

if(MSVC)
    target_compile_options(gtda_witness_complex PUBLIC $<$<CONFIG:RELEASE>: /O2 /Wall /fp:strict>)
    target_compile_options(gtda_witness_complex PUBLIC $<$<CONFIG:DEBUG>:/O1 /DEBUG:FULL /Zi /Zo>)
else()
    target_compile_options(gtda_witness_complex PUBLIC $<$<CONFIG:RELEASE>: -Ofast -shared -pthread -fPIC -fwrapv -Wall -fno-strict-aliasing -frounding-math>)
    target_compile_options(gtda_witness_complex PUBLIC $<$<CONFIG:DEBUG>:-O2 -ggdb -D_GLIBCXX_DEBUG>)
endif()

#######################################################################
#                       Strong Witness Complex                        #
#######################################################################

pybind11_add_module(gtda_strong_witness_complex "${BINDINGS_DIR}/strong_witness_complex_bindings.cpp")

if(OpenMP_FOUND)
    target_link_libraries(gtda_strong_witness_complex PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(gtda_strong_witness_complex LINK_PUBLIC ${Boost_LIBRARIES})
target_compile_definitions(gtda_strong_witness_complex PRIVATE BOOST_RESULT_OF_USE_DECLTYPE=1 BOOST_ALL_NO_LIB=1 BOOST_SYSTEM_NO_DEPRECATED=1)

target_include_directories(gtda_strong_witness_complex PRIVATE "${GUDHI_SRC_DIR}/Witness_complex/include")
target_include_directories(gtda_strong_witness_complex PRIVATE "${GUDHI_SRC_DIR}/Simplex_tree/include")
target_include_directories(gtda_strong_witness_complex PRIVATE "${GUDHI_SRC_DIR}/Cech_complex/include")
target_include_directories(gtda_strong_witness_complex PRIVATE "${GUDHI_SRC_DIR}/Persistent_cohomology/include")
target_include_directories(gtda_strong_witness_complex PRIVATE "${GUDHI_SRC_DIR}/python/include")
target_include_directories(gtda_strong_witness_complex PRIVATE "${GUDHI_SRC_DIR}/common/include")

if(MSVC)
    target_compile_options(gtda_strong_witness_complex PUBLIC $<$<CONFIG:RELEASE>: /O2 /Wall /fp:strict>)
    target_compile_options(gtda_strong_witness_complex PUBLIC $<$<CONFIG:DEBUG>:/O1 /DEBUG:FULL /Zi /Zo>)
else()
    target_compile_options(gtda_strong_witness_complex PUBLIC $<$<CONFIG:RELEASE>: -Ofast -shared -pthread -fPIC -fwrapv -Wall -fno-strict-aliasing -frounding-math>)
    target_compile_options(gtda_strong_witness_complex PUBLIC $<$<CONFIG:DEBUG>:-O2 -ggdb -D_GLIBCXX_DEBUG>)
endif()

#######################################################################
#                             RipsComplex                             #
#######################################################################

pybind11_add_module(gtda_sparse_rips_complex "${BINDINGS_DIR}/rips_complex_bindings.cpp")

if(OpenMP_FOUND)
    target_link_libraries(gtda_sparse_rips_complex PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(gtda_sparse_rips_complex LINK_PUBLIC ${Boost_LIBRARIES})
target_compile_definitions(gtda_sparse_rips_complex PRIVATE BOOST_RESULT_OF_USE_DECLTYPE=1 BOOST_ALL_NO_LIB=1 BOOST_SYSTEM_NO_DEPRECATED=1)

target_include_directories(gtda_sparse_rips_complex PRIVATE "${GUDHI_SRC_DIR}/Simplex_tree/include")
target_include_directories(gtda_sparse_rips_complex PRIVATE "${GUDHI_SRC_DIR}/common/include")
target_include_directories(gtda_sparse_rips_complex PRIVATE "${GUDHI_SRC_DIR}/Cech_complex/include")
target_include_directories(gtda_sparse_rips_complex PRIVATE "${GUDHI_SRC_DIR}/Persistent_cohomology/include")
target_include_directories(gtda_sparse_rips_complex PRIVATE "${GUDHI_SRC_DIR}/Rips_complex/include")
target_include_directories(gtda_sparse_rips_complex PRIVATE "${GUDHI_SRC_DIR}/Subsampling/include")
target_include_directories(gtda_sparse_rips_complex PRIVATE "${GUDHI_SRC_DIR}/python/include")

if(MSVC)
    target_compile_options(gtda_sparse_rips_complex PUBLIC $<$<CONFIG:RELEASE>: /O2 /Wall /fp:strict>)
    target_compile_options(gtda_sparse_rips_complex PUBLIC $<$<CONFIG:DEBUG>:/O1 /DEBUG:FULL /Zi /Zo>)
else()
    target_compile_options(gtda_sparse_rips_complex PUBLIC $<$<CONFIG:RELEASE>: -Ofast -shared -pthread -fPIC -fwrapv -Wall -fno-strict-aliasing -frounding-math>)
    target_compile_options(gtda_sparse_rips_complex PUBLIC $<$<CONFIG:DEBUG>:-O2 -ggdb -D_GLIBCXX_DEBUG>)
endif()

#######################################################################
#                            Cech Complex                             #
#######################################################################

pybind11_add_module(gtda_cech_complex "${BINDINGS_DIR}/cech_complex_bindings.cpp")

if(OpenMP_FOUND)
    target_link_libraries(gtda_cech_complex PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(gtda_cech_complex LINK_PUBLIC ${Boost_LIBRARIES})
target_compile_definitions(gtda_cech_complex PRIVATE BOOST_RESULT_OF_USE_DECLTYPE=1 BOOST_ALL_NO_LIB=1 BOOST_SYSTEM_NO_DEPRECATED=1)

target_include_directories(gtda_cech_complex PRIVATE "${GUDHI_SRC_DIR}/Simplex_tree/include")
target_include_directories(gtda_cech_complex PRIVATE "${GUDHI_SRC_DIR}/Cech_complex/include")
target_include_directories(gtda_cech_complex PRIVATE "${GUDHI_SRC_DIR}/Persistent_cohomology/include")
target_include_directories(gtda_cech_complex PRIVATE "${GUDHI_SRC_DIR}/python/include")
target_include_directories(gtda_cech_complex PRIVATE "${GUDHI_SRC_DIR}/common/include")

if(MSVC)
    target_compile_options(gtda_cech_complex PUBLIC $<$<CONFIG:RELEASE>: /O2 /Wall /fp:strict>)
    target_compile_options(gtda_cech_complex PUBLIC $<$<CONFIG:DEBUG>:/O1 /DEBUG:FULL /Zi /Zo>)
else()
    target_compile_options(gtda_cech_complex PUBLIC $<$<CONFIG:RELEASE>: -Ofast -shared -pthread -fPIC -fwrapv -Wall -fno-strict-aliasing -frounding-math>)
    target_compile_options(gtda_cech_complex PUBLIC $<$<CONFIG:DEBUG>:-O2 -ggdb -D_GLIBCXX_DEBUG>)
endif()

