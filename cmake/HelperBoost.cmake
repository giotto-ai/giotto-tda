
# Add to BOOST_ROOT variable a custom path to 
# ease installation of giotto-tda on Windows platform
# The custom path will be at `C:\\local\`
if(WIN32)
    list(APPEND BOOST_ROOT "C:/local")
    list(APPEND BOOST_ROOT "") # Add custom path to your boost installation
endif()

# Changes induced in latest version of the azure pipelines
# Produces compilations errors because It cannot find boost header files
# After discussing in https://github.com/actions/virtual-environments/issues/687
# This solution is used due to custom paths in the azure pipeline
message(STATUS "BOOST_ROOT_PIPELINE: $ENV{BOOST_ROOT_PIPELINE}")
if(DEFINED $ENV{BOOST_ROOT_PIPELINE})
    list(APPEND BOOST_ROOT "$ENV{BOOST_ROOT_PIPELINE}")
    list(APPEND BOOST_INCLUDEDIR "$ENV{BOOST_ROOT_PIPELINE}\boost\include")
    list(APPEND BOOST_LIBRARYDIR "$ENV{BOOST_ROOT_PIPELINE}\lib")
endif()

message(STATUS "BOOST_ROOT: ${BOOST_ROOT}")

find_package(Boost 1.56 REQUIRED)

message(STATUS "Boost_INCLUDE_DIR: ${Boost_INCLUDE_DIR}")
message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
