
# Add to BOOST_ROOT variable a custom path to 
# ease installation of giotto-tda on Windows platform
# The custom path will be at `C:\\local\`
if(WIN32)
    list(APPEND BOOST_ROOT "C:/local")
    # azure-pipelines maintainers removed BOOST_ROOT
    # see https://github.com/actions/virtual-environments/commit/e91f2138c0b7304831e9aaa43ea47279d4160ef4#diff-55dcabaaca18e802a555dfb957187fbeL474
    list(APPEND BOOST_ROOT ${BOOST_ROOT_1_72_0})
    list(APPEND BOOST_ROOT "") # Add custom path to your boost installation
endif()

message(STATUS "BOOST_ROOT: ${BOOST_ROOT}")

find_package(Boost 1.56 REQUIRED)

message(STATUS "Boost_INCLUDE_DIR: ${Boost_INCLUDE_DIR}")
message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
