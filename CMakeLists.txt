cmake_minimum_required(VERSION 2.8)

project(rdma)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

# if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
#     set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/bin/rdma_lib)
# endif()

# for GDB debugging
set(CMAKE_BUILD_TYPE Debug)

# Compiler Options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLINUX -DDEBUG -std=c++14 -Wall -O3 -Wextra ")

# the compiler flags for compiling C++ sources
MESSAGE( STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS} )

add_subdirectory(src)
add_subdirectory(test)

file(COPY
    ${CMAKE_CURRENT_SOURCE_DIR}/src/conf/RDMA.conf
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin/conf/
)
