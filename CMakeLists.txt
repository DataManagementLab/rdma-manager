cmake_minimum_required(VERSION 2.8)

project(rdma)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

# if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
#     set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/bin/rdma_lib)
# endif()

# for GDB debugging
# set(CMAKE_BUILD_TYPE Debug)

# Compiler Options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLINUX -std=c++17 -Wall -Wextra -DHUGEPAGE")

# If OFF no CUDA related classes will be compiled to support systems without CUDA devices
option(CUDA_ENABLED "CUDA support" ON)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLINUX -DDEBUG")
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()



# the compiler flags for compiling C++ sources
MESSAGE( STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS} )

if (CUDA_ENABLED)
    add_definitions(-DCUDA_ENABLED)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lcuda")
    MESSAGE(STATUS "CUDA SUPPORT:  ENABLED")
else()
    MESSAGE(STATUS "CUDA SUPPORT:  DISABLED")
endif()

add_subdirectory(src)
add_subdirectory(gtest)
add_subdirectory(perftest)
# add_subdirectory(benchmark)


unset(CUDA_ENABLED CACHE)