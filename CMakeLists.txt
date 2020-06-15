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

    # ThreadSanitizer
    set(CMAKE_C_FLAGS_TSAN "-fsanitize=thread -g -O1" CACHE STRING "Flags used by the C compiler during ThreadSanitizer builds." FORCE)
    set(CMAKE_CXX_FLAGS_TSAN "-fsanitize=thread -g -O1" CACHE STRING "Flags used by the C++ compiler during ThreadSanitizer builds." FORCE)

    # AddressSanitize
    set(CMAKE_C_FLAGS_ASAN "-fsanitize=address -fno-optimize-sibling-calls -fsanitize-address-use-after-scope -fno-omit-frame-pointer -g -O1" CACHE STRING "Flags used by the C compiler during AddressSanitizer builds." FORCE)
    set(CMAKE_CXX_FLAGS_ASAN "-fsanitize=address -fno-optimize-sibling-calls -fsanitize-address-use-after-scope -fno-omit-frame-pointer -g -O1" CACHE STRING "Flags used by the C++ compiler during AddressSanitizer builds." FORCE)

    # LeakSanitizer
    set(CMAKE_C_FLAGS_LSAN "-fsanitize=leak -fno-omit-frame-pointer -g -O1" CACHE STRING "Flags used by the C compiler during LeakSanitizer builds." FORCE)
    set(CMAKE_CXX_FLAGS_LSAN "-fsanitize=leak -fno-omit-frame-pointer -g -O1" CACHE STRING "Flags used by the C++ compiler during LeakSanitizer builds." FORCE)

    # MemorySanitizer
    set(CMAKE_C_FLAGS_MSAN "-fsanitize=memory -fno-optimize-sibling-calls -fsanitize-memory-track-origins=2 -fno-omit-frame-pointer -g -O2" CACHE STRING "Flags used by the C compiler during MemorySanitizer builds." FORCE)
    set(CMAKE_CXX_FLAGS_MSAN "-fsanitize=memory -fno-optimize-sibling-calls -fsanitize-memory-track-origins=2 -fno-omit-frame-pointer -g -O2" CACHE STRING "Flags used by the C++ compiler during MemorySanitizer builds." FORCE)

    # UndefinedBehaviour
    set(CMAKE_C_FLAGS_UBSAN "-fsanitize=undefined" CACHE STRING "Flags used by the C compiler during UndefinedBehaviourSanitizer builds." FORCE)
    set(CMAKE_CXX_FLAGS_UBSAN "-fsanitize=undefined" CACHE STRING "Flags used by the C++ compiler during UndefinedBehaviourSanitizer builds." FORCE)
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()



# setup CUDA support (if enabled)
if (CUDA_ENABLED)
    enable_language(CUDA)
    find_package(CUDA 10.1 REQUIRED)
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lcudart")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lcudart")
    add_definitions(-DCUDA_ENABLED)
    MESSAGE(STATUS " CUDA SUPPORT:    ENABLED")
else()
    MESSAGE(STATUS " CUDA SUPPORT:    DISABLED")
endif()

# the compiler flags for compiling C++ sources
MESSAGE( STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS} )

add_subdirectory(src)
add_subdirectory(gtest)
add_subdirectory(perftest_old)
add_subdirectory(perftest)


unset(CUDA_ENABLED CACHE)