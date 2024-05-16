cmake_minimum_required(VERSION 3.10)

# Project name and version
project(kernel_debug CUDA CXX)

# Enable C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CUDA_ARCHITECTURES 75)  # Example for NVIDIA Turing architecture
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")

# Set build type to Debug by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# Find CUDA package
find_package(CUDA REQUIRED)

# Add the source files
add_executable(kernel_debug kernel_debug.cu)

# Set CUDA properties for the target
set_target_properties(kernel_debug PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
