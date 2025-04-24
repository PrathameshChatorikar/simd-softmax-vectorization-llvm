cmake_minimum_required(VERSION 3.5)

set(CMAKE_C_COMPILER clang CACHE STRING "Clang compiler") 
set(CMAKE_CXX_COMPILER clang++ CACHE STRING "Clang++ compiler")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Generate compile commands")

set(CMAKE_CXX_STANDARD 23 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_C_STANDARD 23 CACHE STRING "C++ standard to conform to")
set(CMAKE_C_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Debug)

project("simd")

set(CMAKE_VERBOSE_MAKEFILE ON)
# if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
#   message(STATUS "Using AVX for x86_64 architecture")
#   file(GLOB_RECURSE SIMD_CPP SIMD/vectorAVX.cpp)
# elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
#   message(STATUS "Using NEON for ARM architecture")
#   file(GLOB_RECURSE SIMD_CPP SIMD/vectorNeon.cpp)
# endif()

# If X86 Architecture, use vectorAVX.cpp

file(GLOB_RECURSE SIMD_C SIMD/*.c)
file(GLOB_RECURSE SIMD_CPP SIMD/*.cpp)
add_executable(simd main.cpp ${SIMD_C} ${SIMD_CPP} SIMD/scalars.cpp)
# target_link_options(simd PRIVATE -fmodules-ts -x c++-system-header iostream)
# target_compile_options(simd PRIVATE -fmodules-ts -x c++-system-header iostream)