cmake_minimum_required(VERSION 3.14)
project(neighbor_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MSMPI_ROOT "C:/Program Files (x86)/Microsoft SDKs/MPI")
set(MSMPI_INCLUDE_DIR "${MSMPI_ROOT}/Include")
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(MSMPI_LIB_DIR "${MSMPI_ROOT}/Lib/x64")
else()
    set(MSMPI_LIB_DIR "${MSMPI_ROOT}/Lib/x86")
endif()

find_library(MSMPI_LIB msmpi PATHS "${MSMPI_LIB_DIR}" REQUIRED)

add_executable(benchmark_parallel benchmark_parallel.cpp)
add_executable(benchmark_grid_parallel benchmark_grid_parallel.cpp)

foreach(target_name benchmark_parallel benchmark_grid_parallel)
    target_include_directories(${target_name} PRIVATE "${MSMPI_INCLUDE_DIR}")
    target_link_libraries(${target_name} PRIVATE "${MSMPI_LIB}")
    if(MSVC)
        target_compile_options(${target_name} PRIVATE /openmp)
    else()
        target_compile_options(${target_name} PRIVATE -fopenmp)
        target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)
    endif()
endforeach()
