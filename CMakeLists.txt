cmake_minimum_required(VERSION 3.22)
project(architectures_m1_project C CUDA)

set(CMAKE_C_STANDARD 11)

add_executable(architectures_m1_project main_openmp.c
        src/graph_structs.h
        src/graph_file_reader.c
        src/graph_file_reader.h
        src/utils.c
        src/utils.h
        src/bellman_ford.c
        src/bellman_ford.h
        src/tester.c
        src/tester.h
        main_cuda.cu
)
