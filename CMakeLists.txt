cmake_minimum_required(VERSION 3.26)
project(architectures_m1_project C)

set(CMAKE_C_STANDARD 11)

add_executable(architectures_m1_project main.c
        src/graph_structs.h
        src/graph_file_reader.c
        src/graph_file_reader.h
        src/utils.c
        src/utils.h
        src/bellman_ford.c
        src/bellman_ford.h
)
