cmake_minimum_required(VERSION 3.26)

project(architectures_m1_project LANGUAGES C)

set(CMAKE_C_STANDARD 11)

find_package(OpenMP COMPONENTS C REQUIRED)

add_executable(architectures_m1_project main.c
        src/graph_structs.h
        src/graph_file_reader.c
        src/graph_file_reader.h
        src/utils.c
        src/utils.h
        src/bellman_ford.c
        src/bellman_ford.h
        src/tester.c
        src/tester.h
)

target_link_libraries(architectures_m1_project PRIVATE OpenMP::OpenMP_C)
