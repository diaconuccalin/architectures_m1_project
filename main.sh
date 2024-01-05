gcc -std=c99 -Wall -Wpedantic -fopenmp src/tester.c src/utils.c  src/graph_file_reader.c src/bellman_ford.c main.c -o bin/main_omp -lgomp
./bin/main_omp

nvcc main.c src/tester.c src/graph_file_reader.c src/cu_bellman_ford.cu src/bellman_ford.c src/utils.c src/timing.c -o bin/main_cuda
./bin/main_cuda
