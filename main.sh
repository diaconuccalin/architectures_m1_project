gcc -std=c99 -Wall -Wpedantic -fopenmp src/tester.c src/utils.c  src/graph_file_reader.c src/bellman_ford.c main.c -o main_omp -lgomp
./main_omp