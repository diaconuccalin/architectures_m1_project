mkdir bin

nvcc -Xcompiler -fopenmp main.c src/tester.c src/graph_file_reader.c src/cu_bellman_ford.cu src/bellman_ford.c src/utils.c src/timing.c -o bin/main -lgomp

OMP_NUM_THREADS=4 ./bin/main openmp "/home/students/calin.diaconu/parallel_bellman_ford/data/infoarena_processed" "/home/students/calin.diaconu/parallel_bellman_ford/data/infoarena"
./bin/main cuda "/home/students/calin.diaconu/parallel_bellman_ford/data/infoarena_processed" "/home/students/calin.diaconu/parallel_bellman_ford/data/infoarena"
