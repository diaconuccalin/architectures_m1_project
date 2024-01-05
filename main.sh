nvcc main.c src/tester.c src/graph_file_reader.c src/cu_bellman_ford.cu src/bellman_ford.c src/utils.c src/timing.c -o main_cuda
./main_cuda
