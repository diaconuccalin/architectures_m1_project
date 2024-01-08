mkdir bin

nvcc -Xcompiler -fopenmp main.c src/tester.c src/graph_file_reader.c src/cu_bellman_ford.cu src/bellman_ford.c src/utils.c src/timing.c -o bin/main -lgomp

INPUT_FILES_PATH="./data/infoarena_processed"
OK_FILES_PATH="./data/infoarena"

OMP_NUM_THREADS=4 ./bin/main openmp $INPUT_FILES_PATH $OK_FILES_PATH
./bin/main cuda $INPUT_FILES_PATH $OK_FILES_PATH
