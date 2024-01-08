# Parallel Bellman-Ford
## Purpose
This project aims to analyze the performance of different parallel implementations in OpenMP and CUDA libraries for the
Bellman-Ford algorithm. It is the project developed for the first module of the Architectures and Platforms for
Artificial Intelligence, part of the second year of the Artificial Intelligence Master at Università di Bologna.

## How to Run
On the SLURM server, launch a batch task with command "sbatch project.sbatch" from the root directory. The paths to the test data are already set to a relative path in the bash script, but can be adjusted, as they are passed through arguments. Once launched, the project is built, and two instances will run sequentially, one with the OpenMP implementation, and then one with the CUDA implementation. Both will run through all available tests in the input directory, and should output information through standard output to a "output.txt" file. The execution time is around 6 minutes. The variables in the project, like the scheduling strategy and the chunk size, are set to the best ones resulted from the evaluation, as described in the pdf report.
