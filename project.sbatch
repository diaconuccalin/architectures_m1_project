#!/bin/bash
#SBATCH --job-name=A1_project_Calin_Diaconu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=calin.diaconu@unibo.it
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=output.txt
#SBATCH --gres=gpu:1

./main.sh
