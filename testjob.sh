#!/bin/bash

#SBATCH --job-name ouroboros
#SBATCH --time 00:30
#SBATCH --nodes 1
#SBATCH --mem 1GB
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node 1
#SBATCH --output ouroboros.out

python3.5 ./ouroboros.py -t
