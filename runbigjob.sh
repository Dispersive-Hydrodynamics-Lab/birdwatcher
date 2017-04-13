#!/bin/bash

#SBATCH --job-name ouroboros
#SBATCH --time 08:00:00
#SBATCH --nodes 8
#SBATCH --mem 20GB
#SBATCH --ntasks 64
#SBATCH --ntasks-per-node 8
#SBATCH --output ouroboros.out

python3.5 ./ouroboros.py -s 8 -n 32
