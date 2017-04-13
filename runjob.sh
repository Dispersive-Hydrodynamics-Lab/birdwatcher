#!/bin/bash

#SBATCH --job-name ouroboros
#SBATCH --time 04:00:00
#SBATCH --nodes 4
#SBATCH --mem 20GB
#SBATCH --ntasks 8
#SBATCH --ntasks-per-node 2
#SBATCH --output ouroboros.out

python3.5 ./ouroboros.py -s 1 -n 8 -g 40
