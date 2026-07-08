#!/bin/bash

# --- 1. Resource Request ---
#SBATCH -N 6             
#SBATCH -n 336           
#SBATCH -p cp6           

# --- 2. environment ---
module purge
module load Intel_compiler/19.0.4
module load MKL/19.1.2
module load MPI/mpich/4.0.2-mpi-x-icc19.0


PYTHON_PATH=$HOME/miniconda3/envs/SHGmodel/bin/python


SCRIPT_NAME=Sipe2D.py

echo "Job Start Time: $(date)"
echo "Using Python interpreter: $PYTHON_PATH"


yhrun $PYTHON_PATH -u $SCRIPT_NAME > shg_calc.log 2>&1

echo "Job End Time: $(date)"
