#!/bin/bash
#SBATCH -N 4
#SBATCH -n 224
#SBATCH -p cp7
#SBATCH -J phase_T1a_full
#SBATCH -o phase_T1a_full.%j.out
#SBATCH -e phase_T1a_full.%j.err

module purge
module load Intel_compiler/19.0.4
module load MKL/19.1.2
module load MPI/mpich/4.0.2-mpi-x-icc19.0

PYTHON_PATH=$HOME/miniconda3/envs/SHGmodel/bin/python
SCRIPT_NAME=T1a.py

yhrun $PYTHON_PATH -u $SCRIPT_NAME
