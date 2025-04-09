#!/bin/bash 
#SBATCH -N 3
#SBATCH --ntasks-per-node=48
#SBATCH -p hdr
#SBATCH -t 168:00:00
#SBATCH -o SOLUTION.DAT
#SBATCH -e FAILURE.ERR
#SBATCH --mail-type=BEGIN,END #Mail when job starts and ends
#SBATCH --mail-user=prateekr@mit.edu
#SBATCH --job-name=YP10N1


#module load anaconda3/2020.11
#module load python/3.9.1
module load python/3.11.0
module load gcc/11.3.0
module load openmpi/4.1.5

python3 RunOptimIP.py
#mpirun -np 96 SU2_CFD turb_B738.cfg > log.CFD
