#!/bin/sh
#
#SBATCH --job-name="MeOH_reform"
#SBATCH --partition=compute
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=research-me-mtt

module load 2023r1
module load python/3.8.12
module load py-numpy
module load 2023r1-gcc11
module load 2023r1-intel
module load 2023rc1
module load 2023rc1-gcc11
module load py-pandas/1.5.1
module load py-scipy/1.8.1

srun python reformer_solver.py > solver.txt

