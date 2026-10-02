#!/bin/bash

### TC1 Job Script ###
 
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1

### Specify Memory allocate to this job ###
#SBATCH --mem=32G

### Optional: Default CPU assign = 1; Specify if want to apply more for computation###
### Remove 1st # at next line for the option to take effect ###
#SBATCH --ntasks-per-node=16

### Specify number of node to compute ###
#SBATCH --nodes=1

### Optional: Specify node to execute the job ###
### Remove 1st # at next line for the option to take effect ###
##SBATCH --nodelist=TC1N07

### Specify Time Limit, format: <min> or <min>:<sec> or <hr>:<min>:<sec> or <days>-<hr>:<min>:<sec> or <days>-<hr> ### 
#SBATCH --time=360

### Specify name for the job, filename format for output and error ###
#SBATCH --job-name=fairmappo_test
#SBATCH --output=/tc1home/FYP/n2501107d/localized-fair-marl/slurm_logs/outs/%x_%j.out
#SBATCH --error=/tc1home/FYP/n2501107d/localized-fair-marl/slurm_logs/errs/%x_%j.err

### Must load the required CUDA module if want to use available CUDA in TC1 for computation ###
module load cuda/12.9

# Computation script
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate localfair

PY=$CONDA_PREFIX/bin/python
for d in $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/lib; do
    export LD_LIBRARY_PATH=$d:$LD_LIBRARY_PATH
done
export PYTHONPATH="$(pwd):$PYTHONPATH"

$PY algorithms/train.py --algo IPPO --env cleanup

