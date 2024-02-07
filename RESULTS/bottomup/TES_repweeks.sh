#!/bin/bash

# TES_repweeks

# run with: sbatch submit_sbatch.sh

# Loading the required modules
source /etc/profile
module load anaconda/2021b
module load julia/1.7.3
module load gurobi/gurobi-951

# Initialize and Load Modules
source /etc/profile
module load anaconda/2021b


echo "My task ID: " $LLSUB_RANK
echo "Number of Tasks: " $LLSUB_SIZE


# Run script 
julia TES_script_repweeks.jl $LLSUB_RANK $LLSUB_SIZE