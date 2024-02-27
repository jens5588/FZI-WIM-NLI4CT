#!/bin/bash
#SBATCH --job-name=inf5500
#SBATCH --partition="advanced"
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres="gpu:full:4"
#SBATCH --time=12:00:00

source /home/abc/anaconda3/etc/profile.d/conda.sh
conda activate llama_recipes
export PYTHONPATH=$PYTHONPATH:/home/abc/FZI-WIM-NLI4CT
python3 fsdp/inference/run_test_fsdp.py --start_index=0 --end_index=5500