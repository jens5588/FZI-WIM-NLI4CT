#!/bin/bash
#SBATCH --job-name=llama_recipes
#SBATCH --ntasks=4
#SBATCH --nodes=4
#SBATCH --gres="gpu:full:4"
#SBATCH --mem="500000mb"
#SBATCH --partition=advanced
#SBATCH --time=12:00:00

module load system/ssh_wrapper
module load devel/cuda/11.8

source /home/abc/anaconda3/etc/profile.d/conda.sh
conda activate llama_recipes
export PYTHONPATH=$PYTHONPATH:/home/abc/FZI-WIM-NLI4CT

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
# Enable for A100
export FI_PROVIDER="efa"

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
# debugging flags (optional)
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INFO
export PYTHONFAULTHANDLER=1
#export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=0


# on your cluster you might need these:
# set the network interface
#export NCCL_SOCKET_IFNAME="eno1"
#export FI_EFA_USE_DEVICE_RDMA=1

srun torchrun --nproc_per_node 4 --nnodes 4  --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 fsdp/lora_finetuning.py  --enable_fsdp --use_peft --peft_method lora --low_cpu_fsdp