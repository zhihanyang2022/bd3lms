#!/bin/bash
#SBATCH -J sample_owt_mdlm
#SBATCH --partition=main
#SBATCH --output=slurm/%j_%x.out
#SBATCH --error=slurm/%j_%x.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append

# To enable preemption re-loading, set `hydra.run.dir` or 
# `checkpointing.save_dir` explicitly.

t_value=$1
num_tries_val=$2
first_hitting_val=$3
nucleus_p_val=$4
BLOCK_SIZE=1024 # 4, 8, 16
LENGTH=1024 # arbitrary; needs to be a multiple of the block size
seed_val=$5

python -u main.py \
    seed=${seed_val} \
    loader.eval_batch_size=1 \
    model=small \
    algo=mdlm \
    algo.T=${t_value} \
    algo.backbone=dit \
    data=openwebtext-split \
    model.length=$LENGTH \
    block_size=$BLOCK_SIZE \
    wandb=null \
    mode=sample_eval \
    eval.checkpoint_path=/mnt/weka/home/zhihan.yang/checkpoints/owt-mdlm-302860/checkpoints/14-250000.ckpt \
    model.attn_backend=sdpa \
    sampling.nucleus_p=${nucleus_p_val} \
    sampling.kv_cache=false \
    sampling.logdir=/mnt/weka/home/zhihan.yang/samples/mdlm/blocksize_${BLOCK_SIZE}_tval_${t_value}_numtries_${num_tries_val}_nucleus_${nucleus_p_val}_firsthit_${first_hitting_val}/seed_${seed_val} \
    sampling.num_sample_batches=100 \
    sampling.num_tries=${num_tries_val} \
    sampling.first_hitting=${first_hitting_val}
