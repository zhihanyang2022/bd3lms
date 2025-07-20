#!/bin/bash
#SBATCH -J gen_owt_bd3lm
#SBATCH --partition=main
#SBATCH --output=slurm/%j_%x.out
#SBATCH --error=slurm/%j_%x.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append

# To enable preemption re-loading, set `hydra.run.dir` or 
# `checkpointing.save_dir` explicitly.

LENGTH=$1
SEED=$2
BLOCK_SIZE=$3
T=$4

echo $T

srun python -u main.py \
    loader.eval_batch_size=1 \
    model=small \
    algo=bd3lm \
    algo.T=$T \
    algo.backbone=dit \
    data=openwebtext-split \
    model.length=$LENGTH \
    block_size=$BLOCK_SIZE \
    wandb=null \
    mode=sample_eval \
    eval.checkpoint_path=/mnt/weka/home/zhihan.yang/checkpoints/bd3lm-owt-block_size4-309081/checkpoints/14-250000.ckpt \
    model.attn_backend=sdpa \
    seed=$SEED \
    sampling.first_hitting=false \
    sampling.num_sample_batches=2 \
    sampling.nucleus_p=0.9 \
    sampling.kv_cache=true \
    sampling.profile_throughput=false \
    sampling.logdir=$PWD/sample_logs/samples_bd3lm_len${LENGTH}_blocksize${BLOCK_SIZE}
