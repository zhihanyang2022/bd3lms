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

T=$1
SEED=$2

echo $T
echo $SEED

# use small batch size to max utilize p_x0 cache

srun python -u main.py \
    loader.eval_batch_size=512 \
    model=small \
    algo=bd3lm \
    algo.T=$T \
    algo.backbone=dit \
    data=openwebtext-split \
    model.length=1024 \
    block_size=16 \
    wandb=null \
    mode=sample_eval \
    eval.checkpoint_path=/lustrefs/users/zhuojun.cheng/checkpoints/blockdiffusion-250k-10909/checkpoints/last.ckpt \
    model.attn_backend=sdpa \
    seed=$SEED \
    sampling.first_hitting=false \
    sampling.num_sample_batches=10 \
    sampling.nucleus_p=0.9 \
    sampling.kv_cache=true \
    sampling.profile_throughput=false \
    sampling.logdir=$PWD/logs/samples_5120/samples_bd3lm_blocksize16_T${T}_seed${SEED}
