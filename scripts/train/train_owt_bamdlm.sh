#!/bin/bash
#SBATCH -J train_owt_bamdlm                # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32G                  # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu          # Request partition
#SBATCH --constraint="[a5000|a6000|a100|3090]"
#SBATCH --constraint="gpu-mid|gpu-high"
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

BLOCK_SIZE=16
CKPT_PATH=mdlm_pretrain_owt_850k.ckpt

python -u main.py \
    loader.global_batch_size=512 \
    loader.eval_global_batch_size=512 \
    loader.batch_size=8 \
    loader.eval_batch_size=8 \
    model=small \
    algo=bamdlm \
    data=openwebtext-split \
    model.length=1024 \
    block_size=${BLOCK_SIZE} \
    wandb.name=bamdlm-owt-block_size${BLOCK_SIZE} \
    mode=train \
    model.attn_backend=sdpa \
    training.from_pretrained=$PRETRAIN_CKPT