#!/bin/bash
#SBATCH -J train_bd3lm
#SBATCH --partition=main
#SBATCH --output=slurm/%j_%x.out
#SBATCH --error=slurm/%j_%x.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --open-mode=append

BLOCK_SIZE=8 # we recommend 4, 8, or 16. must be a factor of the context length
DATA_DIR=${HOME}/data/esolm
RUN_NAME=bd3lm-owt-block_size${BLOCK_SIZE}-${SLURM_JOB_ID}
CHECKPOINT_DIR=${HOME}/checkpoints/${RUN_NAME}

srun python -u main.py \
    loader.global_batch_size=512 \
    loader.eval_global_batch_size=512 \
    loader.batch_size=32 \
    loader.eval_batch_size=32 \
    model=small \
    algo=bd3lm \
    algo.clip_search_widths=[0.5,0.6,0.7,0.8,0.9] \
    data=openwebtext-split \
    data.insert_train_special=False \
    data.insert_valid_special=False \
    data.cache_dir=$DATA_DIR \
    model.length=1024 \
    block_size=$BLOCK_SIZE \
    wandb.name=$RUN_NAME \
    +wandb.entity=sahoo-diffusion \
    wandb.project=ELMO \
    mode=train \
    model.attn_backend=flex \
    training.resample=True \
    trainer.max_steps=250000 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=50000 \
    eval.generate_samples=False \
    hydra.run.dir=${CHECKPOINT_DIR}
