#!/bin/bash
#SBATCH -J eval_owt_bd3lm
#SBATCH --partition=main
#SBATCH --output=slurm/%j_%x.out
#SBATCH --error=slurm/%j_%x.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --open-mode=append

# To enable preemption re-loading, set `hydra.run.dir` or 
# `checkpointing.save_dir` explicitly.

DATA_DIR=${HOME}/data/esolm

srun python -u main.py \
    loader.eval_batch_size=32 \
    model=small \
    model.attn_backend=sdpa \
    algo=mdlm \
    algo.backbone=hf_dit \
    algo.ignore_bos=true \
    data=openwebtext-split \
    data.insert_valid_special=False \
    model.length=1024 \
    eval.checkpoint_path=kuleshov-group/mdlm-owt-noeos \
    wandb=null \
    data.cache_dir=${DATA_DIR} \
    mode=ppl_eval > $PWD/logs/mdlm_owt_noes.log
