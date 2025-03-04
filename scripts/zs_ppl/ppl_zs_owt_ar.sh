#!/bin/bash
#SBATCH -J ppl_zs_owt_ar              # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -e watch_folder/%x_%j.err     # error log file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32G                  # server memory requested (per node)
#SBATCH -t 96:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=kuleshov,gpu         # Request partition
# --constraint="gpu-mid|gpu-high"
# --constraint="[r6000|a5000|a6000|3090|a100|a40]"
#SBATCH --constraint="[a5000|a6000]"
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                 # Type/number of GPUs needed
#SBATCH --cpus-per-task=1              # Number of CPU cores per task
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption
#SBATCH --exclude=brandal,davis-compute-02,ellis-compute-01,yu-compute-01,abdelfattah-compute-02,davis-compute-01,lancer-compute-01,snavely-compute-03,rush-compute-02,rush-compute-03,ellis-compute-02

# This YAML is intended to serve as a reference for running on the MosaicML platform
# The config from `yamls/main/mosaic-bert-base-uncased.yaml` is copied into the `parameters` field below.
# You can copy/modify the contents of this file to run different workloads, following the other
# examples in this directory.
#
# Note that some of the fields in this template haven't been filled in yet.
# Please resolve any `null` fields before launching!
#
# When ready, use `mcli run -f yamls/main/mcloud_run_a100_80gb.yaml` to launch

datasets=("ag_news"
          "scientific_papers_pubmed"
          "scientific_papers_arxiv"
          "lambada"
          "wikitext2"
          "wikitext103"
          "ptb"
          "lm1b-gpt2")

for data in "${datasets[@]}"; do
    echo "$data"
    srun python -u main.py \
        loader.eval_batch_size=16 \
        model=small \
        algo=ar \
        data=$data \
        +data.insert_valid_eos=False \
        model.length=1024 \
        eval.checkpoint_path=/share/kuleshov/ssahoo/textdiffusion/text-diffusion-exp-v4-AgBZrc-small-ar-param-ar_data-openwebtext-split_seqlen-1024_maxs-1300001_bs-512/checkpoints/last.ckpt \
        wandb=null \
        mode=ppl_eval > $PWD/logs/ar_$data.log
done