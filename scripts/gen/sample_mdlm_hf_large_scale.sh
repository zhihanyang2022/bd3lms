#!/bin/bash

for t_val in {128,256}; do
    for seed in {1..10}; do
        sbatch scripts/gen/sample_mdlm_hf.sh $t_val 1 false 0.9 $seed
    done
done
