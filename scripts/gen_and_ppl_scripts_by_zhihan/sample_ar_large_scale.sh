#!/bin/bash

for seed in {1..10}; do
    sbatch scripts/gen/sample_ar.sh 1 0.9 $seed
done
