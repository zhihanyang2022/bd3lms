# batch sampling jobs
for SEED in $(seq 1 50); do
  for LEN in 1024 2048; do
    sbatch ./scripts/gen_ppl/genppl_bamdlm.sh $LEN $SEED 16
    sbatch ./scripts/gen_ppl/genppl_bamdlm.sh $LEN $SEED 8
    sbatch ./scripts/gen_ppl/genppl_bamdlm.sh $LEN $SEED 4
    sbatch ./scripts/gen_ppl/genppl_mdlm.sh $LEN $SEED
    sbatch ./scripts/gen_ppl/genppl_ar.sh $LEN $SEED    
  done
  sbatch ./scripts/gen_ppl/genppl_sedd.sh $SEED
done

python ./scripts/log_sample_stats.py