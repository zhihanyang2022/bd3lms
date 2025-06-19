Ts=( 8 16 32 64 128 1024 4096 )

for T in "${Ts[@]}"; do
    python -u -m main \
        seed=1 \
        loader.eval_batch_size=1 \
        model=small \
        algo=mdlm \
        algo.T="$T" \
        algo.backbone=hf_dit \
        data=openwebtext-split \
        model.length=1024 \
        block_size=1024 \
        wandb=null \
        mode=sample_eval \
        eval.checkpoint_path=kuleshov-group/mdlm-owt-noeos \
        model.attn_backend=sdpa \
        sampling.nucleus_p=0.9 \
        sampling.kv_cache=false \
        sampling.logdir=/mnt/weka/home/zhihan.yang/samples/mdlm/bench \
        sampling.num_sample_batches=2 \
        sampling.num_tries=1 \
        sampling.first_hitting=false
    echo
done