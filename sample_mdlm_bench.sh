python -u -m main \
    seed=1 \
    loader.eval_batch_size=1 \
    model=small \
    algo=mdlm \
    algo.T=4096 \
    algo.backbone=hf_dit \
    data=openwebtext-split \
    model.length=1024 \
    block_size=1024 \
    wandb=null \
    mode=sample_eval \
    eval.checkpoint_path=kuleshov-group/mdlm-owt-noeos \
    model.attn_backend=sdpa \
    sampling.nucleus_p=1.0 \
    sampling.kv_cache=false \
    sampling.logdir=/mnt/weka/home/zhihan.yang/samples/mdlm/bench \
    sampling.num_sample_batches=100 \
    sampling.num_tries=1 \
    sampling.first_hitting=false
echo