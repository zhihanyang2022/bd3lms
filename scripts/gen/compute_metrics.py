import os
import sys
import json

import torch
from omegaconf import OmegaConf
from hydra import compose, initialize

sys.path.insert(0, os.getcwd())
from metrics import Metrics

# Register resolvers
OmegaConf.register_new_resolver('device_count', lambda: torch.cuda.device_count())
OmegaConf.register_new_resolver('eval', lambda x: eval(x))
OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)

# works for all samples on OWT (assuming a generation length of 1024)

def compute_metrics(samples_dir):

    print(samples_dir)

    with initialize(version_base=None, config_path="../../configs"):
        config = compose(
            config_name="config", 
            overrides=[
                "data=openwebtext-split"])  # required to get the right tokenizer
    
    assert config.model.length == 1024
  
    # during init, Metrics set the following two important attributes:
    # - self.eval_ppl_batch_size as config.eval.perplexity_batch_size (8 by default)
    # - self.gen_ppl_eval_model_name_or_path as config.eval.gen_ppl_eval_model_name_or_path (gpt2-large by default)
    # nothing else matters

    metrics = Metrics(config=config)

    samples = []
    for fname in os.listdir(samples_dir):
        print(fname)
        fpath = os.path.join(samples_dir, fname)
        with open(fpath, 'r') as f:
            loaded_dict = json.load(f)
        samples_subset = loaded_dict['samples']
        # each sample is like [sample], so we need to unwrap it
        samples.extend([sample[0] for sample in samples_subset])

    # with open('/mnt/weka/home/zhihan.yang/bd3lms/128_5k.json', 'r') as f:
    #     samples = json.load(f)

    print(f"Loaded {len(samples)} samples")

    samples = samples[:1000]

    # gen ppl
    
    metrics.reset()
    metrics.to('cuda')
    metrics.record_generative_perplexity(
        samples,
        config.model.length,
        8,  # overwrites self.eval_ppl_batch_size
        retokenize=True)  
    gen_ppl = metrics.gen_ppl.compute().cpu().item()
    print(f"Generative perplexity: {gen_ppl:.2f}")

    return
    
    # entropy

    tokenizer = None

    entropies = []
    for sample in samples:
        token_ids = torch.tensor(tokenizer.encode(sample, add_special_tokens=False))
        counts = torch.unique(token_ids, return_counts=True)[1]
        entropy = torch.special.entr(counts.float() / counts.sum()).sum().item()
        entropies.append(entropy)
    avg_entropy = sum(entropies) / len(entropies)
    print(f"Average entropy: {avg_entropy:.2f}")

    # mauve

    # TODO

    # save results
    
    results = {
        "gen_ppl": gen_ppl,
        "entropy": avg_entropy,
        "avg_nfe": avg_nfe,
        "mauve": mauve_score
    }
    output_metrics_file = os.path.join(output_dir, f"_res_metrics_nucleus_{p}_t_{t}_numtries_{num_tries_val}_block_{block_size}_firsthit_{first_hit_val}.json")
    
    # TODO

    with open(output_metrics_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved metrics to {output_metrics_file}")

if __name__ == "__main__":
    base_dir = "/mnt/weka/home/zhihan.yang/samples/owt-mdlm-noeos-1m/"
    samples_dir = base_dir + "blocksize_1024_tval_128_numtries_1_nucleus_0.9_firsthit_false"
    compute_metrics(samples_dir)
