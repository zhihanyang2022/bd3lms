import json
import transformers
import sys
import os
sys.path.append('../')

import torch
from tqdm import tqdm
import metrics
from omegaconf import OmegaConf

config_dict = {
      'block_size': 25,
      'algo': 'ssd-lm',
      'training': {
          'importance_sampling': False,
          'sampling_eps': 1e-3,
      },
      'eval': {
          'perplexity_batch_size': 1,
          'gen_ppl_eval_model_name_or_path': 'gpt2-large'
      }
  }
  
tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
METRIC_CONFIG = OmegaConf.create(config_dict)
METRICS = metrics.Metrics(METRIC_CONFIG)
METRICS.to('cuda')

seqlen=2050 # 1025, 2050
total_t = 25 # 25, 1000
for seqlen in {1025, 2050}:
  for total_t in {25, 1000}:
    print(f'sequence length {seqlen}, total timesteps {total_t}, num samples {len(METRICS.gen_ppls)}')
    METRICS.reset()
    for seed in tqdm(range(1,51)):
        path = f'/home/ma2238/sar_os/text-diffusion/ssd-lm/generated_samples/samples_openwebtext-valid_bs25_l{seqlen}_t{total_t}_SEED{seed}'
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        text = data['generated_samples'][0][f'sample_0']['sampled_sequences']
        METRICS.record_generative_perplexity(text, max_length=seqlen)
        if len(METRICS.gen_ppls) == 50:
            break

    print('generative perplexity:', torch.tensor(METRICS.gen_ppls).mean().item())
    print('entropy:', torch.tensor(METRICS.gen_entropies).mean().item())