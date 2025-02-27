"""BAMDLM model for Hugging Face.

"""
import math
import typing

import einops
import flash_attn
import flash_attn.layers.rotary
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import modeling_outputs

from .configuration_bamdlm import BAMDLMConfig

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

def block_causal_mask(num_rows, block_size, mode='full', offset=0):
  mask = block_size * torch.arange(
    1, num_rows // block_size + 1).unsqueeze(1).tile(block_size).flatten().unsqueeze(1)
  if mode == 'full':
    mask = (mask >= mask.T + offset)
  elif mode == 'diag':
    mask = (mask + offset == mask.T)
  elif mode == 'triu_diag':
    mask = torch.zeros(num_rows, num_rows)
    rows = torch.arange(0, num_rows)
    group_indices = rows // (block_size)
    column_indices = group_indices * (block_size) + block_size + offset
    valid_rows = column_indices < num_rows
    mask[rows[valid_rows].unsqueeze(1), column_indices[valid_rows].unsqueeze(1)] = 1
  return mask.int()

def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool) -> torch.Tensor:
  if bias is not None:
    out = scale * F.dropout(x + bias, p=prob, training=training)
  else:
    out = scale * F.dropout(x, p=prob, training=training)

  if residual is not None:
    out = residual + out
  return out


def get_bias_dropout_add_scale(training):
  def _bias_dropout_add(x, bias, scale, residual, prob):
    return bias_dropout_add_scale(
      x, bias, scale, residual, prob, training)

  return _bias_dropout_add


# function overload
def modulate(x: torch.Tensor,
             shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
  return x * (1 + scale) + shift


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
  return modulate(x, shift, scale)


class Rotary(torch.nn.Module):
  def __init__(self, dim, base=10_000):
    super().__init__()
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    self.register_buffer('inv_freq', inv_freq)
    self.seq_len_cached = None
    self.cos_cached = None
    self.sin_cached = None

  def forward(self, x, seq_dim=1):
    seq_len = x.shape[seq_dim]
    if seq_len != self.seq_len_cached:
      self.seq_len_cached = seq_len
      t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
      freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
      emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
      # dims are: batch, seq_len, qkv, head, dim
      self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1,1,3,1,1)
      self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1,1,3,1,1)
      # This makes the transformation on v an identity.
      self.cos_cached[:,:,2,:,:].fill_(1.)
      self.sin_cached[:,:,2,:,:].fill_(0.)

    return self.cos_cached, self.sin_cached


def rotate_half(x):
  x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_torchscript(qkv, cos, sin):
    return (qkv * cos) + (rotate_half(qkv) * sin)

def apply_rotary_pos_emb(qkv, cos, sin):
  cos = cos[0,:,0,0,:cos.shape[-1]//2]
  sin = sin[0,:,0,0,:sin.shape[-1]//2]
  return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)


# function overload
def modulate(x, shift, scale):
  return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.weight = nn.Parameter(torch.ones([dim]))
    self.dim = dim
  def forward(self, x):
    with torch.cuda.amp.autocast(enabled=False):
      x = F.layer_norm(x.float(), [self.dim])
    return x * self.weight[None,None,:]


def residual_linear(x, W, x_skip, residual_scale):
  """x_skip + residual_scale * W @ x"""
  dim_out, dim_in = W.shape[0], W.shape[1]
  return torch.addmm(
    x_skip.view(-1, dim_out),
    x.view(-1, dim_in),
    W.T,
    alpha=residual_scale).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
  """
  Embeds scalar timesteps into vector representations.
  """
  def __init__(self, hidden_size, frequency_embedding_size=256):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(frequency_embedding_size, hidden_size, bias=True),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size, bias=True))
    self.frequency_embedding_size = frequency_embedding_size

  @staticmethod
  def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    freqs = torch.exp(
      - math.log(max_period)
      * torch.arange(start=0, end=half, dtype=torch.float32)
      / half).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat(
        [embedding,
         torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

  def forward(self, t):
    t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    t_emb = self.mlp(t_freq)
    return t_emb


class LabelEmbedder(nn.Module):
  """Embeds class labels into vector representations.
  
  Also handles label dropout for classifier-free guidance.
  """
  def __init__(self, num_classes, cond_size):
    super().__init__()
    self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
    self.num_classes = num_classes

    # TODO think of initializing with 0.02 std deviation like in original DiT paper

  def forward(self, labels):
    embeddings = self.embedding_table(labels)
    return embeddings
    

#################################################################################
#                                 Core Model                                    #
#################################################################################

def regular_attention_multi_headed(qkv):
  # Assuming qkv is a tensor with shape [batch, seq_len, 3, num_heads, head_dim]
  # where the 3 represents Q, K, V packed in that order
  batch_size, seq_len, _, num_heads, head_dim = qkv.shape
  # Separate Q, K, V from the packed qkv tensor
  # [batch_size, seq_len, num_heads, head_dim]
  q = qkv[:, :, 0, :, :]
  k = qkv[:, :, 1, :, :]
  v = qkv[:, :, 2, :, :]  

  # Transpose and reshape Q and K for batched matrix multiplication:
  # [batch_size, num_heads, seq_len, head_dim]
  q = q.transpose(1, 2)
  k = k.transpose(1, 2)
  v = v.transpose(1, 2)

  # Compute scaled dot-product attention
  # [batch_size, num_heads, seq_len, seq_len]
  attention_scores = torch.matmul(
    q, k.transpose(-2, -1)) / math.sqrt(head_dim)

  # Apply softmax to calculate the attention weights
  attention_probs = F.softmax(attention_scores, dim=-1)

  # [batch_size, num_heads, seq_len, head_dim]
  attention_output = torch.matmul(attention_probs, v)

  # [batch_size, seq_len, num_heads, head_dim]
  attention_output = attention_output.transpose(1, 2)
  return einops.rearrange(attention_output,
                          'b s h d -> b s (h d)')


class DDiTBlock(nn.Module):
  def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4,
               dropout=0.1, attn_backend='flash_attn'):
    super().__init__()
    self.n_heads = n_heads
    self.attn_backend = attn_backend

    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout

    self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
    self.adaLN_modulation.weight.data.zero_()
    self.adaLN_modulation.bias.data.zero_()

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference
  

  def get_qkv(self, x, rotary_cos_sin):
    qkv = self.attn_qkv(x)
    qkv = einops.rearrange(
      qkv,
      'b s (three h d) -> b s three h d',
      three=3,
      h=self.n_heads)
    with torch.cuda.amp.autocast(enabled=False):
      cos, sin = rotary_cos_sin
      if self.attn_backend == 'flash_attn':
        qkv = apply_rotary_pos_emb(
          qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
      else:
        qkv = apply_rotary_pos_emb_torchscript(
          qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
    return qkv

  def cross_attn(self, x, rotary_cos_sin, cross_attn_mask=None):
    if cross_attn_mask is not None:
      n = x.shape[1] // 2
      qkv_x = self.get_qkv(x[:,:n], rotary_cos_sin)
      qkv_x0 = self.get_qkv(x[:,n:], rotary_cos_sin)
      qkv = torch.cat((qkv_x, qkv_x0), dim=1)
    else:
      qkv = self.get_qkv(x, rotary_cos_sin)
    scale = qkv.shape[-1]
    qkv = qkv.transpose(1, 3)
    attn_dropout = self.attn_dropout if self.training else 0.0
    cross_attn_mask = cross_attn_mask.bool() if cross_attn_mask is not None else None
    x = F.scaled_dot_product_attention(
      query=qkv[:, :, 0],
      key=qkv[:, :, 1],
      value=qkv[:, :, 2],
      attn_mask=cross_attn_mask,
      dropout_p=attn_dropout,
      is_causal=False,
      scale=1 / math.sqrt(scale))
    x = x.transpose(1, 2)
    x = einops.rearrange(x, 'b s h d -> b s (h d)')
    return x

  def forward(self, x, rotary_cos_sin, c, cross_attn_mask=None):
    bias_dropout_scale_fn = self._get_bias_dropout_scale()

    (shift_msa, scale_msa, gate_msa, shift_mlp,
     scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

    # attention operation
    x_skip = x
    x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

    if cross_attn_mask is None and self.attn_backend == 'flash_attn':
      qkv = self.attn_qkv(x)
      x = regular_attention_multi_headed(qkv)
    else:
      x = self.cross_attn(x, rotary_cos_sin, cross_attn_mask=cross_attn_mask)

    x = bias_dropout_scale_fn(self.attn_out(x),
                              None,
                              gate_msa,
                              x_skip,
                              self.dropout)

    # mlp operation
    x = bias_dropout_scale_fn(
      self.mlp(modulate_fused(
        self.norm2(x), shift_mlp, scale_mlp)),
      None, gate_mlp, x, self.dropout)
    return x


class EmbeddingLayer(nn.Module):
  def __init__(self, dim, vocab_dim):
    super().__init__()
    self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
    torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

  def forward(self, x):
    return self.embedding[x]


class DDitFinalLayer(nn.Module):
  def __init__(self, hidden_size, out_channels, cond_dim):
    super().__init__()
    self.norm_final = LayerNorm(hidden_size)
    self.linear = nn.Linear(hidden_size, out_channels)
    self.linear.weight.data.zero_()
    self.linear.bias.data.zero_()

    self.adaLN_modulation = nn.Linear(cond_dim,
                                      2 * hidden_size,
                                      bias=True)
    self.adaLN_modulation.weight.data.zero_()
    self.adaLN_modulation.bias.data.zero_()


  def forward(self, x, c):
    shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
    x = modulate_fused(self.norm_final(x), shift, scale)
    x = self.linear(x)
    return x


class DITBackbone(nn.Module):
  def __init__(
      self,
      config: BAMDLMConfig):
    super().__init__()

    self.config = config
    self.cross_attn = config.cross_attn
    self.block_size = config.block_size
    self.vocab_size = config.vocab_size

    self.vocab_embed = EmbeddingLayer(
      config.hidden_dim,
      config.vocab_size)
    self.sigma_map = TimestepEmbedder(
      config.cond_dim)
    self.rotary_emb = Rotary(
      config.hidden_dim // config.n_heads)

    blocks = []
    for _ in range(config.n_blocks):
      blocks.append(DDiTBlock(config.hidden_dim,
                              config.n_heads,
                              config.cond_dim,
                              dropout=config.dropout,
                              attn_backend=config.attn_backend,))
    self.blocks = nn.ModuleList(blocks)

    self.output_layer = DDitFinalLayer(
      config.hidden_dim,
      config.vocab_size,
      config.cond_dim)
    if self.cross_attn:
      self.gen_mask(config.model_length, self.block_size)
    self.precision = torch.float32

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return  bias_dropout_add_scale_fused_inference
    
  def gen_mask(self, seqlen, block_size):
    self_attn_mask = block_causal_mask(seqlen, block_size, mode='diag')
    x0_attn_mask = block_causal_mask(seqlen, block_size, mode='full')
    cross_attn_mask = x0_attn_mask.clone()
    cross_attn_mask.masked_fill_(self_attn_mask == 1, 0)

    cross_attn_mask = torch.cat((self_attn_mask, cross_attn_mask), dim=1)
    x0_attn_mask = torch.cat((torch.zeros_like(self_attn_mask), x0_attn_mask), dim=1)
    self.cross_attn_mask = torch.cat((cross_attn_mask, x0_attn_mask), dim=0)

  def forward(self, indices, sigma, disable_cross_attn=False,
              output_hidden_states=False):
    cross_attn = self.cross_attn and not disable_cross_attn
    if not self.config.time_conditioning:
      sigma = torch.zeros_like(sigma)
    all_hidden_states = []
    x = self.vocab_embed(indices)
    if output_hidden_states:
      all_hidden_states.append(x)
    c = F.silu(self.sigma_map(sigma))
    if cross_attn:
      cross_attn_mask = self.cross_attn_mask.to(x.device)
      n = x.shape[1] // 2
      rotary_cos_sin = self.rotary_emb(x[:, :n])
    else:
      cross_attn_mask = None
      rotary_cos_sin = self.rotary_emb(x)

    with torch.cuda.amp.autocast(dtype=self.precision):
      for i in range(len(self.blocks)):
        x = self.blocks[i](x, rotary_cos_sin, c,
                           cross_attn_mask=cross_attn_mask,)
        if output_hidden_states:
          all_hidden_states.append(x)
      logits = self.output_layer(x, c)
    if cross_attn:
      logits = logits[:, :n]
      all_hidden_states = [hidden_states[:, :n] for hidden_states in all_hidden_states]
    return logits, all_hidden_states

class BAMDLM(transformers.PreTrainedModel):
  """HF-compatible model."""
  config_class = BAMDLMConfig
  base_model_prefix = "bamdlm"

  def __init__(
    self,
    config: BAMDLMConfig):
    super().__init__(config)
    self.backbone = DITBackbone(config)
    if config.var_min:
      self.register_buffer(
        'sampling_eps_min',
        torch.tensor(config.sampling_eps_min))
      self.register_buffer(
        'sampling_eps_max',
        torch.tensor(config.sampling_eps_max))

  def forward(
      self,
      input_ids: torch.LongTensor = None,
      timesteps: torch.FloatTensor = None,
      disable_cross_attn: typing.Optional[bool] = None,
      output_hidden_states: typing.Optional[bool] = None,
      return_dict: typing.Optional[bool] = None,
  ) -> typing.Union[
    torch.Tensor, typing.Tuple,
    modeling_outputs.MaskedLMOutput]:
    """HF-compatible forward method."""
    output_hidden_states = (
      output_hidden_states
      if output_hidden_states is not None
      else self.config.output_hidden_states
    )
    return_dict = return_dict \
      if return_dict is not None \
      else self.config.use_return_dict
    
    logits, all_hidden_states = self.backbone(
      indices=input_ids,
      sigma=timesteps,
      disable_cross_attn=disable_cross_attn,
      output_hidden_states=output_hidden_states
    )
    if return_dict:
      return modeling_outputs.MaskedLMOutput(
        logits=logits,
        hidden_states=all_hidden_states if output_hidden_states else None,
        loss=None
      )
    elif output_hidden_states:
      return logits, all_hidden_states
    else:
      return logits