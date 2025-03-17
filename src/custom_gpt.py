# -*- coding: utf-8 -*-
"""custom_gpt.ipynb

"""

import torch
import torch.nn as nn
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
}


class LayerNorm(nn.Module):
  def __init__(self, emb_dim, scale=1, shift=0):
    super().__init__()
    self.eps = 1e-5
    self.scale = nn.Parameter(torch.ones(emb_dim)*scale)
    self.shift = nn.Parameter(torch.zeros(emb_dim)+shift)

  def forward(self, x):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True)

    x = (x-mean)/torch.sqrt(var+self.eps)
    x = self.scale*x + self.shift

    return x

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
  def __init__(self, embed_size, multiplier=4):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(embed_size, multiplier*embed_size),
        GELU(),
        nn.Linear(multiplier*embed_size, embed_size)
    )

  def forward(self,x):
    return self.layers(x)


class MultiHeadAttention(nn.Module):
  def __init__(self,
               d_in,
               d_out,
               context_length,
               dropout,
               num_heads,
               qkv_bias=False):
    super().__init__()
    assert (d_out % num_heads == 0, "d_out must be divisible by num_heads")

    self.d_out = d_out
    self.num_heads = num_heads
    self.head_dim = self.d_out // num_heads
    self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.dropout = nn.Dropout(dropout)
    self.out_proj = nn.Linear(d_out, d_out)
    ### set the upper diagnol to be ones
    self.register_buffer(
        'mask',
        torch.triu(torch.ones(context_length, context_length), diagonal=1)
    )


  def forward(self, x):
    b, num_tokens, d_in = x.shape
    keys = self.W_key(x)
    queries = self.W_query(x)
    values = self.W_value(x)

    keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
    values = values.view(b, num_tokens, self.num_heads, self.head_dim)
    queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

    keys = keys.transpose(1,2)
    values = values.transpose(1,2)
    queries = queries.transpose(1,2)

    attn_scores = queries @ keys.transpose(2,3)
    mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

    attn_scores.masked_fill_(mask_bool, -torch.inf)

    attn_weights = torch.softmax(
        attn_scores/keys.shape[-1]**0.5, dim=-1
    )

    attn_weights = self.dropout(attn_weights)

    context_vec = (attn_weights @ values).transpose(1,2)

    context_vec = context_vec.contiguous().view(
        b, num_tokens, self.d_out
    )

    context_vec = self.out_proj(context_vec)
    return context_vec


class TransformerBlock(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.att = MultiHeadAttention(d_in=cfg["emb_dim"],
                         d_out=cfg["emb_dim"],
                         context_length=cfg["context_length"],
                         dropout=cfg["drop_rate"],
                         num_heads=cfg["n_heads"])

    self.ff = FeedForward(cfg["emb_dim"])
    self.norm1 = LayerNorm(cfg["emb_dim"])
    self.norm2 = LayerNorm(cfg["emb_dim"])
    self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

  def forward(self, x):
    shortcut = x
    x = self.norm1(x)
    x = self.att(x)
    x = self.drop_shortcut(x)
    x = x+shortcut # first shortcut connection

    shortcut = x
    x = self.norm2(x)
    x = self.ff(x)
    x = self.drop_shortcut(x)
    x = x+shortcut # second shortcut connection
    return x



class GPTModel(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
    self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
    self.drop_emb = nn.Dropout(cfg["drop_rate"])

    self.trf_blocks = nn.Sequential(
        *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
    self.final_norm = LayerNorm(cfg["emb_dim"])
    self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

  def forward(self, in_idx):
    batch_size, seq_len = in_idx.shape
    tok_embeds = self.tok_emb(in_idx)
    pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

    x = tok_embeds+pos_embeds
    x = self.drop_emb(x)

    x = self.trf_blocks(x)
    x = self.final_norm(x)
    logits = self.out_head(x)

    return logits