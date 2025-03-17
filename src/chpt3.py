# -*- coding: utf-8 -*-
"""chpt3.ipynb


### Chapter 3 Coding Attention Mechanism

#### Self Attention Mechnism
"""

import torch

#### simplified version of attention
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
  attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)

def softmax_naive(x):
  return torch.exp(x)/torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)
print(attn_weights_2_naive)
print(attn_weights_2_naive.sum(dim=0))


attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print(attn_weights_2)

query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
  context_vec_2+=x_i * attn_weights_2[i]

print(context_vec_2)

### matrix multiplication to grab attention scores
attn_scores = inputs @ inputs.T
### softmax to normalize attention weights across columns
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)
### attention weights times back inputs
all_context_vects = attn_weights @ inputs
print(all_context_vects)

"""#### Self Attention Mechanism With Trainable Weights"""

import torch.nn as nn

class SelfAttention_v1(nn.Module):
  def __init__(self, d_in, d_out):
    super().__init__()
    self.W_query = nn.Parameter(torch.rand(d_in, d_out))
    self.W_key = nn.Parameter(torch.rand(d_in, d_out))
    self.W_value = nn.Parameter(torch.rand(d_in, d_out))


  def forward(self,x):
    keys = x@self.W_key # context length * d_out
    queries = x@self.W_query # context length * d_out
    values = x@self.W_value # context length * d_out

    attn_scores = queries @ keys.T # context length * context length
    attn_weights = torch.softmax(
        attn_scores/keys.shape[-1]**0.5, dim=-1
    )
    context_vec = attn_weights @ values # context length * d_out
    return context_vec



class SelfAttention_v2(nn.Module):
  def __init__(self, d_in, d_out, qkv_bias=False):
    super().__init__()
    self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

  def forward(self,x):
    keys = self.W_key(x) # context length * d_out
    queries = self.W_query(x) # context length * d_out
    values = self.W_value(x) # context length * d_out

    attn_scores = queries @ keys.T # context length * context length
    attn_weights = torch.softmax(
        attn_scores/keys.shape[-1]**0.5, dim=-1
    )
    context_vec = attn_weights @ values # context length * d_out
    return context_vec

torch.manual_seed(123)
d_in, d_out = 3, 2
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))



torch.manual_seed(789)
d_in, d_out = 3, 2
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))
#### nn.Linear stores the weight matrix in a transposed form
print(sa_v2.W_query.weight)
print(sa_v2.W_value.weight)
print(sa_v1.W_value)

# sa_v1.W_value = sa_v2.W_value.weight.T

"""#### Causal Attention Mask"""

class CausalAttention(nn.Module):
  def __init__(self,
               d_in,
               d_out,
               context_length,
               dropout,
               qkv_bias=False):
    super().__init__()
    self.d_out = d_out
    self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    self.dropout = nn.Dropout(dropout)
    self.register_buffer(
        'mask',
        torch.triu(torch.ones(context_length, context_length), diagonal=1)
    )


  def forward(self, x):
    b, num_tokens, d_in = x.shape
    keys = self.W_key(x)
    queries = self.W_query(x)
    values = self.W_value(x)

    attn_scores = queries @ keys.transpose(1,2) # b * num_tokens * num_tokens
    ###step1: filling diagnol values with -inf
    attn_scores.masked_fill(
        self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
    )
    ###step2: apply softmax which will ignore -inf
    attn_weights = torch.softmax(
        attn_scores/keys.shape[-1]**0.5, dim=-1
        )
    ###step3: apply dropout
    attn_weights = self.dropout(attn_weights)

    context_vec = attn_weights@values
    return context_vec



batch = torch.stack((inputs, inputs), dim=0)
torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print(context_vecs.shape)

"""### Multi Head Attention"""

class MultiHeadAttentionWrapper(nn.Module):
  def __init__(self,
               d_in,
               d_out,
               context_length,
               dropout,
               num_heads,
               qkv_bias=False):
    super().__init__()
    self.heads = nn.ModuleList(
        [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
        for _ in range(num_heads)]
    )

  def forward(self, x):
    return torch.cat([head(x) for head in self.heads], dim=-1)

torch.manual_seed(123)
context_length = batch.shape[1]
d_in, d_out = 3,2
mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2
)
context_vecs = mha(batch)
print(context_vecs.shape)

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





torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)