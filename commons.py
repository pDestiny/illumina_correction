"""
    Attention Related codes and modules

    Author : Sangkeun Jung (hugmanskj@gmai.com)
    All rights reserved. (2021)
"""

# In this code, we will implement
#   - Scaled Dot-Product attention mechanism 
#   - Query Key Value attention 
#   - Multihead attention


import torch
import torch.nn as nn
import torch.nn.functional as F

import math


def scaled_dot_product_attention(q: torch.Tensor,
                                 k: torch.Tensor,
                                 v: torch.Tensor,
                                 mask: torch.Tensor = None,
                                 dropout: float = 0.1,
                                 ) -> torch.Tensor:
    """
        In here, we try to calculate all multi-heads attentions at once. 
        So, we assumed that the first dimension of q, k and v is B*num_heads=...
            q : expect [..., query_seq_len, d_k]
            k : expect [..., key_seq_len,   d_k]
            v : expect [..., key_seq_len,   d_v]
        mask : expect extended shaped [B, num_heads, query_seq_len, key_seq_len] 1.0 for attend elements, 0 for masking elements
        dropout : expect float value. 
    """
    # for scaling
    d_k = k.size()[-1]
    attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # [B, num_heads, query_seq_len, key_seq_len]

    # masking 
    if mask != None:
        inverted_mask = 1.0 - mask
        inverted_mask = inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(attn.dtype).min)
        attn = attn + inverted_mask  # checkout before and after attn[0][0][0], mask[0][0][0]

    # calculate softmax 
    attention_weights = F.softmax(attn, dim=-1)  # over key dimension   # [..., seq_len, d_k]

    # Original Paper didn't mention about dropout on attention weights. 
    # But many working architectures use dropout on attentions 
    # so, in here we will apply dropout on scores
    if type(dropout) == float:
        attention_weights = F.dropout(attention_weights, dropout)
    elif type(dropout) == nn.Dropout:
        attention_weights = dropout(attention_weights)

    # blending
    output = torch.matmul(attention_weights, v)
    return output, attention_weights


class Attention(nn.Module):
    ## this Attention implementation is almost identical to original transformer paper.
    def __init__(self, d_model, num_heads, dropout=0.1, use_bias=True):
        super(Attention, self).__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads

        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads  # ex) d_model = 512, num_head = 8 --> d_k = 64
        self.d_v = d_model // num_heads  # ex) d_model = 512, num_head = 8 --> d_v = 64

        # why * num_head? --> preapre N heads's input
        # d_model = self.d_k * self.num_head
        # 
        # there are variations to use 'biases' in q,k,v, and o 
        # but, in this implementation, we will use bias 
        self.wq = nn.Linear(d_model, d_model, bias=use_bias)
        self.wk = nn.Linear(d_model, d_model, bias=use_bias)
        self.wv = nn.Linear(d_model, d_model, bias=use_bias)

        # dropout
        self.dropout = nn.Dropout(dropout)

        # to make output 
        # we follow original transformer paper : 
        # in the paper, they mentioned WO for projection on concat vector.
        self.wo = nn.Linear(d_model, d_model, bias=use_bias)

    def split_heads(self, x, batch_size):
        # split the projected dimension 
        # [B, seq_len, heads * d_k ] --> [B, heads, seq_len, d_k] 
        x = x.view(batch_size, -1, self.num_heads, self.d_k)  # to be [B, seq_len, heads, d_k]
        x = x.transpose(1, 2).contiguous()  # to be [B, heads, seq_len, d_k]
        return x

    def forward(self, query, key, value, mask=None):
        q = self.wq(query)  # d_k --> d_k*num_head
        k = self.wk(key)  # d_k --> d_k*num_head
        v = self.wv(value)  # d_k --> d_k*num_head

        # shape change to [B, heads, seq_len, d_k]
        _, qS = q.size()[0], q.size()[1]  # qS = query_seq_len
        B, S = k.size()[0], k.size()[1]  # S  = key_seq_len

        q = self.split_heads(q, B)  # [B, num_heads, query_seq_len, d_k]
        k = self.split_heads(k, B)  # [B, num_heads, key_seq_len,   d_k]
        v = self.split_heads(v, B)  # [B, num_heads, key_seq_len,   d_k]

        # scaled dot-product attention
        # scaled_attention  = [..., query_seq_len, d_k]
        # attention_weights = [..., query_seq_len, key_seq_len]
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask, self.dropout)

        # [Concat Process - for merging multiheads] 
        # recover the tensor form
        scaled_attention = scaled_attention.transpose(1, 2)  # to be [B, query_seq_len, num_heads, d_k]

        # concat
        concat_attention = scaled_attention.reshape(B, qS, -1)  # to be [B, query_seq_len, (num_heads*d_k)=d_model]

        # to output
        output = self.wo(concat_attention)

        # output : # [B, query_seq_len, d_model]
        # attention_weights : [B, num_heads, query_seq_len, key_seq_len]
        return output, attention_weights
