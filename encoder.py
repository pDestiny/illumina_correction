from commons import Attention

import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderLayer(nn.Module):
    # - a single layer for Transformer-Encoder block
    # - This Encoder block is almost identical to original transformer block
    # - activation function is changed to RELU
    #       - (note that, recently RELU is frequently replaced as GELU)

    def __init__(self, d_model, num_head, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.dropout = dropout

        # self-attention
        self.self_attn = Attention(d_model, num_head, dropout)

        # MLP
        self.act_fc = nn.GELU() # <- I changed RELU to GELU
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)

        # LN for after attention and final
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.final_layer_norm     = nn.LayerNorm(d_model)

    def forward(self, x):
        # 1) self-multihead-attention with add & norm
        residual = x
        x, attn_scores = self.self_attn(query=x, key=x, value=x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x) # POST Layer Normalization

        # 2) MLP with add & norm
        residual = x
        x = self.act_fc(self.fc1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)     # POST Layer Normalization

        # out : [batch_size, step_size=S, d_model]
        return x, attn_scores