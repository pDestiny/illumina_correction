"""
    Transformer 101 > Encoder + Decoder
        - this code is for educational purpose.
        - the code is written for easy understanding not for optimized code.
    Author : Sangkeun Jung (hugmanskj@gmai.com)
    All rights reserved. (2021)
"""

# In this code, we will implement
#   - Original Transformer
#       - Encoder (we will re-use encoder implementations)
#       - Decoder (we will implement it from the scratch)
#   - Check carefully, How to implement
#       - Cross-Attention (for giving encoder info. to decoder)
#       - Look-ahead Masking (for ignoring future-information)
#   - Also note that
#       - encoder sequence length might not same as decoder sequence length
#       - carefully, check query length and key length
#
#   - For the test dataset,
#       - We will use number sorting dataset
#       - Generate ordered sequences of numbers removing duplicated numbers


import torch
import torch.nn as nn
import torch.nn.functional as F

import math


def result_collapse(outputs, target):
    if len([x[target] for x in outputs][0].shape) == 0:
        target_value = torch.stack([x[target] for x in outputs])
    else:
        target_value = torch.cat([x[target] for x in outputs])
    return target_value


from commons import Attention, clones
from commons import TransformerEncoder


## ---------------- DECODER ----------------------- ##
class TransformerDecoderLayer(nn.Module):
    # - a single layer for Transformer-Decoder block
    def __init__(self, d_model, num_head, droput, dim_feedforward, eps=1e-12):
        super(TransformerDecoderLayer, self).__init__()
        self.embed_dim = d_model
        self.dropout = droput

        ## self-attention
        self.self_attn = Attention(self.embed_dim, num_head, droput)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)  ## residual + LN

        ## cross-attention over encoder's output
        self.encoder_attn = Attention(self.embed_dim, num_head, droput)
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim, eps=eps)

        ## MLP
        self.act_fc = nn.GELU()
        self.activation_dropout = droput  # same as hidden state dropout

        self.fc1 = nn.Linear(self.embed_dim, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=eps)

    def forward(self, x, enc_output, look_ahead_mask, enc_pad_mask):
        "Follow Figure 1 (right) of the original paper for connections."
        # enc_output : [B, input_seq_len, d_model]
        # x : input
        # look_ahead_mask : for decoder's input
        # enc pad_mask    : for encoder's output

        # 1) self-multihead-attention with add & norm
        residual = x
        x, dec_attn_scores = self.self_attn(query=x, key=x, value=x, mask=look_ahead_mask)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        # 2) cross attention
        residual = x
        x, cross_attn_scores = self.encoder_attn(query=x, key=enc_output, value=enc_output, mask=enc_pad_mask)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        x = self.encoder_attn_layer_norm(x)

        # 3) MLP
        residual = x
        x = self.act_fc(self.fc1(x))
        x = F.dropout(x, self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)

        # out : [batch_size, target_seq_len, d_model]
        # attn_scores_1 : [batch_size, num_head, target_seq_len, target_seq_len] = [B, H, query_len, query_len]
        # attn_scores_2 : [batch_size, num_head, target_seq_len, source_seq_len] = [B, H, key_len, query_len]
        return x, dec_attn_scores, cross_attn_scores


class TransformerDecoder(nn.Module):
    "Decoder Block - a stack of N layers"

    def __init__(self, num_layers, d_model, num_heads, dropout, dim_feedforward=None):
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers

        if dim_feedforward == None: dim_feedforward = 4 * d_model
        a_layer = TransformerDecoderLayer(d_model, num_heads, dropout, dim_feedforward)

        # prepare N sub-blocks
        self.layers = clones(a_layer, self.num_layers)

    def forward(self, x, enc_output, look_ahead_mask=None, enc_pad_mask=None):
        # x : [B, tar_seq_len, d_model]
        # enc_output : [B, src_seq_len, d_model]
        # look_ahead_mask : for decoding (causual mask)
        # enc_pad_mask : for blending encoder's hidden states(key) with decoder's input(query),
        #                need to ignore 'pad' positioned hidden states.

        layers_attn_scores_1 = []
        layers_attn_scores_2 = []

        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x, attn_scores_1, attn_scores_2 = layer(x, enc_output, look_ahead_mask, enc_pad_mask)
            layers_attn_scores_1.append(attn_scores_1)  # for encoder
            layers_attn_scores_2.append(attn_scores_2)  # for decoder

        return x, layers_attn_scores_1, layers_attn_scores_2


## -------------------- TRANSFORMER (Encoder + Decoder) ----------------------- ##
##  Additionally we need to implement
##      - Embedding modules ( we will re-use BertEmbeddings ) with differnt name
##          - wihtout token type embedding
class InputEmbeddings(nn.Module):
    """ this embedding moudles are from huggingface implementation
        but, it is simplified -- removing token type embedding since it is not for BERT
    """

    def __init__(self, vocab_size, hidden_size, pad_token_id, max_length_size, layer_norm_eps, hidden_dropout_prob):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_length_size, hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(max_length_size).expand((1, -1)))
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
            persistent=False,
        )

        # always absolute
        self.position_embedding_type = "absolute"

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class Transformer(nn.Module):
    # in here, embedding and decoder output processing parts are not included
    def __init__(self, num_layers, d_model, num_heads, dropout, dim_feedforward=None):
        super().__init__()

        ## transformer blocks only
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, dropout, dim_feedforward)
        self.decoder = TransformerDecoder(num_layers, d_model, num_heads, dropout, dim_feedforward)

    def create_padding_mask(self, mask):
        # prepare padding mask for attention matrix compatible
        return mask[:, None, None, :]  # [B, 1, 1, seq_len]

    def create_look_ahead_mask(self, seq_len):
        """
        prepare causual mask or look-ahead-mask for the decoding
        In decoder, self-attention should be performed with only visible items
        at each time steps. This mask is for preventing future-items at each self-attention in decoer
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.int), diagonal=1)
        mask = 1 - mask  # reverse (1 for visible, 0 for invisible)
        return mask

    def forward(self, enc_input, dec_input, enc_pad_mask):
        # enc_input : [B, src_len, d_model]
        # dec_input : [B, tar_len, d_model]
        #               - in training, it is a right-shifted decoder output starting with <start>
        #               - in inference, it is a previous decoder output appended data starting with <start>
        #
        # enc_pad_mask :
        #       - padding mask for encoder attention
        #       - padding mask for decoder's 2nd attention (to blend encoder's outputs)

        # --------
        # encoder
        # --------
        enc_pad_mask = self.create_padding_mask(enc_pad_mask)
        enc_output, enc_layer_att_scores = self.encoder(enc_input, enc_pad_mask)

        # --------
        # decoder
        # --------
        # masking for self-attention in decoder (LOOK-AHEAD)
        dec_length = dec_input.size()[1]
        look_ahead_mask = self.create_look_ahead_mask(dec_length).to(dec_input.device)

        # masking for cross-attention in bleding decoder input(query) with encoder output(key, value)
        # since multiple-items are from encoder,
        # the mask should be encoder padding mask
        dec_output, dec_layer_att_scores, dec_layer_cross_att_scores = self.decoder(
            dec_input,
            enc_output,
            look_ahead_mask=look_ahead_mask,
            enc_pad_mask=enc_pad_mask
        )
        return enc_output, dec_output, enc_layer_att_scores, dec_layer_att_scores, dec_layer_cross_att_scores