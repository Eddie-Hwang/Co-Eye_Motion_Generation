import torch.nn as nn
import torch
from Transformer.modules import MultihHeadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.slf_attn = MultihHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffc = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffc(enc_output)

        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.slf_attn = MultihHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultihHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffc = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffc(dec_output)

        return dec_output, dec_slf_attn, dec_enc_attn