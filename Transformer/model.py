from Transformer.layers import EncoderLayer, DecoderLayer
from Transformer.modules import PositionalEncoding

import torch.nn as nn
import torch


def get_pad_mask(seq, pad_idx):
    return (seq == pad_idx).unsqueeze(-2)

def get_subsequent_mask(seq):
    sz_b, len_s, sz_d = seq.size()
    subsequent_mask = torch.triu(torch.ones((1, len_s, len_s), device=seq.device), diagonal=1).bool()
    return subsequent_mask

class Encoder(nn.Module):
    # d_k, d_v = d_model / n_head
    # d_model == d_word_vec
    def __init__(self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner, pad_idx, dropout=0.1, n_position=200):
        super().__init__()
        self.emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.postion_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
                    EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
                    for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seq, src_mask, return_attns=False):
        enc_intput = self.dropout(self.postion_enc(self.emb(src_seq)))
        enc_output = enc_intput
        # to store self attention
        enc_slf_attn_list = []
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []
        enc_output = self.layer_norm(enc_output)
        
        if return_attns:
            return enc_output, enc_slf_attn_list
        else:
            return enc_output


class Decoder(nn.Module):

    def __init__(self, n_component, n_layers, n_head, d_model, d_k, d_v, d_inner, n_position=200, dropout=0.1):
        super().__init__()

        self.postion_enc = PositionalEncoding(n_component, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):
        dec_intput = self.dropout(self.postion_enc(trg_seq))
        dec_output = dec_intput

        dec_slf_attn_list, dec_enc_attn_list = [], []
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                                            dec_output, enc_output, 
                                            slf_attn_mask=trg_mask, 
                                            dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []
        dec_output = self.layer_norm(dec_output)
        
        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output


class Transformer(nn.Module):

    def __init__(self, n_src_vocab, src_pad_idx,
                d_word_vec, d_model, d_inner, n_layers,
                n_head, d_k, d_v, dropout, 
                src_n_position, n_component, trg_n_position):
        super().__init__()
        self.src_pad_idx = src_pad_idx

        self.encoder = Encoder(n_src_vocab=n_src_vocab, d_word_vec=d_word_vec,
                                n_layers=n_layers, n_head=n_head,
                                d_model=d_model, d_k=d_k, d_v=d_v, d_inner=d_inner,
                                pad_idx=src_pad_idx, n_position=src_n_position)

        self.decoder = Decoder(n_component=n_component, n_layers=n_layers,
                            n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v,
                            d_inner=d_inner, n_position=trg_n_position)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

    def forward(self, src_seq, trg_seq):
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_subsequent_mask(trg_seq)

        enc_output, *_ = self.encoder(src_seq, src_mask)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)   

        return dec_output