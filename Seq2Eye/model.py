import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from Seq2Eye.encoder import EncoderRNN
from Seq2Eye.luong_attn_decoder import Generator


class Seq2Seq(nn.Module):
    
    def __init__(self, rnn_type, pre_trained_embedding, n_pre_motions, 
                hidden, bidirectional, n_layers, trg_dim, use_residual, dropout):
        super().__init__()
        self.rnn_type = rnn_type
        self.n_layers = n_layers
        self.hidden = hidden
        self.pre_trained_embedding = pre_trained_embedding
        self.use_residual = use_residual
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.n_pre_motions = n_pre_motions
        self.trg_dim = trg_dim
        
        # encoder
        self.encoder = EncoderRNN(self.pre_trained_embedding, self.rnn_type, 
                                self.hidden, self.bidirectional, self.n_layers, self.dropout)
        # decoder
        self.decoder = Generator(self.encoder, self.trg_dim, self.dropout, self.use_residual)

    def forward(self, src, src_len, trg):
        # reshape to S x B x dim
        src = src.transpose(0, 1)
        trg = trg.transpose(0, 1)
        
        # run words through the encoder
        enc_out, enc_hid = self.encoder(src, src_len)
        # initialize decoder's hidden state as encoder's last hidden state (2 x b x dim)
        dec_hid = enc_hid
        # set output to be stored
        all_dec_out = torch.zeros(trg.size(0), trg.size(1), trg.size(2)).to(trg.device) # B x S x dim
        
        # set initial motion
        dec_in = torch.zeros(trg.size(1), trg.size(2)).to(trg.device)
        # run through decoder one time step at a time
        # dec_in = trg[0] # set inital motion (B x dim)
        all_dec_out[0] = dec_in
        for step in range(0, trg.size(0)):
            dec_out, dec_hid, _ = self.decoder(dec_in, dec_hid, enc_out)
            all_dec_out[step] = dec_out
            if step < self.n_pre_motions: # use teacher forcing until n-previous motions
                dec_in = trg[step]
            else:
                dec_in = dec_out

        return all_dec_out.transpose(0, 1)