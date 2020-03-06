import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class Attn(nn.Module):
    
    def __init__(self, hidden):
        super().__init__()
        self.hidden = hidden
        self.attn = nn.Linear(self.hidden * 2, hidden)
        self.v = nn.Parameter(torch.rand(hidden)) # module parameter
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, enc_out):
        l = enc_out.size(0)
        b = enc_out.size(1)
        
        # reshape B x S x H
        H = hidden.repeat(l, 1, 1).transpose(0, 1)
        enc_out = enc_out.transpose(0, 1) 
        attn_score = self.score(H, enc_out)
        
        return F.softmax(attn_score, dim=1).unsqueeze(1)

    def score(self, hidden, enc_out):
        '''
        concat score function
        score(s_t, h_i) = vT_a tanh(Wa[s_t; h_i])
        '''
        # normalize energy by tanh activation function (0 ~ 1)
        energy = torch.tanh(self.attn(torch.cat([hidden, enc_out], 2))) # B x S x 2H -> B x S x H
        energy = energy.transpose(2, 1) # B x H x S
        v = self.v.repeat(enc_out.data.shape[0], 1).unsqueeze(1) # B x 1 x H
        energy = torch.bmm(v, energy) # B x 1 x S
        return energy.squeeze(1) # B x S


class BahdanauAttnDecoderRNN(nn.Module):
    
    def __init__(self, encoder, hidden=200, trg_dim=15, n_layers=2, dropout=0.1):
        super().__init__()
        self.hidden = hidden * encoder.n_directions
        self.trg_dim = trg_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.attn = Attn(self.hidden)
        self.pre = nn.Sequential(
            nn.Linear(trg_dim + self.hidden, self.hidden),
            nn.BatchNorm1d(self.hidden),
            nn.ReLU(inplace=True)
        )
        self.rnn_type = encoder.rnn_type
        self.rnn = getattr(nn, self.rnn_type)(
            self.hidden, self.hidden, n_layers, dropout=dropout)
        self.post = nn.Linear(self.hidden, trg_dim)

    def forward(self, trg, last_hidden, enc_out):
        trg = trg.view(1, trg.size(0), -1) # 1 x B x dim
        
        # attention
        attn_weights = self.attn(last_hidden[-1], enc_out) # B x 1 x S
        context = attn_weights.bmm(enc_out.transpose(0, 1)) # B x 1 x H(attn_size)
        context = context.transpose(0, 1) # 1 x B x H(attn_size)

        # pre-linear layer
        pre_input = torch.cat((trg, context), 2) # 1 x B x (dim + attn_size)
        pre_out = self.pre(pre_input.squeeze(0))
        pre_out = pre_out.unsqueeze(0)

        # rnn layer
        rnn_out, hidden = self.rnn(pre_out, last_hidden) # out: 1 x B x dim, hid: n_layer x B x H

        # post-linear layer
        post_out = self.post(rnn_out.squeeze(0)) # 1 x B x dim -> B x dim

        return post_out, hidden, attn_weights


class Generator(nn.Module):
    
    def __init__(self, encoder, hidden=200, trg_dim=10, n_layers=2, dropout=0.1, use_residual=True):
        super().__init__()
        self.n_layers = n_layers
        self.use_residual = use_residual
        self.decoder = BahdanauAttnDecoderRNN(encoder, hidden, trg_dim, n_layers, dropout)

    def forward(self, trg, last_hidden, enc_out):
        if self.use_residual:
            output, hid, attn = self.decoder(trg, last_hidden, enc_out)
            output = trg + output # residual connection
        else:
            output, hid, attn = self.decoder(trg, last_hidden, enc_out)
        
        return output, hid, attn