import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class LuongAttnDecoderRNN(nn.Module):

    def __init__(self, encoder, trg_dim, dropout=0.1):
        super().__init__()
        self.hidden = encoder.hidden * encoder.n_directions
        self.n_layers = encoder.n_layers
        self.dropout = dropout
        self.trg_dim = trg_dim

        # rnn layer
        self.rnn_type = encoder.rnn_type
        self.rnn = getattr(nn, self.rnn_type)(
                    self.trg_dim, self.hidden, 
                    self.n_layers, dropout=self.dropout)
        self.post = nn.Linear(self.hidden, trg_dim)
        # attention layer
        self.W_a = nn.Linear(encoder.hidden * encoder.n_directions, self.hidden)
        self.W_c = nn.Linear(encoder.hidden * encoder.n_directions + self.hidden,
                            self.hidden)
        # output layer
        self.W_s = nn.Linear(self.hidden, trg_dim)
        
    def forward(self, trg, decoder_hidden, encoder_output):
        trg = trg.view(1, trg.size(0), -1) # 1 x B x dim
        # rnn forward
        decoder_output, decoder_hidden = self.rnn(trg, decoder_hidden)
        decoder_output = decoder_output.transpose(0, 1)

        # attention
        attn_score = torch.bmm(decoder_output, 
                        self.W_a(encoder_output).transpose(0, 1).transpose(1, 2))
        try:
            attn_weights = F.softmax(attn_score.squeeze(1), dim=1).unsqueeze(1)
        except:
            attn_weights = F.softmax(attn_score.squeeze(1)).unsqueeze(1)

        # context vector
        context = torch.bmm(attn_weights, encoder_output.transpose(0, 1))
        # concat
        concat_input = torch.cat([context, decoder_output], -1)
        concat_output = torch.tanh(self.W_c(concat_input))
        attn_weights = attn_weights.squeeze(1)
        # output
        output = self.W_s(concat_output)
        output = output.squeeze(1)

        return output, decoder_hidden, attn_weights


class Generator(nn.Module):
    
    def __init__(self, encoder, trg_dim, dropout=0.1, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.decoder = LuongAttnDecoderRNN(encoder, trg_dim, dropout)

    def forward(self, trg, decoder_hidden, encoder_output):
        if self.use_residual:
            output, decoder_hidden, attn_weights = self.decoder(trg, decoder_hidden, encoder_output)
            output = trg + output
        else:
            output, decoder_hidden, attn_weights = self.decoder(trg, decoder_hidden, encoder_output)

        return output, decoder_hidden, attn_weights 




