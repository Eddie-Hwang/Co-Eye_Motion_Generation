import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class EncoderRNN(nn.Module):
    
    def __init__(self, pre_trained_embedding, rnn_type, hidden, bidirectional, n_layers, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.n_directions = 2 if bidirectional else 1

        # get embedding layer - glove
        self.embedding = nn.Embedding.from_pretrained(
                            torch.from_numpy(pre_trained_embedding).float(),
                            freeze=True)
        self.embedding_size = self.embedding.embedding_dim
        # initialize rnn
        self.rnn = getattr(nn, self.rnn_type)(
                        self.embedding_size, hidden, n_layers,
                        dropout=self.dropout,
                        bidirectional=self.bidirectional)

    def forward(self, src, src_len=None, hidden=None):
        embedded = self.embedding(src)
        if embedded.size(1) > 1: # if batch size is bigger than 1, use pack and padded seq
            packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, src_len)
            output, hidden = self.rnn(packed, hidden) # output: B x S x dim, hidden: S x B x H
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(output) # unpacked
        else:
            output, hidden = self.rnn(embedded, hidden)

        if self.bidirectional:
            hidden = self._cat_directions(hidden)
            # output = output[:, :, :self.hidden] + output[:, :, self.hidden:] # sum bidirectional outputs

        return output, hidden

    def _cat_directions(self, hidden):
        """ If the encoder is bidirectional, do the following transformation.
            Ref: https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/DecoderRNN.py#L176
            -----------------------------------------------------------
            In: (num_layers * num_directions, batch_size, hidden_size)
            (ex: num_layers=2, num_directions=2)

            layer 1: forward__hidden(1)
            layer 1: backward_hidden(1)
            layer 2: forward__hidden(2)
            layer 2: backward_hidden(2)

            -----------------------------------------------------------
            Out: (num_layers, batch_size, hidden_size * num_directions)

            layer 1: forward__hidden(1) backward_hidden(1)
            layer 2: forward__hidden(2) backward_hidden(2)
        """
        def _cat(h):
            return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        
        if isinstance(hidden, tuple):
            # LSTM hidden contains a tuple (hidden state, cell state)
            hidden = tuple([_cat(h) for h in hidden])
        else:
            # GRU hidden
            hidden = _cat(hidden)
            
        return hidden