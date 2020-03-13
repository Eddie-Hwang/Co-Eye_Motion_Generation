import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.attn_dropout = attn_dropout

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultihHeadAttention(nn.Module):

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        
        self.w_qs = nn.Linear(d_model, d_k * n_head, bias=False)
        self.w_ks = nn.Linear(d_model, d_k * n_head, bias=False)
        self.w_vs = nn.Linear(d_model, d_v * n_head, bias=False)
        self.fc = nn.Linear(d_v * n_head, d_model, bias=False)
        self.sdp_attention = ScaledDotProductAttention(d_k ** 0.5)

        # additional layer to improve performance
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        residual = q

        # b x lq x n x dv
        q = self.w_qs(q).view(batch_size, len_q, n_head, d_k)
        k = self.w_ks(k).view(batch_size, len_k, n_head, d_k)
        v = self.w_vs(v).view(batch_size, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        output, attn = self.sdps_attention(q, k, v, mask=mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual

        return output, attn

    
class PositionwiseFeedForward(nn.Module):
    
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_in, d_hid)
        self.w2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, esp=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.w2(F.relu(self.w1(x)))
        x = self.dropout(x)
        x += residual

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super().__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    '''Sinusoid position encoding table'''
    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        
        def get_position_angle_vec(position):
            pos_anlge_vec = [position / np.power(10000, 2 * (hid_j // 2 / d_hid)) for hid_j in range(d_hid)]
            return pos_anlge_vec

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i + 1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def foward(self, x):
        return x + self.pos_table[:, x.size(1).clone().detach()]