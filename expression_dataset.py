import numpy as np
import torch
import pickle

from torch.utils.data import Dataset, DataLoader


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def load_torch(path):
    return torch.load(path)


class EyeExpressionDataset(Dataset):

    def __init__(self, word2idx, idx2word, src_insts, trg_ints):
        self._word2idx = word2idx
        self._idx2word = idx2word
        self._src_insts = src_insts
        self._trg_insts = trg_ints

    @property
    def n_insts(self):
        return len(self._src_insts)

    @property
    def vocab_size(self):
        return len(self._word2idx)

    @property
    def word2idx(self):
        return self._word2idx

    @property
    def idx2word(self):
        return self._idx2word

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        return self._src_insts[idx], self._trg_insts[idx]