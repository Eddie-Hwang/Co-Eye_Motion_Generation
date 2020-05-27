import argparse
import torch

from tqdm import tqdm, tqdm_gui
from constant import *
from Transformer.model import Transformer
from dataloader import prepare_dataloaders


def train_epoch(model, train_data, optim=None, device=None):
    model.train()
    total_loss = 0
    for batch in tqdm(train_data, mininterval=2, desc=' - (Training)', leave=False):
        batch_loss = 0
        for src_seq, _, trg_seq in batch:
            model(src_seq, trg_seq)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-batch_size', type=int, default=1)
    parser.add_argument('-is_shuffle', type=bool, default=True)
    parser.add_argument('-num_workers', type=int, default=0)
    parser.add_argument('-dropout', type=float, default=0.1)
    opt = parser.parse_args()
    print(opt)

    data = torch.load(DATA)
    train_data, valid_data = prepare_dataloaders(data, opt)

    # prepare transformer
    n_src_vocab = data['lang'].n_words

    model = Transformer(n_src_vocab=data['lang'].n_words,
                        src_pad_idx=PAD,
                        d_word_vec=D_WORD_VEC,
                        d_model=D_MODEL,
                        d_inner=D_INNER,
                        n_layers=N_LAYERS,
                        n_head=N_HEADS,
                        d_k=D_K,
                        d_v=D_K,
                        dropout=opt.dropout,
                        src_n_position=POSITION,
                        trg_n_position=POSITION,
                        n_component=data['estimator'].n_components-2)

    train_epoch(model, train_data)


if __name__ == "__main__":
    main()
