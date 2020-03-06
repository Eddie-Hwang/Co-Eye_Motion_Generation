import numpy as np

from torch.utils.data import DataLoader
from expression_dataset import EyeExpressionDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import Sampler
from sklearn.model_selection import KFold, RepeatedKFold
from collate import collate_fn


def prepare_kfold_dataloaders(data, opt):
    train_loaders = []
    test_loaders = []
    # KFold cross validation
    kfold = RepeatedKFold(opt.n_splits, int(opt.epoch/opt.n_splits), 45)
    eye_expression_dataset = EyeExpressionDataset(
                                word2idx=data['lang'].word2index,
                                idx2word=data['lang'].index2word,
                                src_insts=data['src_insts'],
                                trg_ints=data['trg_insts'],)
    for train_indicies, test_indices in kfold.split(eye_expression_dataset):
        # get sampler based on generated indicies
        train_sampler = SubsetRandomSampler(train_indicies)
        test_sampler = SubsetRandomSampler(test_indices)
        # get train and test dataloader
        train_loader = DataLoader(
                        dataset=eye_expression_dataset,
                        batch_size=opt.batch_size,
                        num_workers=opt.num_workers,
                        sampler=train_sampler,
                        collate_fn=collate_fn)
        test_loader = DataLoader(
                        dataset=eye_expression_dataset,
                        batch_size=opt.batch_size,
                        num_workers=opt.num_workers,
                        sampler=test_sampler,
                        collate_fn=collate_fn)
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    return train_loaders, test_loaders


def prepare_dataloaders(data, opt):
    validation_split = 0.2 # which is 20% of whole dataset
    random_seed = 45
    # get dataset class
    eye_expression_dataset = EyeExpressionDataset(
                                word2idx=data['lang'].word2index,
                                idx2word=data['lang'].index2word,
                                src_insts=data['src_insts'],
                                trg_ints=data['trg_insts'],)
    dataset_indicies = list(range(eye_expression_dataset.__len__()))
    split_index = int(np.floor(validation_split * eye_expression_dataset.__len__()))
    if opt.is_shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(dataset_indicies)
    # get train and validation dataset indicies
    train_indicies, valid_indicies = dataset_indicies[split_index:], dataset_indicies[:split_index]
    train_sampler = SubsetRandomSampler(train_indicies)
    valid_sampler = SubsetRandomSampler(valid_indicies)
    # get train and valid loader
    train_loader = DataLoader(
                        dataset=eye_expression_dataset,
                        batch_size=opt.batch_size,
                        num_workers=opt.num_workers,
                        sampler=train_sampler,
                        collate_fn=collate_fn)
    valid_loader = DataLoader(
                        dataset=eye_expression_dataset,
                        batch_size=opt.batch_size,
                        num_workers=opt.num_workers,
                        sampler=valid_sampler,
                        collate_fn=collate_fn)

    return train_loader, valid_loader