import torch
import torch.nn as nn
import joblib
import pandas as pd
import os

import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
import numpy as np
from voca import VOCA

from toolz.curried import *
from sklearn.model_selection import train_test_split
from decoder import InputEmbeddings, Transformer

def create_mask(encoder_seq):
    return np.where(encoder_seq == VOCA.inverse["<MASK>"], 1, 0)

vectorize = compose(np.concatenate,list, map(lambda x: x.reshape(1, -1)))

class FastqDataset(Dataset):
    """Dataset."""

    def __init__(self,
                 enc_input,
                 dec_input,
                 label=None
                 ):

        super(FastqDataset, self).__init__()

        self.enc_input = enc_input
        self.enc_mask = create_mask(self.enc_input)
        self.dec_input = dec_input
        self.label = label
        if self.label is None:
            assert len(self.enc_input) == len(self.dec_input)
        else:
            assert len(self.enc_input) == len(self.dec_input) == len(self.enc_mask) == len(self.label)

    def __len__(self):
        return len(self.enc_input)

    def __getitem__(self, idx):
        enc_input = self.enc_input[idx]
        enc_mask = self.enc_mask[idx]
        dec_input = self.dec_input[idx]
        if self.label is None:
            item = [
                np.array(enc_input),
                np.array(enc_mask),  # encoder padding

                # input - decoder
                np.array(dec_input)
            ]
        else:
            label = self.label[idx]
            item = [
                # input - encoder
                np.array(enc_input),
                np.array(enc_mask),  # encoder padding

                # input - decoder
                np.array(dec_input),

                # label
                np.array(label)
            ]

        return item

class FastqPredictDataModule(pl.LightningDataModule):
    def __init__(self,fn, batch_size=512):
        super(FastqPredictDataModule, self).__init__()

        self.fn = fn
        self.dataset = self.load_data(self.fn)
        self.batch_size = batch_size

        enc_seq = vectorize(self.dataset[["encoder_input"]].values)
        dec_seq = vectorize(self.dataset[["decoder_input"]].values)

        self.predict_dataset = FastqDataset(enc_seq, dec_seq, None)

    def load_data(self, fn):
        fastq_data: pd.DataFrame = joblib.load(os.path.join("data", "test", fn))
        return fastq_data

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size)


class FastqDataModule(pl.LightningDataModule):
    def __init__(self,
                 enc_seq_length: int = 140,
                 dec_seq_length: int = 150,  # for testing
                 batch_size: int = 512,
                 is_exp_train=True,
                 train_ratio=0.8):

        super(FastqDataModule, self).__init__()

        self.batch_size = batch_size
        self.enc_seq_length = enc_seq_length
        self.dec_seq_length = dec_seq_length

        dataset: pd.DataFrame = self.load_data(is_exp_train)

        enc_seq = vectorize(dataset[["encoder_input"]].values)
        dec_seq = vectorize(dataset[["decoder_input"]].values)
        label = vectorize(dataset[["label"]].values) - 2

        # divide into train, validation and train set -> ! train set has seperate process.
        # test set has process for creating fasta for assembly
        np.random.seed(12345677)
        permutation = np.random.permutation(len(enc_seq))
        enc_seq = enc_seq[permutation]
        dec_seq = dec_seq[permutation]
        label_seq = label[permutation]

        X_enc_train, X_enc_val = train_test_split(enc_seq,  train_ratio=train_ratio, shuffle=False)
        X_enc_val, X_enc_test = train_test_split(X_enc_val, train_ratio=0.5, shuffle=False)
        X_dec_train, X_dec_val, y_train, y_val = train_test_split(dec_seq, label_seq, train_ratio=train_ratio, shuffle=False)
        X_dec_val, X_dec_test, y_val, y_test = train_test_split(X_dec_val, y_val, train_ratio=0.5, shuffle=False)


        self.train_dataset = FastqDataset(X_enc_train, X_dec_train, y_train)
        self.valid_dataset = FastqDataset(X_enc_val, X_dec_val, y_val)
        self.test_dataset = FastqDataset(X_enc_test, X_dec_test, y_test)


    def load_data(self, is_exp_train=True):
        if is_exp_train:
            data = joblib.load("data/train/train_exp_set.joblib")
        else:
            data = joblib.load("data/train/train_set.joblib")

        return data


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)  # NOTE : Shuffle

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)








