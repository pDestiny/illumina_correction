import joblib
import pandas as pd
import os
from argparse import ArgumentParser

import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
import numpy as np

from toolz.curried import *
from sklearn.model_selection import train_test_split

vectorize = compose(np.concatenate,list, map(lambda x: x.reshape(1, -1)))

class FastqDataset(Dataset):
    """Dataset."""

    def __init__(self,
                 enc_input,
                 label=None
                 ):

        super(FastqDataset, self).__init__()

        self.enc_input = enc_input
        self.label = label
        if self.label is not None:
            assert len(self.enc_input) == len(self.label)

    def __len__(self):
        return len(self.enc_input)

    def __getitem__(self, idx):
        enc_input = self.enc_input[idx]
        if self.label is None:
            item = [
                np.array(enc_input)
            ]
        else:
            label = self.label[idx]
            item = [
                # input - encoder
                np.array(enc_input),
                # label
                np.array(label)
            ]

        return item

class FastqPredictDataModule(pl.LightningDataModule):
    def __init__(self,
                 fn,
                 batch_size=512
                 ):
        super(FastqPredictDataModule, self).__init__()
        self.save_hyperparameters()

        self.dataset = self.load_data(self.hparams.fn)

        enc_seq = vectorize(self.dataset[["encoder_input"]].values)

        self.predict_dataset = FastqDataset(enc_seq, None)

    def load_data(self, fn):
        fastq_data: pd.DataFrame = joblib.load(os.path.join("data", "test", fn))
        return fastq_data

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.hparams.batch_size)


class FastqDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size=512,
                 is_exp_train=True,
                 train_ratio=0.8):

        super(FastqDataModule, self).__init__()
        self.save_hyperparameters()

        dataset: pd.DataFrame = self.load_data(self.hparams.batch_size)

        enc_seq = vectorize(dataset[["encoder_input"]].values)
        label = vectorize(dataset[["label"]].values)

        # divide into train, validation and train set -> ! train set has seperate process.
        # test set has process for creating fasta for assembly
        np.random.seed(12345677)
        permutation = np.random.permutation(len(enc_seq))
        enc_seq = enc_seq[permutation]
        label_seq = label[permutation]

        x_train, x_val, y_train, y_val = train_test_split(enc_seq, label_seq,  train_size=self.hparams.train_ratio, shuffle=False)
        x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, train_size=0.5, shuffle=False)

        self.train_dataset = FastqDataset(x_train, y_train)
        self.valid_dataset = FastqDataset(x_val, y_val)
        self.test_dataset = FastqDataset(x_test, y_test)

    def load_data(
            self,
            is_exp_train=True # train data's size is 8.3GB. so, small experiment purpose data is needed.
        ):
        if is_exp_train:
            data = joblib.load("data/train/train_exp_set.joblib")
        else:
            data = joblib.load("data/train/train_set.joblib")

        return data

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size)  # NOTE : Shuffle

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size)

    @staticmethod
    def add_datamodule_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("FastqDataModule")
        parser.add_argument('--is_exp_true', default=True, type=bool)
        parser.add_argument('--train_ratio', default=0.8, type=float)
        return parent_parser

if __name__ == "__main__":
    predict_ds_fns = [
        "SRR18888098_1.joblib",
        # "SRR18888098_2.joblib",
        # "SRR19325679_1.joblib",
        # "SRR19325679_2.joblib",
        # "SRR19347957_1.joblib",
        # "SRR19347957_2.joblib",
        # "SRR19347961_1.joblib",
        # "SRR19347961_2.joblib"
    ]
    for fn in predict_ds_fns:
        pdm = FastqPredictDataModule(
            fn=fn, batch_size=512
        )

    train_test_ds = FastqDataModule(
        batch_size=512,
        is_exp_train=True,
        train_ratio=0.8
    )




