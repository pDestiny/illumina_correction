import torch
import torch.nn as nn

from argparse import ArgumentParser

from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning as pl
from model import IlluminaSequenceCorrector
from dataset import FastqDataModule, FastqPredictDataModule
from voca import VOCA

# TODO: main 함수 정리하기.
def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    ## Transformer is very sensitive to sorting task
    ## Good settings so far
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=200, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    # ------------
    # data
    # ------------
    dm = FastqDataModule.add_argparse_args(parser)
    x = iter(dm.train_dataloader()).next() # <for testing

    ## ------------
    ## model
    ## ------------

    model = IlluminaSequenceCorrector(
                                    input_voca_size=len(VOCA) - 1,
                                    4,
                                    args.num_layers,    # number of layers
                                    args.d_model,       # dim. in attemtion mechanism
                                    args.num_heads,     # number of heads
                                    dm.enc_padding_idx,
                                    dm.dec_padding_idx,
                                    dm.max_enc_seq_length,
                                    dm.max_dec_seq_length,
                                    args.learning_rate
                                    )

    # ------------
    # training
    # ------------
    trainer = pl.Trainer(
                            max_epochs=1,
                            callbacks=[EarlyStopping(monitor='val_loss')],
                            gpus = 1 # if you have gpu -- set number, otherwise zero
                        )
    trainer.fit(model, datamodule=dm)

    # copy
    import shutil
    best_model_fn = trainer.checkpoint_callback.best_model_path
    import os; os.makedirs('./outputs/release/', exist_ok=True)
    shutil.copy(best_model_fn, './outputs/release/release.ckpt')

    # ------------
    # testing
    # ------------
    transformer_model = model.load_from_checkpoint('./outputs/release/release.ckpt')
    model.set_vocabs(dm.input_vocab, dm.output_vocab)
    result = trainer.test(model, test_dataloaders=dm.test_dataloader())



if __name__ == '__main__':
    cli_main()
