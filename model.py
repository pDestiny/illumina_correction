import torch
import torch.nn as nn

from voca import VOCA
from argparse import ArgumentParser

import pytorch_lightning as pl

from encoder import TransformerEncoderLayer

from torchmetrics import functional as FM


class IlluminaSequenceCorrector(pl.LightningModule):
    def __init__(self,
                 # network setting
                 input_voca_size,
                 output_vocab_size,
                 d_model,  # dim. in attention mechanism
                 num_heads,  # number of heads
                 num_layers,
                 learning_rate=1e-4):
        super(IlluminaSequenceCorrector, self).__init__()
        self.save_hyperparameters()

        # symbol_number_character to vector_number
        self.input_emb = nn.Embedding(self.hparams.input_vocab_size,
                                      self.hparams.d_model,
                                      padding_idx=self.hparams.padding_idx)

        # Now, we use transformer-encoder for encoding
        #   - multiple items and a query item together
        encoders = {}
        for i in range(self.hparams.num_layers):
            encoders[f"enc-{i + 1}"] =\
                TransformerEncoderLayer(
                    self.hparams.d_model,
                    self.hparams.num_heads,
                    dim_feedforward=self.hparams.d_model * 4,  # by convention
                    dropout=0.1
                )

        self.encoder = nn.Sequential(encoders)

        # [to output]
        self.to_output = nn.Linear(self.hparams.d_model, self.hparams.output_vocab_size)  # D -> a single number

        # loss
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, seq_ids, weight):
        # INPUT EMBEDDING
        # [ Digit Character Embedding ]
        # seq_ids : [B, max_seq_len]
        seq_embs = self.input_emb(seq_ids.long())  # [B, max_seq_len, d_model]

        # ENCODING BY Transformer-Encoder
        # [mask shaping]
        # masking - shape change
        #   mask always applied to the last dimension explicitly.
        #   so, we need to prepare good shape of mask
        #   to prepare [B, dummy_for_heads, dummy_for_query, dim_for_key_dimension]
        mask = weight[:, None, None, :]  # [B, 1, 1, max_seq_len]
        seq_encs, attention_scores = self.encoder(seq_embs, mask)  # [B, max_seq_len, d_model]

        # seq_encs         : [B, max_seq_len, d_model]
        # attention_scores : [B, max_seq_len_query, max_seq_len_key]

        # Output Processing
        # pooling
        blendded_vector = seq_encs[:, 0]  # taking the first(query) - step hidden state

        # To output
        logits = self.to_output(blendded_vector)
        return logits, attention_scores

    def training_step(self, batch, batch_idx):
        seq_ids, weights, y_id = batch
        logits, _ = self(seq_ids, weights)  # [B, output_vocab_size]
        loss = self.criterion(logits, y_id.long())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # all logs are automatically stored for tensorboard
        return loss

    def validation_step(self, batch, batch_idx):
        seq_ids, weights, y_id = batch

        logits, _ = self(seq_ids, weights)  # [B, output_vocab_size]
        loss = self.criterion(logits, y_id.long())

        ## get predicted result
        prob = F.softmax(logits, dim=-1)
        acc = FM.accuracy(prob, y_id)
        metrics = {'val_acc': acc, 'val_loss': loss}
        self.log_dict(metrics)
        return metrics

    def validation_step_end(self, val_step_outputs):
        val_acc = val_step_outputs['val_acc'].cpu()
        val_loss = val_step_outputs['val_loss'].cpu()

        self.log('validation_acc', val_acc, prog_bar=True)
        self.log('validation_loss', val_loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        seq_ids, weights, y_id = batch

        logits, _ = self(seq_ids, weights)  # [B, output_vocab_size]
        loss = self.criterion(logits, y_id.long())

        ## get predicted result
        prob = F.softmax(logits, dim=-1)
        acc = FM.accuracy(prob, y_id)
        metrics = {'test_acc': acc, 'test_loss': loss}
        self.log_dict(metrics, on_epoch=True)
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    @staticmethod
    def add_model_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("IlluminaSequenceCorrector")
        parser.add_argument('--d_model', type=int, default=512)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--num_layers', type=int, default=4)
        parser.add_argument("--learning_rate", type=float, default=1e-04)
        return parent_parser
