import torch
import torch.nn as nn
from decoder import InputEmbeddings, Transformer
from voca import VOCA

from torchmetrics import functional as FM

class Transformer_DNA_prediction(pl.LightningModule):
    def __init__(self,
                 num_layers,  # number of layers
                 d_model,  # dim. in attemtion mechanism
                 num_heads,
                 output_dim=4,
                 # optiimzer setting
                 learning_rate=1e-3):

        super(Transformer_DNA_prediction, self).__init__()
        self.save_hyperparameters()

        ## embeddings for encoder and decoder (not shared so far)
        self.emb = InputEmbeddings(
            vocab_size=len(VOCA) - 1,
            hidden_size=self.hparams.d_model,
            layer_norm_eps=1e-12,
            pad_token_id=VOCA.inverse["<PRE>"],
            hidden_dropout_prob=0.1
        )

        ## Transformer Block
        self.transformer = Transformer(
            num_layers=self.hparams.num_layers,
            d_model=self.hparams.d_model,
            num_heads=self.hparams.num_heads,
            dropout=0.1,
            dim_feedforward=4 * self.hparams.d_model
        )

        ## to output class
        self.to_output = nn.Linear(self.hparams.d_model, self.hparams.output_dim)  # D -> a single number

        # loss
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, data):
        enc_input_id, enc_input_mask, dec_input_id, label = data
        # ----------------------- ENCODING -------------------------------#
        # [ Digit Character Embedding ]
        #   for encoder and decoder
        enc_input = self.emb(enc_input_id.long())
        dec_input = self.emb(dec_input_id.long())

        enc_output, dec_output, _, _, _ = self.transformer(enc_input, dec_input, enc_input_mask)

        # to symbol
        step_logits = self.to_output(dec_output)  # [B, tar_seq_len, num_output_vocab]

        return step_logits

    def training_step(self, batch, batch_idx):
        enc_input_ids, enc_input_pad_mask, \
        dec_input_ids, \
        dec_output_ids = batch

        step_logits, _ = self([enc_input_ids, dec_input_ids, enc_input_pad_mask])
        C = step_logits.size()[-1]
        loss = self.criterion(step_logits.view(-1, C), dec_output_ids.view(-1).long())

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        enc_input_ids, enc_input_pad_mask, \
        dec_input_ids, \
        dec_output_ids = batch

        step_logits, _ = self(enc_input_ids, dec_input_ids, enc_input_pad_mask)
        C = step_logits.size()[-1]
        loss = self.criterion(step_logits.view(-1, C), dec_output_ids.view(-1).long())

        ## get predicted result
        metrics = {'val_loss': loss}
        self.log_dict(metrics)
        return metrics

    def validation_step_end(self, val_step_outputs):
        val_loss = val_step_outputs['val_loss'].cpu()
        self.log('validation_loss', val_loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        enc_input_ids, enc_input_pad_mask, \
        dec_input_ids, \
        dec_output_ids, dec_output_pad_mask = batch

        step_logits, _ = self(enc_input_ids, dec_input_ids, enc_input_pad_mask)
        C = step_logits.size()[-1]
        loss = self.criterion(step_logits.view(-1, C), dec_output_ids.view(-1).long())

        step_probs = torch.softmax(step_logits, axis=-1)  # [B, tar_seq_len, num_output_vocab]
        step_best_ids = torch.argmax(step_probs, axis=-1)

        ## prediction
        result = {}
        result['input'] = enc_input_ids.cpu()
        result['predicted'] = step_best_ids.cpu()
        result['reference'] = dec_output_ids.cpu()
        result['step_probs'] = step_probs.cpu()
        return result

    def set_vocabs(self, input_vocab, output_vocab):
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab

        self.r_input_vocab = {v: k for k, v in input_vocab.items()}
        self.r_output_vocab = {v: k for k, v in output_vocab.items()}

    def test_epoch_end(self, outputs):
        input = result_collapse(outputs, 'input').cpu()
        predicted = result_collapse(outputs, 'predicted').cpu()
        reference = result_collapse(outputs, 'reference').cpu()
        step_probs = result_collapse(outputs, 'step_probs').cpu()

        def get_valid_items(tensor, pad_idx):
            a = tensor.data.cpu().numpy()
            a = a[a != pad_idx]
            return a

        import os
        os.makedirs("./outputs", exist_ok=True)
        with open('./outputs/sorted_result.txt', 'w') as f:
            for _input, _pred, _ref, _prob in zip(input, predicted, reference, step_probs):
                _input = get_valid_items(_input, self.hparams.enc_padding_idx)
                _pred = get_valid_items(_pred, self.hparams.dec_padding_idx)
                _ref = get_valid_items(_ref, self.hparams.dec_padding_idx)

                ## trim _pred with first <end>
                _N = -1
                for idx, _i in enumerate(_pred):
                    if _i == self.output_vocab['<end>']:
                        _N = idx
                        break
                _pred = _pred[:_N]

                input_seq = [self.r_input_vocab[x] for x in _input]
                pred_seq = [self.r_output_vocab[x] for x in _pred]
                ref_seq = [self.r_output_vocab[x] for x in _ref if self.output_vocab['<end>'] != x]

                input_seq = ",".join(input_seq)
                pred_seq = ",".join(pred_seq)
                ref_seq = ",".join(ref_seq)

                flag = 'O' if pred_seq == ref_seq else 'X'
                print(f'[{flag}] {input_seq}', file=f)
                print(f'\t\tREF  : {ref_seq}', file=f)
                print(f'\t\tPRED : {pred_seq}', file=f)
                print(f'-------------------', file=f)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Transformer")
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parent_parser