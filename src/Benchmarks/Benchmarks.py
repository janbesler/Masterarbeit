# %% [markdown]
# # Libraries

# %%
# Machine Learning
import torch
import torch.nn as nn

# Modules
from Benchmarks.layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from Benchmarks.layers.SelfAttention_Family import ProbAttention, AttentionLayer, FullAttention
from eFormer.embeddings import Encoding


# %% [markdown]
# # Vanilla Transformer

# %%
class VanillaTransformer(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs, seq_len):
        super(VanillaTransformer, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.Embeddings = Encoding(
            in_features=seq_len,
            batch_size=configs.batch_size,
            seq_len=configs.seq_len,
            len_embedding_vector=configs.len_embedding
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def DecoderInput(self, data):
        mid_point = data.size(1) // 2  # Calculate the midpoint
        first_half_last_six = data[:, mid_point-6:mid_point, :]  # Select the last six values from the first half
        second_half_last_six = data[:, -6:, :]  # Select the last six values from the second half
        concatenated_tensor = torch.cat((first_half_last_six, second_half_last_six), dim=1)
        
        return concatenated_tensor

    def forward(self, input_data,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_in = self.Embeddings(input_data)
        enc_out, attns = self.encoder(enc_in, attn_mask=enc_self_mask)

        dec_in = self.DecoderInput(enc_in)
        dec_out = self.decoder(dec_in, enc_in, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, 0], attns
        else:
            return dec_out[:, -self.pred_len:, 0]  # [B, 1]


# %% [markdown]
# # Informer

# %%
class Informer(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
    def __init__(self, configs, seq_len):
        super(Informer, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.Embeddings = Encoding(
            in_features=seq_len,
            batch_size=configs.batch_size,
            seq_len=configs.seq_len,
            len_embedding_vector=configs.len_embedding
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [
                ConvLayer(
                    configs.d_model
                ) for l in range(configs.e_layers - 1)
            ] if configs.distil else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def DecoderInput(self, data):
        mid_point = data.size(1) // 2  # Calculate the midpoint
        first_half_last_six = data[:, mid_point-6:mid_point, :]  # Select the last six values from the first half
        second_half_last_six = data[:, -6:, :]  # Select the last six values from the second half
        concatenated_tensor = torch.cat((first_half_last_six, second_half_last_six), dim=1)
        
        return concatenated_tensor

    def forward(self, input_data,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_in = self.Embeddings(input_data)
        enc_out, attns = self.encoder(enc_in, attn_mask=enc_self_mask)

        dec_in = self.DecoderInput(enc_in)
        dec_out = self.decoder(dec_in, enc_in, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, 0], attns
        else:
            return dec_out[:, -self.pred_len:, 0]  # [B, 1]