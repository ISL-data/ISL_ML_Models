import math
import torch
import torch.nn as nn


# =========================
# Collate functions
# =========================

def collate_fn_packed_lstm(batch):
    """
    For LSTM training using pack_padded_sequence.
    Returns:
      data: (B, T, F)
      labels: (B,)
      lengths: (B,)
    """
    data, labels, lengths = zip(*batch)
    data = torch.stack(data)
    labels = torch.stack(labels).long()
    lengths = torch.stack(lengths).long()
    return data, labels, lengths


def collate_fn_packed(batch):
    """
    For Transformer training.
    Pads to max length in batch and creates padding mask.
    Returns:
      data: (B, Tmax, F)
      labels: (B,)
      lengths: (B,)
      padding_mask: (B, Tmax) True where padded
    """
    data, labels, lengths = zip(*batch)

    lengths = torch.stack(lengths).long()
    labels = torch.stack(labels).long()

    max_len = int(lengths.max().item())
    feat_dim = data[0].shape[-1]

    padded = []
    for seq, L in zip(data, lengths):
        L = int(L.item())
        if seq.shape[0] < max_len:
            pad = torch.zeros(max_len - seq.shape[0], feat_dim, dtype=seq.dtype)
            seq = torch.cat([seq, pad], dim=0)
        else:
            seq = seq[:max_len]
        padded.append(seq)

    data = torch.stack(padded)

    padding_mask = torch.zeros(data.size(0), data.size(1), dtype=torch.bool)
    for i, L in enumerate(lengths):
        L = int(L.item())
        if L < data.size(1):
            padding_mask[i, L:] = True

    return data, labels, lengths, padding_mask


# =========================
# Positional Encoding
# =========================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.05, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: (B, T, D)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# =========================
# Transformer Classifier
# =========================

class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_size,
        num_classes,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.3,
        model_dim=256
    ):
        super().__init__()

        self.embedding = nn.Linear(input_size, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.final_dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(model_dim, num_classes)

    def forward(self, x, lengths=None, padding_mask=None):
        # x: (B, T, F)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        if padding_mask is not None:
            mask = (~padding_mask).unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            x = x.mean(dim=1)

        x = self.final_dropout(x)
        return self.fc(x)


# =========================
# LSTM Classifier
# =========================

class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_classes,
        num_layers=2,
        dropout=0.2,
        bidirectional=True
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        out_dim = hidden_size * (2 if bidirectional else 1)

        self.bn = nn.BatchNorm1d(out_dim)
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x, lengths, padding_mask=None):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        _, (h_n, _) = self.lstm(packed)

        if self.lstm.bidirectional:
            forward_final = h_n[-2]
            backward_final = h_n[-1]
            feat = torch.cat([forward_final, backward_final], dim=1)
        else:
            feat = h_n[-1]

        feat = self.bn(feat)
        feat = self.drop(feat)
        return self.fc(feat)

class ToyTransformerClassifier(nn.Module):
    """
    Small transformer
    """
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        model_dim: int = 96,
        nhead: int = 4,              
        num_layers: int = 1,          
        dim_feedforward: int = 192,   
        dropout: float = 0.1,         
        attn_dropout: float = 0.1,
        max_len: int = 600,
    ):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(input_size, model_dim),
            nn.LayerNorm(model_dim),
        )

        self.pos_encoder = PositionalEncoding(model_dim, dropout=dropout, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",   
            norm_first=True,  
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Dropout(attn_dropout),
            nn.Linear(model_dim, num_classes),
        )

    def forward(self, x, lengths=None, padding_mask=None):
        # x: (B, T, F)
        x = self.embedding(x)          # (B, T, D)
        x = self.pos_encoder(x)        # (B, T, D)
        x = self.encoder(x, src_key_padding_mask=padding_mask)  # (B, T, D)

        # masked mean pooling
        if padding_mask is not None:
            mask = (~padding_mask).unsqueeze(-1).float()  # 1 for valid
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            x = x.mean(dim=1)

        return self.head(x)