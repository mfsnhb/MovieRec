from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SASRecConfig:
    num_items: int
    max_len: int = 50
    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 2
    dropout: float = 0.2


class SASRec(nn.Module):
    def __init__(self, config: SASRecConfig) -> None:
        super().__init__()
        self.config = config
        self.item_embedding = nn.Embedding(config.num_items + 1, config.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(config.max_len, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.register_buffer("causal_mask", torch.triu(torch.ones(config.max_len, config.max_len, dtype=torch.bool), diagonal=1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].fill_(0.0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"SASRec input_ids must be 2D, got shape {tuple(input_ids.shape)}")
        seq_len = input_ids.shape[1]
        if seq_len > self.config.max_len:
            raise ValueError(f"Input sequence length {seq_len} exceeds max_len {self.config.max_len}")
        padding_mask = input_ids.eq(0)
        positions = torch.arange(self.config.max_len - seq_len, self.config.max_len, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        hidden = self.item_embedding(input_ids) + self.position_embedding(positions)
        hidden = self.dropout(hidden)
        hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        hidden = self.encoder(hidden, mask=self.causal_mask[:seq_len, :seq_len])
        hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return self.layer_norm(hidden)

    def last_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        if torch.any(input_ids.ne(0).sum(dim=1) == 0):
            raise ValueError("SASRec received an all-padding sequence")
        hidden = self.forward(input_ids)
        return hidden[:, -1]

    def score_all(self, input_ids: torch.Tensor) -> torch.Tensor:
        query = self.last_hidden(input_ids)
        return query @ self.item_embedding.weight[1:].t()

    def score_items(self, input_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        if item_ids.ndim == 1:
            query = self.last_hidden(input_ids)
            return (query * self.item_embedding(item_ids)).sum(dim=-1)
        hidden = self.forward(input_ids)
        item_vectors = self.item_embedding(item_ids)
        return torch.einsum("bld,bld->bl", hidden, item_vectors)

    def to_config_dict(self) -> dict[str, int | float]:
        return asdict(self.config)
