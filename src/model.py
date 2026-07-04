"""Attention encoder-decoder for TSP (Kool et al. 2019, scaled-down)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(x, x, x)
        x = self.norm1(x + a)
        x = self.norm2(x + self.ff(x))
        return x


class TSPEncoder(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 8,
                 n_layers: int = 3, d_ff: int = 512) -> None:
        super().__init__()
        self.embed = nn.Linear(2, d_model)
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )

    def forward(self, cities: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        cities: (B, N, 2)
        returns: node_emb (B, N, d), graph_emb (B, d)
        """
        h = self.embed(cities)
        for layer in self.layers:
            h = layer(h)
        return h, h.mean(dim=1)


class TSPDecoder(nn.Module):
    def __init__(self, d_model: int = 128, tanh_clip: float = 10.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.tanh_clip = tanh_clip
        # context = [graph_emb, first_city_emb, last_city_emb] → 3*d_model → d_model
        self.ctx_proj = nn.Linear(3 * d_model, d_model)
        # key/val projections for cross-attention over nodes
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5

    def forward(
        self,
        node_emb: torch.Tensor,
        graph_emb: torch.Tensor,
        first_emb: torch.Tensor,
        last_emb: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        node_emb:  (B, N, d)
        graph_emb: (B, d)
        first_emb: (B, d)  — embedding of first city chosen so far
        last_emb:  (B, d)  — embedding of most-recently chosen city
        mask:      (B, N) bool — True = visited (must mask out)
        returns: logits (B, N)
        """
        ctx = torch.cat([graph_emb, first_emb, last_emb], dim=-1)  # (B, 3d)
        ctx = self.ctx_proj(ctx).unsqueeze(1)  # (B, 1, d)

        q = self.W_q(ctx)       # (B, 1, d)
        k = self.W_k(node_emb)  # (B, N, d)

        logits = (q * k).sum(dim=-1) * self.scale  # (B, N)
        logits = self.tanh_clip * torch.tanh(logits)

        logits = logits.masked_fill(mask, float("-inf"))
        return logits


class TSPModel(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 8,
                 n_encoder_layers: int = 3, d_ff: int = 512,
                 tanh_clip: float = 10.0) -> None:
        super().__init__()
        self.encoder = TSPEncoder(d_model, n_heads, n_encoder_layers, d_ff)
        self.decoder = TSPDecoder(d_model, tanh_clip)

    def forward(
        self,
        cities: torch.Tensor,
        greedy: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        cities: (B, N, 2)
        greedy: use argmax instead of sampling
        returns: tour (B, N), log_probs (B,)
        """
        B, N, _ = cities.shape
        device = cities.device

        node_emb, graph_emb = self.encoder(cities)

        visited = torch.zeros(B, N, dtype=torch.bool, device=device)
        tour = torch.zeros(B, N, dtype=torch.long, device=device)
        log_probs = torch.zeros(B, device=device)
        batch_idx = torch.arange(B, device=device)

        # first step: no "first city" yet — use graph_emb as placeholder
        first_emb = graph_emb
        last_emb = graph_emb

        for step in range(N):
            logits = self.decoder(node_emb, graph_emb, first_emb, last_emb, visited)
            # log_softmax once; probs derived via exp() rather than a second
            # softmax reduction over the same logits.
            log_p_all = F.log_softmax(logits, dim=-1)
            probs = log_p_all.exp()

            if greedy:
                chosen = probs.argmax(dim=-1)
            else:
                chosen = torch.multinomial(probs, 1).squeeze(1)

            log_p = log_p_all[batch_idx, chosen]
            log_probs += log_p

            tour[:, step] = chosen
            # scatter produces a new tensor; avoids autograd version-mismatch
            visited = visited.scatter(1, chosen.unsqueeze(1), True)

            chosen_emb = node_emb[batch_idx, chosen]
            if step == 0:
                first_emb = chosen_emb
            last_emb = chosen_emb

        return tour, log_probs

    @torch.no_grad()
    def greedy_tour(self, cities: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(cities, greedy=True)


if __name__ == "__main__":
    model = TSPModel()
    cities = torch.rand(4, 20, 2)
    tour, lp = model(cities)
    print(f"tour shape: {tour.shape}, log_probs shape: {lp.shape}")
    print(f"tour[0]: {tour[0].tolist()}")
    print(f"log_probs: {lp}")
    # check tour is a valid permutation
    for i in range(4):
        assert sorted(tour[i].tolist()) == list(range(20)), "invalid tour"
    print("All tours valid permutations.")
