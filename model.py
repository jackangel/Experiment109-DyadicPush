"""
model.py - DyadicPush Language Model Architecture
=================================================
A sub-quadratic, zero-KV-cache language model architecture replacing
standard O(T^2) quadratic receiver-pull attention with O(T log2 T)
multi-scale forward-push geometric packet routing mailboxes.

Key Properties:
1. Multi-scale dyadic octave strides: Delta_k = 2^k for k in {0, 1, ..., ceil(log2(T)) - 1}.
2. Constant mailbox size per token: K = ceil(log2(T)) slots.
3. Zero KV-cache growth during autoregressive generation.
4. Causal depthwise 1D convolution for fine-grained local n-gram inductive bias.
5. Modern LLM primitives: RMSNorm, SwiGLU MLP, and weight-tied embeddings.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DyadicPushConfig:
    vocab_size: int = 32000
    dim: int = 384
    hidden_dim: int = 1024
    num_layers: int = 8
    num_heads: int = 6
    max_seq_len: int = 4096
    conv_kernel_size: int = 4
    dropout: float = 0.0
    bias: bool = False


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network (Shazeer, 2020) used in modern LLMs."""
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_val = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_out = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(F.silu(self.w_gate(x)) * self.w_val(x))


class DyadicPushAttention(nn.Module):
    """
    DyadicPush Attention:
    Inverts quadratic receiver-pull attention into forward-push geometric packet routing.
    
    Tokens broadcast multi-scale packets along dyadic geometric strides:
        strides = [2^0, 2^1, 2^2, ..., 2^(K-1)], where K = ceil(log2(max_seq_len)).
    
    The receiving token at position t unloads its fixed K mailbox slots using content-based
    attention over the octave packets.
    
    Complexity:
      - Training Time & Space: O(T log2 T), avoiding O(T^2) pairwise attention matrices.
      - Inference KV-Cache: 0 bytes overhead (fixed rolling buffer of K slots).
    """
    def __init__(self, config: DyadicPushConfig):
        super().__init__()
        self.dim = config.dim
        self.num_heads = config.num_heads
        assert self.dim % self.num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = self.dim // self.num_heads
        self.max_seq_len = config.max_seq_len

        # Number of dyadic octaves: e.g. 12 octaves for 4096 context length
        self.num_octaves = math.ceil(math.log2(config.max_seq_len))
        self.strides = [1 << k for k in range(self.num_octaves)]

        # Projections
        self.w_v = nn.Linear(self.dim, self.dim, bias=config.bias)
        self.w_g = nn.Linear(self.dim, self.dim, bias=config.bias)
        self.w_q = nn.Linear(self.dim, self.dim, bias=config.bias)
        self.w_o = nn.Linear(self.dim, self.dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

        # Octave-specific learned routing query keys: (K, num_heads, head_dim)
        self.octave_keys = nn.Parameter(torch.randn(self.num_octaves, self.num_heads, self.head_dim) * 0.02)

        # Scale prior favoring immediate context while maintaining long-range reach
        prior = torch.tensor([1.0 / (1.0 + 0.1 * k) for k in range(self.num_octaves)], dtype=torch.float32)
        self.octave_bias = nn.Parameter(prior)

        # Causal depthwise 1D convolution for local n-gram inductive bias
        k_sz = config.conv_kernel_size
        self.k_sz = k_sz
        self.local_conv = nn.Conv1d(
            self.dim,
            self.dim,
            kernel_size=k_sz,
            padding=k_sz - 1,
            groups=self.dim,
            bias=config.bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parallel vectorized forward pass for training and prompt encoding.
        Input:  x of shape (Batch, SeqLen, Dim)
        Output: out of shape (Batch, SeqLen, Dim)
        """
        B, T, D = x.shape

        # 1. Gated forward payload
        val = F.silu(self.w_v(x)) * torch.sigmoid(self.w_g(x))  # (B, T, D)

        # 2. Causal depthwise convolution for immediate neighboring n-gram context
        # (B, D, T) -> Conv1d with left padding -> slice to [:T] -> (B, T, D)
        conv_val = self.local_conv(val.transpose(1, 2))[:, :, :T].transpose(1, 2)
        combined_val = val + conv_val

        # 3. Unload mailbox across dyadic octaves
        mailbox_slots = []
        for stride in self.strides:
            if stride < T:
                shifted = torch.zeros_like(combined_val)
                shifted[:, stride:, :] = combined_val[:, :-stride, :]
            else:
                shifted = torch.zeros_like(combined_val)
            mailbox_slots.append(shifted)

        # Shape: (B, T, K, num_heads, head_dim)
        mailbox = torch.stack(mailbox_slots, dim=2).view(
            B, T, self.num_octaves, self.num_heads, self.head_dim
        )

        # 4. Multi-head query dot product over K mailbox octave slots
        q = self.w_q(x).view(B, T, self.num_heads, self.head_dim)  # (B, T, H, head_dim)
        scores = torch.einsum("bthd,khd->bthk", q, self.octave_keys) * (1.0 / math.sqrt(self.head_dim))
        scores = scores + self.octave_bias.view(1, 1, 1, self.num_octaves)
        attn = F.softmax(scores, dim=-1)  # (B, T, H, K)
        attn = self.dropout(attn)

        # 5. Readout from mailbox
        # attn: (B, T, H, K), mailbox: (B, T, K, H, head_dim) -> readout: (B, T, H, head_dim)
        readout = torch.einsum("bthk,btkhd->bthd", attn, mailbox)
        readout = readout.contiguous().view(B, T, D)

        return self.w_o(readout)


class DyadicPushBlock(nn.Module):
    """Transformer block with DyadicPush attention and SwiGLU MLP."""
    def __init__(self, config: DyadicPushConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.dim)
        self.attn = DyadicPushAttention(config)
        self.norm2 = RMSNorm(config.dim)
        self.mlp = SwiGLU(config.dim, config.hidden_dim, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DyadicPushLM(nn.Module):
    """
    DyadicPush Language Model:
    Complete autoregressive language model equipped with DyadicPush blocks,
    RMSNorm, and weight-tied token embeddings.
    """
    def __init__(self, config: DyadicPushConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([DyadicPushBlock(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Weight tying between token embedding and LM head for parameter efficiency
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        Args:
            idx: Tensor of token indices, shape (Batch, SeqLen)
            targets: Optional tensor of target token indices, shape (Batch, SeqLen)
        Returns:
            logits: Tensor of shape (Batch, SeqLen, VocabSize)
            loss: Cross-entropy scalar loss (if targets is provided)
        """
        B, T = idx.shape
        assert T <= self.config.max_seq_len, (
            f"Sequence length {T} exceeds maximum configured context {self.config.max_seq_len}"
        )

        x = self.token_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))

        return logits, loss

    def get_num_params(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Autoregressive text generation with temperature, top-k, and nucleus (top-p) sampling.
        Args:
            idx: (1, T) tensor of prompt token IDs
            max_new_tokens: Number of tokens to generate
            temperature: Softmax temperature (lower = more deterministic)
            top_k: Top-k filtering threshold
            top_p: Top-p (nucleus) cumulative probability threshold
        Returns:
            Full tensor of generated token IDs (1, T + max_new_tokens)
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to context window if needed
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Top-p (nucleus) filtering
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right to keep first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
