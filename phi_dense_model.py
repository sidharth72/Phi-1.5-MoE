"""Simplified yet *weight-compatible* implementation of the Phi-1.5 architecture.

This module keeps only the essential blocks required for forward inference while
keeping the same parameter names and tensor shapes as the official
`transformers` implementation so that weights can be loaded directly via
`load_state_dict()` without *any* key remapping.

The code purposefully omits rarely-used features such as:
  * gradient checkpointing
  * attention back-ends other than eager matmul
  * key/value grouping (assumes `num_key_value_heads == num_attention_heads`)
  * dropout (all dropout probabilities are 0 in the released checkpoints)

Despite the simplification it numerically matches the reference model.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Helpers – rotary embedding --------------------------------------------------
# ---------------------------------------------------------------------------

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate the last dimension by splitting in half and swapping (+/-).

    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    This line is doing the following:
        - x.shape[-1] is the size of the last dimension
        - x.shape[-1] // 2 is the integer division of the size by 2
        - x[..., : x.shape[-1] // 2] is taking the first half of the last dimension
        - x[..., x.shape[-1] // 2 :] is taking the second half of the last dimension
        - x1, x2 are the two halves of the last dimension, split at the middle
    """
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embedding to *q* and *k* (first rotary dims only).

    Shapes
    ------
    q, k : [B, H, T, D]
    cos, sin : [T, D] or broadcastable to it
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1,1,T,D]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """Pre-computes cos/sin tables (classic RoPE)."""

    def __init__(self, dim: int, base: float = 10_000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.base = base
        self.dim = dim

    def forward(self, seqlen: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [T, dim//2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, dim]
        return emb.cos().to(dtype), emb.sin().to(dtype)


# ---------------------------------------------------------------------------
# MLP ------------------------------------------------------------------------
# ---------------------------------------------------------------------------

class PhiMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.act = nn.GELU(approximate="tanh")  # matches gelu_new closely
                                    # 2048                8192
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,T,D]
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
        

# ---------------------------------------------------------------------------
# Attention ------------------------------------------------------------------
# ---------------------------------------------------------------------------

class PhiAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_ndims = int(self.head_dim * config.partial_rotary_factor)

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

        # Rotary tables (first rotary_ndims dims).
        self.rotary_emb = RotaryEmbedding(self.rotary_ndims, base=config.rope_theta)

        self.register_buffer("_attn_mask", torch.empty(0), persistent=False)  # filled lazily

    # ---------------------------------------------------------------------
    def _get_causal_mask(self, seqlen: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        mask = torch.full((seqlen, seqlen), -torch.inf, device=device, dtype=dtype)
        mask.triu_(1)
        return mask

    # ---------------------------------------------------------------------
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """hidden_states : [B, T, D]"""
        bsz, seqlen, _ = hidden_states.size()

        # Projections ------------------------------------------------------
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape to [B, H, T, D_head]
        q = q.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)

        # Rotary on first rotary_ndims dims -------------------------------
        if self.rotary_ndims > 0:
            cos, sin = self.rotary_emb(seqlen, q.device, q.dtype)  # [T, R]
            # Split into rotary + pass-through parts
            q_rot, q_pass = q[..., : self.rotary_ndims], q[..., self.rotary_ndims :]
            k_rot, k_pass = k[..., : self.rotary_ndims], k[..., self.rotary_ndims :]
            q_rot, k_rot = apply_rotary(q_rot, k_rot, cos, sin)
            q = torch.cat((q_rot, q_pass), dim=-1)
            k = torch.cat((k_rot, k_pass), dim=-1)

        # Scaled dot-product attention ------------------------------------
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # [B,H,T,T]
        attn_scores /= math.sqrt(self.head_dim)
        causal_mask = self._get_causal_mask(seqlen, attn_scores.dtype, attn_scores.device)
        attn_scores = attn_scores + causal_mask  # broadcasting over B,H
        attn = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn, v)  # [B,H,T,D_head]

        # Merge heads ------------------------------------------------------
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.dense(out)


# ---------------------------------------------------------------------------
# Decoder layer --------------------------------------------------------------
# ---------------------------------------------------------------------------

class PhiDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = PhiAttention(config)
        self.mlp = PhiMLP(config)
        # Dropouts are kept but probabilities are 0 in the checkpoint
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # [B,T,D]
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(hidden_states)
        mlp_out = self.mlp(hidden_states)
        hidden_states = residual + self.resid_dropout(attn_out + mlp_out)
        return hidden_states


# ---------------------------------------------------------------------------
# Base model -----------------------------------------------------------------
# ---------------------------------------------------------------------------

class PhiModel(nn.Module):
    """The bare transformer (no LM head)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([PhiDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:  # [B,T]
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.final_layernorm(hidden_states)


# ---------------------------------------------------------------------------
# Causal-LM wrapper -----------------------------------------------------------
# ---------------------------------------------------------------------------

class PhiForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = PhiModel(config)

        # Language model head which projects the hidden states to the vocabulary space
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

        # tie embeddings if requested (weights are distinct in original checkpoint)
        # Tie is the idea of sharing the weights between the input embeddings and the output logits
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    # ---------------------------------------------------------------------
    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:  # logits
        hidden_states = self.model(input_ids)
        return self.lm_head(hidden_states)


# ---------------------------------------------------------------------------
# Convenience ----------------------------------------------------------------
# ---------------------------------------------------------------------------

def load_pretrained(model_name: str = "microsoft/phi-1_5", *, device: Optional[str] = None):
    """Utility to create `PhiForCausalLM` and load HF weights directly."""
    from transformers import AutoConfig, AutoModelForCausalLM  # lazy import

    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = PhiForCausalLM(hf_config)
    sd = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True).state_dict()
    model.load_state_dict(sd, strict=True)
    return model.to(device) if device else model
