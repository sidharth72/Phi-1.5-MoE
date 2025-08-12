import torch
import torch.nn as nn
import math

class RotaryEmbedding(nn.Module):
    """Simplified and commented implementation of Rotary Positional Embedding."""

    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_len_offset=0):
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        t += seq_len_offset
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_emb = emb.cos().unsqueeze(0).unsqueeze(2)
        sin_emb = emb.sin().unsqueeze(0).unsqueeze(2)

        # Apply rotary embeddings
        x_rot = self._apply_rotary_pos_emb(x, cos_emb, sin_emb)
        return x_rot

    def _apply_rotary_pos_emb(self, x, cos_emb, sin_emb):
        cos_emb = cos_emb[..., : self.dim // 2]
        sin_emb = sin_emb[..., : self.dim // 2]
        x1, x2 = x[..., : self.dim // 2], x[..., self.dim // 2 :]
        x_rotated_part1 = x1 * cos_emb - x2 * sin_emb
        x_rotated_part2 = x1 * sin_emb + x2 * cos_emb
        return torch.cat((x_rotated_part1, x_rotated_part2), dim=-1)

class Embedding(nn.Module):
    """Token embedding with dropout."""

    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)

    def forward(self, input_ids):
        return self.drop(self.wte(input_ids))

class MLP(nn.Module):
    """Simple Multi-Layer Perceptron."""

    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, config.n_inner)
        self.fc2 = nn.Linear(config.n_inner, config.n_embd)
        self.act = nn.GELU(approximate='tanh')

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class MultiHeadAttention(nn.Module):
    """Multi-head attention module."""

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.dense = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(self, x, past_kv=None, use_cache=False):
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        seq_len_offset = past_kv[0].size(2) if past_kv is not None else 0

        q = self.rotary_emb(q, seq_len_offset)
        k = self.rotary_emb(k, seq_len_offset)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        present_kv = (k, v) if use_cache else None

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        causal_mask = torch.zeros(seq_len, seq_len, device=x.device).masked_fill(mask, float('-inf'))
        if seq_len_offset > 0:
            # Adjust mask for cached keys/values
            causal_mask = torch.cat([torch.zeros(seq_len, seq_len_offset, device=x.device), causal_mask], dim=1)

        scores += causal_mask

        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.dense(output)

        return output, present_kv

class Block(nn.Module):
    """Transformer block."""

    def __init__(self, config):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.n_embd)
        self.self_attn = MultiHeadAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, past_kv=None, use_cache=False):
        residual = x
        x_norm = self.input_layernorm(x)
        attn_output, present_kv = self.self_attn(x_norm, past_kv=past_kv, use_cache=use_cache)
        mlp_output = self.mlp(x_norm)
        output = residual + attn_output + mlp_output
        return output, present_kv

class PhiForCausalLM(nn.Module):
    """Main Phi-1.5 model."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = Embedding(config)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.final_layernorm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)

    def forward(self, input_ids, past_key_values=None, use_cache=False):
        hidden_states = self.embedding(input_ids)
        # print("hidden_states.shape: ", hidden_states.shape)

        if past_key_values is None:
            past_key_values = [None] * len(self.blocks)

        presents = []
        for block, past_kv in zip(self.blocks, past_key_values):
            hidden_states, present_kv = block(hidden_states, past_kv=past_kv, use_cache=use_cache)
            if use_cache:
                presents.append(present_kv)

        hidden_states = self.final_layernorm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, presents
