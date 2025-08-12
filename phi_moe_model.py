# MOEfy Phi-1.5


from __future__ import annotations

"""phi_moe_model.py

Mixture-of-Experts (MoE) version of *Phi-1.5* built on top of the *clustered*
weights that were produced by ``cluster_experts.py``.

Key design goals:
    • Keep **weight-compatibility** with the original dense checkpoint so we can
      slice the balanced clusters into individual experts without manual
      transposition.
    • Focus on **readability & maintainability** – the implementation relies on
      straightforward tensor ops and only minimal routing tricks (no capacity
      equations or fused CUDA kernels).
    • Use **noisy-top-k routing** during training for exploration & load
      balancing; fall back to deterministic top-k routing at eval time.

High-level structure (mirrors ``phi_dense_model``):
    ΦMoeExpert   – per-expert MLP (fc1/gelu/fc2)
    NoisyTopKRouter – computes gates + top-k indices
    MixtureOfExperts – wraps router + list[ΦMoeExpert]
    PhiMoeDecoderLayer – Φ attention + MoE block (residual-drop identical)
    PhiMoeModel  – stack of decoder layers
    PhiForCausalLM_MoE – LM head wrapper that ties embeddings if requested

There is also a ``load_pretrained_moe`` helper that
    1. Loads the *cluster-reordered* checkpoint (.pth) produced by
       ``cluster_experts.py``
    2. Builds the MoE model with ``n_experts`` experts per layer & given ``top_k``
       router selection.

This file purposefully does **not** implement gradient checkpointing or expert
capacity constraints – they can be added later without touching the public
interfaces defined here.
"""

import math
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from phi_dense_model import (
    PhiAttention,  # reuse attention block verbatim
    apply_rotary,  # needed for rotary initialisation downstream
)

# ---------------------------------------------------------------------------
# Expert – classic FFN (fc1/gelu/fc2) but with reduced intermediate size -----
# ---------------------------------------------------------------------------

class PhiMoeExpert(nn.Module):
    """Single expert MLP used inside the MoE block."""

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, intermediate_dim, bias=True)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(intermediate_dim, hidden_dim, bias=True)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [*, D]
        x = self.act(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Router – linear projection + noise + top-k selection -----------------------
# ---------------------------------------------------------------------------

class NoisyTopKRouter(nn.Module):
    """Implements G(x) as a linear layer with learnable noise scale.

    During *training* we add 𝒩(0, σ²) noise (σ = softplus(noise_std)) to the
    logits **before** the top-k selection (like Mixtral).  At *eval* noise is
    skipped for deterministic routing.
    """

    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        assert 1 <= top_k <= num_experts, "top_k must be in [1, num_experts]"
        self.top_k = top_k
        self.num_experts = num_experts
        self.linear = nn.Linear(hidden_dim, num_experts, bias=False)
        # σ is parameterised via softplus to keep it positive while allowing 0
        self._noise_std = nn.Parameter(torch.tensor(0.0))  # log-std ~ 0 → σ≈0.69

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``top_k_indices`` and ``gating_weights``.

        Shapes
        ------
        x                : [N, D]            (flattened batch of tokens)
        top_k_indices    : [N, top_k]        long
        gating_weights   : [N, top_k]        float   (sum along last dim = 1)
        """
        logits = self.linear(x)  # [N, E]

        if self.training:
            # Add noise ~𝒩(0, σ²) element-wise
            std = F.softplus(self._noise_std)
            logits = logits + torch.randn_like(logits) * std

        # Select top-k per token ------------------------------------------------
        topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)  # both [N,k]

        # Mask out non-top-k by subtracting large constant before softmax ------
        mask = torch.full_like(logits, float("-inf"))
        mask.scatter_(dim=-1, index=topk_idx, src=topk_vals)
        gates = F.softmax(mask, dim=-1)  # [N, E] but zeros outside top-k
        # Re-gather only the non-zero gates for return convenience
        gathered_gates = torch.gather(gates, dim=-1, index=topk_idx)  # [N,k]
        return topk_idx, gathered_gates


# ---------------------------------------------------------------------------
# Mixture-of-Experts block ----------------------------------------------------
# ---------------------------------------------------------------------------

class MixtureOfExperts(nn.Module):
    """Combines a router with *num_experts* independent MLP experts."""

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        num_experts: int,
        top_k: int = 2,
        *,
        experts: List[PhiMoeExpert] | None = None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = NoisyTopKRouter(hidden_dim, num_experts, top_k)
        # Build experts if not given (fresh random init)
        if experts is None:
            experts = [PhiMoeExpert(hidden_dim, intermediate_dim // num_experts) for _ in range(num_experts)]
        assert len(experts) == num_experts, "Need exactly num_experts MLPs"
        self.experts = nn.ModuleList(experts)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, D]
        bsz, seq_len, hidden = x.shape
        flat_x = x.view(-1, hidden)  # [N, D] where N = B*T

        # 1. Routing ------------------------------------------------------
        topk_idx, topk_gates = self.router(flat_x)  # both [N,k]
        N = flat_x.size(0)

        # 2. Prepare output tensor ---------------------------------------
        combined = torch.zeros_like(flat_x)  # [N, D]

        # 3. Dispatch to each expert (vectorised per expert) -------------
        topk_idx_flat = topk_idx  # alias for clarity [N,k]
        topk_gates_flat = topk_gates  # [N,k]

        for eid, expert in enumerate(self.experts):
            # Boolean mask of tokens that use *this* expert (appears in top-k)
            sel_mask = (topk_idx_flat == eid).any(dim=-1)  # [N]
            if not torch.any(sel_mask):
                continue  # no token routed here

            # 3a. Gather inputs for this expert --------------------------
            tokens = flat_x[sel_mask]            # [M, D]
            # Effective gate weight per token = sum of gates where expert appears
            gate_weights = (topk_gates_flat[sel_mask] * (topk_idx_flat[sel_mask] == eid).float()).sum(dim=-1)  # [M]

            # 3b. Forward through expert ---------------------------------
            expert_out = expert(tokens)          # [M, D]
            expert_out = expert_out * gate_weights.unsqueeze(-1)  # apply gating

            # 3c. Scatter add back to combined output --------------------
            combined[sel_mask] += expert_out

        # 4. Reshape back to [B,T,D] -------------------------------------
        return combined.view(bsz, seq_len, hidden)


# ---------------------------------------------------------------------------
# Decoder layer with MoE -----------------------------------------------------
# ---------------------------------------------------------------------------

class PhiMoeDecoderLayer(nn.Module):
    """Phi decoder layer where the dense FFN is replaced by a MoE block."""

    def __init__(self, config, *, num_experts: int, top_k: int, clustered_layer=None):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = PhiAttention(config)

        # Build experts from clustered weights if provided ----------------
        if clustered_layer is not None:
            experts = _experts_from_clustered_layer(clustered_layer, num_experts)
        else:
            experts = None  # random init

        self.moe = MixtureOfExperts(
            hidden_dim=config.hidden_size,
            intermediate_dim=config.intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            experts=experts,
        )
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    # ------------------------------------------------------------------
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # [B,T,D]
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(hidden_states)
        moe_out = self.moe(hidden_states)
        return residual + self.resid_dropout(attn_out + moe_out)


# ---------------------------------------------------------------------------
# Model wrapper --------------------------------------------------------------
# ---------------------------------------------------------------------------

class PhiMoeModel(nn.Module):
    """Stack of *n_layers* PhiMoeDecoderLayer blocks."""

    def __init__(self, config, *, num_experts: int, top_k: int, clustered_sd: dict | None = None):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # If we have clustered weights we need to pass the per-layer slices ----
        layers: List[nn.Module] = []
        for lid in range(config.num_hidden_layers):
            clustered_layer = None
            if clustered_sd is not None:
                # Extract and POP layer weights to free memory immediately
                prefix = f"model.layers.{lid}.mlp"
                fc1_w = clustered_sd.pop(f"{prefix}.fc1.weight")
                fc1_b = clustered_sd.pop(f"{prefix}.fc1.bias")
                fc2_w = clustered_sd.pop(f"{prefix}.fc2.weight")
                clustered_layer = (fc1_w, fc1_b, fc2_w)
            layer = PhiMoeDecoderLayer(
                config,
                num_experts=num_experts,
                top_k=top_k,
                clustered_layer=clustered_layer,
            )
            layers.append(layer)
            # Free layer tensors after expert creation
            if clustered_layer is not None:
                del clustered_layer
        self.layers = nn.ModuleList(layers)
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # ------------------------------------------------------------------
    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:  # [B,T]
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.final_layernorm(hidden_states)


class PhiForCausalLM_MoE(nn.Module):
    """Causal-LM wrapper identical to dense one but using `PhiMoeModel`."""

    def __init__(self, config, *, num_experts: int, top_k: int, clustered_sd: dict | None = None):
        super().__init__()
        self.model = PhiMoeModel(config, num_experts=num_experts, top_k=top_k, clustered_sd=clustered_sd)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    # ------------------------------------------------------------------
    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:  # logits
        hidden_states = self.model(input_ids)
        return self.lm_head(hidden_states)


# ---------------------------------------------------------------------------
# Utilities ------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _experts_from_clustered_layer(cluster_tuple, n_experts: int) -> List[PhiMoeExpert]:
    """Slice the clustered (fc1/fc2) weights into *n_experts* ΦMoeExpert modules."""
    fc1_w, fc1_b, fc2_w = cluster_tuple  # tensors already on CPU
    inter_dim_total, hidden_dim = fc1_w.shape  # [8192, 2048]
    assert inter_dim_total % n_experts == 0, "intermediate size must divide num_experts"
    per_expert = inter_dim_total // n_experts

    experts: List[PhiMoeExpert] = []
    for eid in range(n_experts):
        # slice(0, 4)
        # slice(4, 8)
        # slice(8, 12)
        # slice(12, 16)
        rows = slice(eid * per_expert, (eid + 1) * per_expert)
        expert = PhiMoeExpert(hidden_dim, per_expert)
        # Copy slices (in-place, no grad)
        # Loading the weights into the expert modules
        with torch.no_grad():
            expert.fc1.weight = nn.Parameter(fc1_w[rows, :].clone())
            expert.fc1.bias = nn.Parameter(fc1_b[rows].clone())
            expert.fc2.weight = nn.Parameter(fc2_w[:, rows].clone())
        experts.append(expert)
    
    # Explicitly delete the large tensors to free memory
    del fc1_w, fc1_b, fc2_w
    return experts



# ---------------------------------------------------------------------------
# Convenience loader ---------------------------------------------------------
# ---------------------------------------------------------------------------

def load_pretrained_moe(
    base_model_name: str = "microsoft/phi-1_5",
    *,
    clustered_ckpt: str | Path = "phi_1_5_clustered.pth",
    num_experts: int = 4,
    top_k: int = 2,
    device: str | torch.device | None = None,
):
    """Factory that returns *PhiForCausalLM_MoE* initialised from a clustered ckpt.

    Args
    ----
    base_model_name : HF repo or local dir containing *dense* Phi weights (for
                      config).  Defaults to official Phi-1.5.
    clustered_ckpt  : Path to the **permuted** state_dict produced by
                      ``cluster_experts.py``
    num_experts     : Number of experts per layer (must match clustering).
    top_k           : Router top-k selection.
    device          : Torch device string / object.
    """
    from transformers import AutoConfig, AutoModelForCausalLM  # local import to avoid heavy dep at top

    cfg = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
    
    from transformers import AutoModelForCausalLM
    with torch.inference_mode():
        # Load dense model with low memory settings
        dense_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            
        )
        embed_w = dense_model.model.embed_tokens.weight.clone()
        lm_head_w = dense_model.lm_head.weight.clone()
        lm_head_b = dense_model.lm_head.bias.clone()

        # Clone per-layer LayerNorm parameters for later transfer
        ln_weights = [layer.input_layernorm.weight.clone() for layer in dense_model.model.layers]
        ln_biases  = [layer.input_layernorm.bias.clone()  for layer in dense_model.model.layers]
        final_ln_w = dense_model.model.final_layernorm.weight.clone()
        final_ln_b = dense_model.model.final_layernorm.bias.clone()

        del dense_model  # Free memory immediately

    # Load clustered weights ---------------------------------------------------
    with torch.no_grad():
        sd = torch.load(clustered_ckpt, map_location="cpu")
        moe_model = PhiForCausalLM_MoE(cfg, num_experts=num_experts, top_k=top_k, clustered_sd=sd)
        # Clear any remaining state_dict entries
        del sd
        # Force garbage collection and clear CUDA cache if available
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    moe_model.half()


    with torch.no_grad():
        moe_model.model.embed_tokens.weight.copy_(embed_w.detach())
        moe_model.lm_head.weight.copy_(lm_head_w.detach())
        moe_model.lm_head.bias.copy_(lm_head_b.detach())

        # Transfer per-layer input_layernorm parameters -------------------
        for lid, layer in enumerate(moe_model.model.layers):
            layer.input_layernorm.weight.copy_(ln_weights[lid].detach())
            layer.input_layernorm.bias.copy_(ln_biases[lid].detach())

        # Transfer final LayerNorm ----------------------------------------
        moe_model.model.final_layernorm.weight.copy_(final_ln_w.detach())
        moe_model.model.final_layernorm.bias.copy_(final_ln_b.detach())
        
    if device is not None:
        moe_model = moe_model.to(device)
    return moe_model


# ---------------------------------------------------------------------------
# TEST-ONLY helper: wrap dense weights into MoE ------------------------------
# ---------------------------------------------------------------------------

def load_dense_as_moe(
    base_model_name: str = "microsoft/phi-1_5",
    *,
    device: str | torch.device | None = None,
) -> "PhiForCausalLM_MoE":
    """Load a *dense* Phi model and copy its weights into a MoE container
    configured with **one** expert and **top_k = 1**.

    This lets us verify that the architectural wiring in ``phi_moe_model.py``
    matches the original dense implementation.  If everything is correct the
    logits of the dense model and the MoE-wrapped version should match to
    numerical precision (~1e-3).

    NOTE: This function is **for debugging only**.  It provides no memory
    savings because the full dense checkpoint is still loaded.
    """
    import gc
    from transformers import AutoModelForCausalLM

    # 1. Load dense reference --------------------------------------------------
    dense = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="cpu",
    ).eval()

    cfg = dense.config

    # 2. Build MoE skeleton with 1 expert + top_k=1 ----------------------------
    moe = PhiForCausalLM_MoE(cfg, num_experts=1, top_k=1, clustered_sd=None)
    moe.half()

    # 3. Global parameter copy -------------------------------------------------
    with torch.no_grad():
        # Embedding + LM head
        moe.model.embed_tokens.weight.copy_(dense.model.embed_tokens.weight)
        moe.lm_head.weight.copy_(dense.lm_head.weight)
        moe.lm_head.bias.copy_(dense.lm_head.bias)

        # Per-layer weights ----------------------------------------------------
        for lid in range(cfg.num_hidden_layers):
            d_layer = dense.model.layers[lid]
            m_layer = moe.model.layers[lid]

            # Attention block + layernorms
            m_layer.self_attn.load_state_dict(d_layer.self_attn.state_dict())
            m_layer.input_layernorm.load_state_dict(d_layer.input_layernorm.state_dict())

            # Final layernorm inside block (if any) is part of Phi; we simply
            # rely on residual path being identical.

            # Copy dense MLP -> sole expert
            expert = m_layer.moe.experts[0]
            expert.fc1.weight.copy_(d_layer.mlp.fc1.weight)
            expert.fc1.bias.copy_(d_layer.mlp.fc1.bias)
            expert.fc2.weight.copy_(d_layer.mlp.fc2.weight)
            expert.fc2.bias.copy_(d_layer.mlp.fc2.bias)

            # Ensure router outputs gate=1 for all tokens ------------------
            m_layer.moe.router.linear.weight.zero_()

        # Top-level final layernorm
        moe.model.final_layernorm.load_state_dict(dense.model.final_layernorm.state_dict())

    # 4. Cleanup ---------------------------------------------------------------
    del dense
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if device is not None:
        moe = moe.to(device)
    return moe
