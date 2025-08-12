"""cluster_experts.py

Cluster the FFN neurons (`fc1` / `fc2`) of every decoder layer in the Phi-1.5
model into a fixed number of **balanced experts** (equal cluster sizes).  The
script re-orders the weights & biases according to the assignments and stores a
new `state_dict` that can be loaded with the regular model class.

Example
-------
    python cluster_experts.py \
        --model microsoft/phi-1_5 \
        --n_experts 4 \
        --save_path phi_1_5_clustered.pth

A balanced K-Means implementation from the `k-means-constrained` package is
used.  Install it with:

    pip install k-means-constrained
"""
from __future__ import annotations

import argparse
import sys

from pathlib import Path
from typing import List

import numpy as np
import torch

try:
    from k_means_constrained import KMeansConstrained  # type: ignore
except ImportError as e:
    sys.stderr.write(
        "[ERROR] Missing dependency 'k-means-constrained'. Install with:\n"
        "    pip install k-means-constrained\n"
    )
    raise

from phi_dense_model import load_pretrained, PhiForCausalLM  # local import

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def balanced_kmeans_rows(mat: torch.Tensor, n_clusters: int, seed: int | None = 42) -> np.ndarray:
    """Cluster *rows* of ``mat`` into equal-sized clusters, returning labels.

    Args
    ----
    mat: ``[N, D]`` tensor (will be moved to CPU & numpy)
    n_clusters: number of clusters (experts)
    seed: random seed forwarded to estimator
    """
    mat_np: np.ndarray = mat.cpu().float().detach().numpy()
    n_samples = mat_np.shape[0]
    if n_samples % n_clusters != 0:
        raise ValueError(
            f"Number of rows ({n_samples}) must be divisible by n_clusters ({n_clusters})."
        )
    size = n_samples // n_clusters
    est = KMeansConstrained(
        n_clusters=n_clusters,
        size_min=size,
        size_max=size,
        init="k-means++",
        n_init=10,
        random_state=seed,
        verbose=0,
    )
    labels: np.ndarray = est.fit_predict(mat_np)  # shape (N,)
    return labels


def permutation_from_labels(labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """Return indices that group rows by cluster id (0..k-1) preserving order."""
    grouped: List[np.ndarray] = [np.where(labels == cid)[0] for cid in range(n_clusters)]
    return np.concatenate(grouped, axis=0)


def apply_permutation_to_layer(layer, perm: torch.Tensor):
    """Re-order weights of a single ``PhiDecoderLayer``'s MLP in-place."""
    fc1, fc2 = layer.mlp.fc1, layer.mlp.fc2
    # fc1: [intermediate, hidden] – permute rows
    fc1.weight.data = fc1.weight.data[perm, :]
    fc1.bias.data = fc1.bias.data[perm]
    # fc2: [hidden, intermediate] – permute **columns**
    fc2.weight.data = fc2.weight.data[:, perm]
    # fc2.bias unaffected

# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Cluster Phi-1.5 FFN neurons into experts")
    p.add_argument("--model", default="microsoft/phi-1_5", help="HF model name or path")
    p.add_argument("--n_experts", type=int, default=4, help="Number of experts / clusters")
    p.add_argument("--device", default="cpu", help="Device to load the model on")
    p.add_argument("--save_path", default="phi_1_5_clustered.pth", help="Where to store the new checkpoint")
    p.add_argument("--seed", type=int, default=42, help="Random seed for clustering")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # 1. Load model
    print("[INFO] Loading model…", flush=True)
    model: PhiForCausalLM = load_pretrained(args.model, device=str(device))
    model.eval()

    # 2. Iterate over decoder layers
    print("[INFO] Clustering FFN neurons layer-by-layer…", flush=True)
    for idx, layer in enumerate(model.model.layers):  # type: ignore[attr-defined]
        fc1_weight = layer.mlp.fc1.weight  # shape [8192, 2048]
        labels = balanced_kmeans_rows(fc1_weight, args.n_experts, seed=args.seed)
        perm_np = permutation_from_labels(labels, args.n_experts)
        perm = torch.from_numpy(perm_np).to(fc1_weight.device)

        # The row-wise fc1.weight arrangement will automatically apply the column wise arrangement for fc2.weight
        apply_permutation_to_layer(layer, perm)
        print(f"  • Layer {idx}: OK", flush=True)

    # 3. Save checkpoint
    save_path = Path(args.save_path)
    torch.save(model.state_dict(), save_path)
    print(f"[INFO] Saved clustered model to {save_path.resolve()}")



if __name__ == "__main__":
    main()
