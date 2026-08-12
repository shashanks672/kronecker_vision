"""
Evaluation metrics and helpers for the Kronecker Vision project.

Provides:
- Accuracy, Precision, Recall, F1 (macro / weighted)
- Confusion matrix
- Classification report
- Simple timing utilities
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    return_predictions: bool = False,
) -> Dict[str, Any]:
    """
    Run a full evaluation pass.

    Returns a dictionary containing:
        loss, accuracy, precision, recall, f1,
        confusion_matrix, y_true, y_pred (optional)
    """
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []
    total_loss = 0.0
    n_batches = 0

    for images, targets in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        # Handle models that return (logits, attn) tuples
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        if criterion is not None:
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            n_batches += 1

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    results = {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
    }
    if return_predictions:
        results["y_true"] = y_true
        results["y_pred"] = y_pred
    return results


def compute_model_stats(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 32, 32),
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Compute parameter count, approximate FLOPs (if thop available),
    and a rough inference latency.
    """
    device = device or next(model.parameters()).device
    model = model.to(device)
    model.eval()

    # Parameter count
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params_total = sum(p.numel() for p in model.parameters())

    # FLOPs via thop (optional dependency)
    flops = None
    try:
        from thop import profile
        dummy = torch.randn(*input_size).to(device)
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        flops = int(flops)
    except Exception:
        pass

    # Inference latency (median of 50 runs after warm-up)
    dummy = torch.randn(*input_size).to(device)
    # Warm-up
    for _ in range(10):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(50):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    latency_ms = float(np.median(times) * 1000)

    # Peak GPU memory (if CUDA)
    gpu_mem_mb = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        _ = model(dummy)
        torch.cuda.synchronize()
        gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return {
        "parameters_trainable": n_params,
        "parameters_total": n_params_total,
        "flops": flops,
        "inference_latency_ms": latency_ms,
        "gpu_memory_mb": gpu_mem_mb,
    }


def print_metrics(metrics: Dict[str, Any], title: str = "Results") -> None:
    """Pretty-print a metrics dictionary."""
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")
    for k, v in metrics.items():
        if k in ("confusion_matrix", "y_true", "y_pred"):
            continue
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f}")
        else:
            print(f"  {k:25s}: {v}")
    print(f"{'=' * 50}\n")
