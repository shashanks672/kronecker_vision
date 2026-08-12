"""
Benchmark utilities that compare CNN, ViT and Kronecker-ViT
on the same test set and produce a consolidated report.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import evaluate_model, compute_model_stats, print_metrics


def run_benchmark(
    models: Dict[str, nn.Module],
    test_loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    save_dir: str = "./results",
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate every model in `models` on the test set and collect
    accuracy / precision / recall / F1 / latency / params / FLOPs.

    Parameters
    ----------
    models : dict mapping model_name → model instance (already loaded with weights)
    test_loader : DataLoader for the test split
    device : torch device
    criterion : optional loss function (for reporting test loss)
    save_dir : directory where a JSON summary will be written

    Returns
    -------
    results : nested dict  {model_name: {metric: value, ...}}
    """
    os.makedirs(save_dir, exist_ok=True)
    all_results: Dict[str, Dict[str, Any]] = {}

    for name, model in models.items():
        print(f"\n>>> Benchmarking: {name}")
        model = model.to(device)
        model.eval()

        # Classification metrics
        metrics = evaluate_model(
            model, test_loader, device, criterion=criterion, return_predictions=True
        )

        # Resource / efficiency metrics
        stats = compute_model_stats(model, device=device)

        combined = {**metrics, **stats}
        # Remove large arrays before JSON serialisation
        combined.pop("y_true", None)
        combined.pop("y_pred", None)
        # Convert numpy cm to list
        if "confusion_matrix" in combined:
            combined["confusion_matrix"] = combined["confusion_matrix"].tolist()

        all_results[name] = combined
        print_metrics(combined, title=name)

    # Save JSON summary
    out_path = Path(save_dir) / "benchmark_summary.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nBenchmark summary saved to {out_path}")

    return all_results


def format_comparison_table(results: Dict[str, Dict[str, Any]]) -> str:
    """Return a markdown-style comparison table."""
    headers = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Params",
        "Latency (ms)",
        "GPU Mem (MB)",
    ]
    rows = []
    for name, m in results.items():
        rows.append([
            name,
            f"{m.get('accuracy', 0):.4f}",
            f"{m.get('precision', 0):.4f}",
            f"{m.get('recall', 0):.4f}",
            f"{m.get('f1', 0):.4f}",
            f"{m.get('parameters_trainable', 0):,}",
            f"{m.get('inference_latency_ms', 0):.2f}",
            f"{m.get('gpu_memory_mb', 0) or 0:.1f}",
        ])

    # Simple markdown table
    col_widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    def fmt_row(row):
        return "| " + " | ".join(str(c).ljust(w) for c, w in zip(row, col_widths)) + " |"

    sep = "|-" + "-|-".join("-" * w for w in col_widths) + "-|"
    lines = [fmt_row(headers), sep] + [fmt_row(r) for r in rows]
    return "\n".join(lines)
