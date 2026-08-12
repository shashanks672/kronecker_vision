# Kronecker Patch Embedding: Extending Kronecker Embedding to Vision using CIFAR-10

**A research prototype proposing a novel content × position composition for Vision Transformer patch embeddings.**

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Motivation](#2-motivation)
3. [Problem Statement](#3-problem-statement)
4. [Background](#4-background)
5. [Kronecker Embedding](#5-kronecker-embedding)
6. [Proposed Kronecker Patch Embedding (KPE)](#6-proposed-kronecker-patch-embedding-kpe)
7. [Mathematical Formulation](#7-mathematical-formulation)
8. [Architecture](#8-architecture)
9. [Implementation Details](#9-implementation-details)
10. [Training Procedure](#10-training-procedure)
11. [Experimental Setup](#11-experimental-setup)
12. [Results](#12-results)
13. [Discussion](#13-discussion)
14. [Limitations](#14-limitations)
15. [Future Work](#15-future-work)
16. [Project Structure](#16-project-structure)
17. [How to Run](#17-how-to-run)
18. [References](#18-references)

---

## 1. Introduction

Vision Transformers (ViTs) have become a dominant architecture for image recognition. Their first and most critical step is the *patch embedding*: an image is divided into non-overlapping patches that are linearly projected into a latent space. This linear projection treats every pixel inside a patch independently of its relative spatial location *within* the patch.

This project investigates whether a more structured, *factorised* embedding that explicitly composes **content** and **position** can improve representation quality. We draw inspiration from the classical *Kronecker Embedding* idea and propose **Kronecker Patch Embedding (KPE)** — a differentiable module that realises the composition

\[
E_{\text{patch}} = \sum_{i} (e_i \otimes p_i)
\]

where \(e_i\) are local feature vectors extracted from the patch and \(p_i\) are learnable position vectors associated with those local features.

We evaluate KPE inside a small Vision Transformer on CIFAR-10 and compare it against a strong CNN baseline and a standard linear-patch ViT under identical training conditions.

> **Important**: This is *not* a re-implementation of an existing paper. It is an original research prototype intended to explore a mathematically coherent extension of Kronecker-style embeddings to the vision domain.

---

## 2. Motivation

- Linear patch embeddings ignore the internal spatial structure of each patch.
- Classical Kronecker embeddings elegantly factorise interactions between two spaces (e.g., content and position).
- Modern vision architectures still largely rely on simple linear or convolutional projections for the first embedding stage.
- A lightweight, fully differentiable Kronecker composition could inject a useful inductive bias without sacrificing the flexibility of Transformers.

---

## 3. Problem Statement

Can we replace the standard linear patch embedding of a Vision Transformer with a Kronecker-inspired composition of local content features and local position embeddings, and obtain competitive or superior performance on a standard image-classification benchmark (CIFAR-10) while remaining computationally practical?

---

## 4. Background

### 4.1 Vision Transformer Patch Embedding

Given an image \(x \in \mathbb{R}^{C \times H \times W}\) and patch size \(P\), the image is reshaped into \(N = (H/P)\times(W/P)\) patches of dimension \(C\cdot P\cdot P\). A linear layer then maps each patch to an embedding of dimension \(D\):

\[
z_i = W \cdot \text{flatten}(x_i) + b, \qquad i=1\dots N
\]

### 4.2 Kronecker Product

For vectors \(a \in \mathbb{R}^{m}\) and \(b \in \mathbb{R}^{n}\) the Kronecker product \(a \otimes b \in \mathbb{R}^{mn}\) is the vectorised outer product:

\[
(a \otimes b)_{(i-1)n+j} = a_i b_j
\]

It provides a natural bilinear interaction between two factors.

---

## 5. Kronecker Embedding

In the classical setting a Kronecker embedding represents an entity as the Kronecker product (or a sum of such products) of lower-dimensional factors. The philosophy is that complex interactions can be recovered from the tensor product of simpler factors. We transplant this idea to the *inside* of a single image patch.

---

## 6. Proposed Kronecker Patch Embedding (KPE)

### Pipeline

```
Patch (4×4×3 = 48 values)
        │
        ▼
  Small MLP (48 → 64 → K·d_f)
        │
        ▼
  K local feature vectors e_i ∈ ℝ^{d_f}
        │
        ▼
  Learnable local position embeddings p_i ∈ ℝ^{d_p}
        │
        ▼
  For each i:  e_i ⊗ p_i   ∈ ℝ^{d_f·d_p}
        │
        ▼
  E_patch = Σ_i (e_i ⊗ p_i)   ∈ ℝ^{128}
```

**Design choices used in this prototype**

| Parameter              | Value |
|------------------------|-------|
| Patch size             | 4×4   |
| Local feature dim \(d_f\) | 8  |
| Position dim \(d_p\)   | 16    |
| Number of local features \(K\) | 4 |
| Resulting embed dim    | 128   |

Absolute (global) patch positions are still added later by the standard ViT positional embedding; KPE only models *relative* structure *inside* each patch.

---

## 7. Mathematical Formulation

Let a flattened patch be \(x \in \mathbb{R}^{48}\).

1. **Local feature extraction**

\[
f = \text{MLP}(x) \in \mathbb{R}^{K\cdot d_f}
\qquad
\{e_i\}_{i=1}^{K} = \text{reshape}(f)
\]

2. **Position factors** – learnable parameters

\[
\{p_i\}_{i=1}^{K} \subset \mathbb{R}^{d_p}
\]

3. **Kronecker composition**

\[
E = \sum_{i=1}^{K} (e_i \otimes p_i) \in \mathbb{R}^{d_f\cdot d_p}
\]

4. **Normalisation**

\[
E \leftarrow \text{LayerNorm}(E)
\]

The resulting sequence of patch embeddings is then concatenated with a class token and fed to a standard Transformer encoder.

---

## 8. Architecture

```
Image (3×32×32)
      │
      ▼
┌─────────────────────────────┐
│  Kronecker Patch Embedding  │  → 64 tokens × 128 dims
└─────────────────────────────┘
      │
      ▼
+ Class Token + Absolute Positional Embedding
      │
      ▼
┌─────────────────────────────┐
│  Transformer Encoder (×6)   │  4 heads, MLP ratio 4
└─────────────────────────────┘
      │
      ▼
[CLS] token → Linear head → 10-class logits
```

A pure CNN baseline and a linear-patch ViT share the same training recipe for fair comparison.

---

## 9. Implementation Details

- **Framework**: PyTorch ≥ 2.0
- **Modular design**: every component (embedding, transformer block, training loop, metrics, visualisation) lives in its own file.
- **Configuration**: all hyper-parameters are centralised in `configs/config.yaml`.
- **Device agnostic**: automatically falls back to CPU when CUDA is unavailable.
- **Mixed precision**: optional AMP for faster GPU training.
- **Reproducibility**: fixed seeds, deterministic algorithms where practical.

---

## 10. Training Procedure

1. **Data**: CIFAR-10 with standard normalisation and light RandomCrop + HorizontalFlip augmentation.
2. **Optimiser**: AdamW (lr = 3e-4, weight decay = 0.05)
3. **Scheduler**: Cosine annealing
4. **Regularisation**: Label smoothing 0.1, Dropout 0.1, DropPath 0.05
5. **Epochs**: 50 (configurable)
6. **Batch size**: 128
7. Best model (highest validation accuracy) is checkpointed.

---

## 11. Experimental Setup

| Component          | Setting                          |
|--------------------|----------------------------------|
| Dataset            | CIFAR-10 (50 k train / 10 k test)|
| Image size         | 32×32                            |
| Patch size         | 4×4 → 64 patches                 |
| Embedding dim      | 128                              |
| Transformer depth  | 6                                |
| Attention heads    | 4                                |
| Hardware           | Single GPU (or CPU fallback)     |

**Models compared**

| Model            | Patch Embedding              |
|------------------|------------------------------|
| Baseline CNN     | Convolutional                |
| Baseline ViT     | Linear                       |
| Kronecker-ViT    | Kronecker Patch Embedding    |

**Metrics reported**

- Train / Val / Test accuracy
- Precision, Recall, F1 (macro)
- Confusion matrix
- Parameter count
- Inference latency
- GPU memory (when available)
- FLOPs (via `thop` when installed)

---

## 12. Results

*(After running the training scripts the tables and figures below will be populated automatically.)*

| Model         | Test Acc | Precision | Recall | F1    | Params  | Latency (ms) |
|---------------|----------|-----------|--------|-------|---------|--------------|
| CNN           | –        | –         | –      | –     | –       | –            |
| ViT           | –        | –         | –      | –     | –       | –            |
| Kronecker-ViT | –        | –         | –      | –     | –       | –            |

Training curves, confusion matrices, t-SNE / PCA of embeddings, attention maps and embedding heatmaps are saved under `figures/`.

---

## 13. Discussion

The Kronecker composition injects an explicit factorisation of content and position inside every patch. Because the position factors are shared across the whole image, the model learns a *relative* geometry of the patch interior that is independent of absolute location. Whether this inductive bias helps or hurts depends on the dataset statistics and on the capacity of the subsequent Transformer layers; the experiments in this repository are designed to quantify that trade-off on CIFAR-10.

---

## 14. Limitations

- Evaluated only on CIFAR-10 (32×32). Scaling behaviour on higher-resolution datasets is unknown.
- The number of local features \(K\) and the split \(d_f / d_p\) were chosen heuristically; a systematic ablation is left for future work.
- FLOPs of the Kronecker path are higher than a pure linear projection (outer products), although the absolute cost remains modest for small patches.
- No ImageNet-scale experiments yet.

---

## 15. Future Work

- Ablation on \(K\), \(d_f\), \(d_p\) and alternative local-feature extractors (tiny conv stem vs MLP).
- Scaling to ImageNet-1k / CIFAR-100.
- Integration with hierarchical / pyramid Transformers.
- Theoretical analysis of the expressivity of the sum-of-Kronecker form.
- Learning the position factors in a data-dependent way (e.g., via a small attention mechanism).

---

## 16. Project Structure

```
kronecker_vision/
├── configs/
│   └── config.yaml
├── datasets/
│   └── cifar.py
├── models/
│   ├── cnn.py
│   ├── vit.py
│   ├── transformer.py
│   ├── patch_embedding.py
│   └── kronecker_embedding.py
├── train/
│   ├── train_cnn.py
│   ├── train_vit.py
│   ├── train_kvit.py
│   └── utils.py
├── evaluate/
│   ├── metrics.py
│   └── benchmark.py
├── visualization/
│   ├── tsne.py
│   ├── pca.py
│   ├── heatmap.py
│   └── attention.py
├── app/
│   └── streamlit_app.py
├── weights/                 # saved checkpoints
├── figures/                 # generated plots (created at runtime)
├── results/                 # benchmark JSON (created at runtime)
├── README.md
└── requirements.txt
```

---

## 17. How to Run

### Installation

```bash
cd kronecker_vision
pip install -r requirements.txt
```

### Train the three models

```bash
# Phase 1 – Baseline CNN
python train/train_cnn.py --config configs/config.yaml

# Phase 2 – Baseline ViT
python train/train_vit.py --config configs/config.yaml

# Phase 4 – Kronecker-ViT
python train/train_kvit.py --config configs/config.yaml
```

All scripts accept overrides, e.g.:

```bash
python train/train_kvit.py --epochs 30 --batch-size 64 --device cpu
```

### Launch the interactive demo

```bash
streamlit run app/streamlit_app.py
```

### Reproduce visualisations

After training, the scripts under `visualization/` can be imported or called from a notebook / custom script to generate t-SNE, PCA, attention maps, etc.

---

## 18. References

1. Dosovitskiy et al., “An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale”, ICLR 2021.
2. Classical Kronecker product embeddings in knowledge-graph and multi-relational learning literature (e.g., RESCAL, ComplEx-style factorisations).
3. Touvron et al., “Training data-efficient image transformers & distillation through attention”, ICML 2021 (DeiT).
4. CIFAR-10 dataset – Krizhevsky, 2009.

---

**Citation**

If you use this prototype in academic work, please cite it as an independent research exploration of Kronecker-style patch embeddings for vision.

```
@misc{kronecker-patch-embedding-2026,
  title  = {Kronecker Patch Embedding: Extending Kronecker Embedding to Vision},
  year   = {2026},
  note   = {Research prototype}
}
```

---

*Built as a clean, modular, publication-oriented research codebase.*
