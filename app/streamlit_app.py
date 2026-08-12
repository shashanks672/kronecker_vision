"""
Kronecker Patch Embedding — Premium Research Demo
A high-end neural / research-lab style interface.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import yaml

# Dark matplotlib style
matplotlib.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#0d1117",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
})

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.cifar import get_class_names
from models.vit import ViTConfig, build_vit, build_kvit
from models.cnn import BaselineCNN, CNNConfig


# ─────────────────────────────────────────────────────────────────────────────
# Page config + Global CSS
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Kronecker Vision Lab",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ─────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background: linear-gradient(160deg, #05070a 0%, #0d1117 40%, #0a0e14 100%);
    color: #c9d1d9;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1400px;}

/* ── Sidebar ──────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #090c10 100%);
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * {color: #c9d1d9 !important;}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stFileUploader label {color: #8b949e !important; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em;}

/* ── Cards ────────────────────────────────────────── */
.kv-card {
    background: linear-gradient(145deg, rgba(22,27,34,0.9) 0%, rgba(13,17,23,0.95) 100%);
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35);
    position: relative;
    overflow: hidden;
}
.kv-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #58a6ff, #a371f7, #58a6ff);
    opacity: 0.7;
}

.kv-card-title {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #8b949e;
    margin-bottom: 0.6rem;
}

.kv-card-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #f0f6fc;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.2;
}

.kv-card-sub {
    font-size: 0.85rem;
    color: #8b949e;
    margin-top: 0.3rem;
}

/* ── Hero ─────────────────────────────────────────── */
.kv-hero {
    background: linear-gradient(135deg, rgba(88,166,255,0.08) 0%, rgba(163,113,247,0.06) 50%, rgba(88,166,255,0.04) 100%);
    border: 1px solid #21262d;
    border-radius: 16px;
    padding: 2rem 2.2rem;
    margin-bottom: 1.8rem;
    position: relative;
}
.kv-hero h1 {
    font-size: 1.9rem;
    font-weight: 700;
    color: #f0f6fc;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.02em;
}
.kv-hero .tagline {
    font-size: 1rem;
    color: #8b949e;
    margin-bottom: 1rem;
    max-width: 720px;
    line-height: 1.55;
}
.kv-badge {
    display: inline-block;
    background: rgba(88,166,255,0.12);
    border: 1px solid rgba(88,166,255,0.3);
    color: #58a6ff;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.25rem 0.7rem;
    border-radius: 20px;
    letter-spacing: 0.04em;
    margin-right: 0.4rem;
}

/* ── Section headers ──────────────────────────────── */
.kv-section {
    font-size: 1.05rem;
    font-weight: 600;
    color: #f0f6fc;
    margin: 1.8rem 0 0.9rem 0;
    padding-bottom: 0.45rem;
    border-bottom: 1px solid #21262d;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.kv-section span {
    font-size: 0.7rem;
    font-weight: 500;
    color: #58a6ff;
    background: rgba(88,166,255,0.1);
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
}

/* ── Explanation boxes ────────────────────────────── */
.kv-explain {
    background: rgba(88,166,255,0.05);
    border-left: 3px solid #58a6ff;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.1rem;
    margin: 0.8rem 0 1.2rem 0;
    font-size: 0.88rem;
    color: #a0aec0;
    line-height: 1.55;
}
.kv-explain strong {color: #c9d1d9;}

.kv-math {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #79c0ff;
    margin: 0.6rem 0;
    overflow-x: auto;
}

/* ── Prediction highlight ─────────────────────────── */
.kv-pred {
    background: linear-gradient(135deg, rgba(63,185,80,0.12) 0%, rgba(46,160,67,0.06) 100%);
    border: 1px solid rgba(63,185,80,0.35);
    border-radius: 12px;
    padding: 1.3rem 1.6rem;
    margin-bottom: 1rem;
}
.kv-pred-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #3fb950;
    font-weight: 600;
}
.kv-pred-class {
    font-size: 2.1rem;
    font-weight: 700;
    color: #f0f6fc;
    margin: 0.2rem 0;
}
.kv-pred-conf {
    font-size: 1rem;
    color: #8b949e;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Metric chips ─────────────────────────────────── */
.kv-chip-row {display: flex; flex-wrap: wrap; gap: 0.6rem; margin: 0.8rem 0;}
.kv-chip {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 0.5rem 0.9rem;
    min-width: 110px;
}
.kv-chip-label {font-size: 0.65rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em;}
.kv-chip-val {font-size: 1.05rem; font-weight: 600; color: #f0f6fc; font-family: 'JetBrains Mono', monospace;}

/* ── Divider ──────────────────────────────────────── */
.kv-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #30363d, transparent);
    margin: 1.5rem 0;
}

/* ── Status pill ──────────────────────────────────── */
.kv-status {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.75rem;
    padding: 0.3rem 0.7rem;
    border-radius: 20px;
    font-weight: 500;
}
.kv-status.ok {background: rgba(63,185,80,0.12); color: #3fb950; border: 1px solid rgba(63,185,80,0.3);}
.kv-status.warn {background: rgba(210,153,34,0.12); color: #d29922; border: 1px solid rgba(210,153,34,0.3);}

/* ── Streamlit overrides ──────────────────────────── */
.stSlider > div > div > div {background: #58a6ff !important;}
div[data-baseweb="select"] > div {background: #161b22 !important; border-color: #30363d !important;}
.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@st.cache_resource
def load_config():
    with open(ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_model(model_name: str, weights_path: str, device_str: str):
    cfg = load_config()
    device = torch.device(device_str)

    if model_name == "CNN":
        model = BaselineCNN(CNNConfig(num_classes=10))
    else:
        vit_cfg = ViTConfig(
            img_size=32,
            patch_size=4,
            embed_dim=cfg["model"]["embed_dim"],
            depth=cfg["model"]["depth"],
            num_heads=cfg["model"]["num_heads"],
            local_feat_dim=cfg["kronecker"]["local_feat_dim"],
            pos_dim=cfg["kronecker"]["pos_dim"],
            num_local_features=cfg["kronecker"]["num_local_features"],
        )
        model = build_vit(vit_cfg) if model_name == "ViT" else build_kvit(vit_cfg)

    weights_exist = Path(weights_path).exists()
    if weights_exist:
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    model = model.to(device)
    model.eval()
    return model, device, weights_exist


def preprocess_image(img: Image.Image, mean, std) -> torch.Tensor:
    img = img.convert("RGB").resize((32, 32), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    for c in range(3):
        tensor[c] = (tensor[c] - mean[c]) / std[c]
    return tensor.unsqueeze(0)


def extract_patches_visual(raw_t: torch.Tensor, patch_size: int = 4):
    x = raw_t
    B, C, H, W = x.shape
    p = patch_size
    patches = x.unfold(2, p, p).unfold(3, p, p)
    patches = patches.contiguous().view(1, C, 64, p, p)
    patches = patches.permute(0, 2, 1, 3, 4).squeeze(0)
    return patches


def entropy(probs: np.ndarray) -> float:
    p = probs[probs > 1e-12]
    return float(-np.sum(p * np.log(p)))


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1.2rem 0;">
        <div style="font-size: 1.3rem; font-weight: 700; color: #f0f6fc; letter-spacing: -0.02em;">◈ Kronecker Lab</div>
        <div style="font-size: 0.75rem; color: #8b949e; margin-top: 0.2rem;">Research Prototype · CIFAR-10</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    model_choice = st.selectbox(
        "MODEL ARCHITECTURE",
        ["Kronecker-ViT", "ViT (baseline)", "CNN (baseline)"],
        help="Kronecker-ViT uses the proposed Kronecker Patch Embedding. The others are baselines.",
    )

    weights_map = {
        "Kronecker-ViT": ROOT / "weights" / "kvit_best.pt",
        "ViT (baseline)": ROOT / "weights" / "vit_best.pt",
        "CNN (baseline)": ROOT / "weights" / "cnn_best.pt",
    }
    model_key = {
        "Kronecker-ViT": "KViT",
        "ViT (baseline)": "ViT",
        "CNN (baseline)": "CNN",
    }[model_choice]

    device_str = get_best_device()
    model, device, weights_loaded = load_model(model_key, str(weights_map[model_choice]), device_str)

    # Status
    if weights_loaded:
        st.markdown('<div class="kv-status ok">● Weights loaded</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="kv-status warn">● Random init (train first)</div>', unsafe_allow_html=True)

    st.markdown(f'<div style="font-size:0.75rem;color:#8b949e;margin-top:0.5rem;">Device: <code>{device}</code></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="font-size:0.7rem;color:#8b949e;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.5rem;">Input Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload PNG / JPG", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

    st.markdown("---")
    with st.expander("About KPE", expanded=False):
        st.markdown("""
**Kronecker Patch Embedding** replaces the standard linear projection of each image patch with a sum of Kronecker products:

$$E = \\sum_i (e_i \\otimes p_i)$$

- $e_i$ — local content features extracted by a small MLP  
- $p_i$ — learnable position vectors inside the patch  

This injects an explicit **content × position** inductive bias before the Transformer even starts.
        """)


# ─────────────────────────────────────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────────────────────────────────────

cfg = load_config()
mean = cfg["dataset"]["mean"]
std = cfg["dataset"]["std"]
class_names = get_class_names()

# Hero
st.markdown("""
<div class="kv-hero">
    <div style="margin-bottom:0.7rem;">
        <span class="kv-badge">RESEARCH PROTOTYPE</span>
        <span class="kv-badge">CIFAR-10</span>
        <span class="kv-badge">PYTORCH</span>
    </div>
    <h1>Kronecker Patch Embedding</h1>
    <div class="tagline">
        A novel patch embedding that composes local content features with position vectors via Kronecker products —
        replacing the classic linear projection inside Vision Transformers.
    </div>
</div>
""", unsafe_allow_html=True)

# Image handling
if uploaded is not None:
    pil_img = Image.open(uploaded)
else:
    st.markdown("""
    <div class="kv-explain">
        <strong>No image uploaded.</strong> A random noise sample is shown below.
        Upload any image — it will be resized to 32×32 and processed by the selected model.
    </div>
    """, unsafe_allow_html=True)
    arr = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    pil_img = Image.fromarray(arr)

# ── Input + Patches ─────────────────────────────────────────────────────────

st.markdown('<div class="kv-section">Input Representation <span>STEP 1</span></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="kv-card-title">Original Image → 32×32</div>', unsafe_allow_html=True)
    st.image(pil_img.resize((256, 256), Image.NEAREST), use_container_width=True)
    st.markdown("""
    <div class="kv-explain">
        The network never sees the full-resolution image. Everything is forced to <strong>32×32</strong>
        (CIFAR-10 native size). This is the only spatial resolution the model understands.
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="kv-card-title">Patch Extraction · 4×4 → 64 tokens</div>', unsafe_allow_html=True)
    raw = np.asarray(pil_img.convert("RGB").resize((32, 32))).astype(np.float32) / 255.0
    raw_t = torch.from_numpy(raw).permute(2, 0, 1).unsqueeze(0)
    patches = extract_patches_visual(raw_t)

    mosaic = np.zeros((32, 32, 3), dtype=np.float32)
    for i in range(8):
        for j in range(8):
            idx = i * 8 + j
            mosaic[i * 4:(i + 1) * 4, j * 4:(j + 1) * 4] = patches[idx].permute(1, 2, 0).numpy()

    st.image(mosaic, use_container_width=True, clamp=True)
    st.markdown("""
    <div class="kv-explain">
        The image is split into a non-overlapping <strong>8×8 grid of 4×4 patches</strong> (64 patches total).
        Each patch is a 48-dimensional vector (4×4×3). These become the input tokens of the Transformer.
    </div>
    """, unsafe_allow_html=True)

# Preprocess for model
img_tensor = preprocess_image(pil_img, mean, std).to(device)

# ── Inference ───────────────────────────────────────────────────────────────

st.markdown('<div class="kv-section">Model Prediction <span>STEP 2</span></div>', unsafe_allow_html=True)

with torch.no_grad():
    if model_key in ("ViT", "KViT"):
        logits, attn_maps = model(img_tensor, return_attention=True)
    else:
        logits = model(img_tensor)
        attn_maps = None

probs = F.softmax(logits, dim=1).cpu().numpy()[0]
pred_idx = int(probs.argmax())
confidence = float(probs[pred_idx])
ent = entropy(probs)
top3_idx = probs.argsort()[::-1][:3]

# Prediction card
st.markdown(f"""
<div class="kv-pred">
    <div class="kv-pred-label">Predicted Class</div>
    <div class="kv-pred-class">{class_names[pred_idx]}</div>
    <div class="kv-pred-conf">{confidence * 100:.2f}% confidence &nbsp;·&nbsp; entropy {ent:.3f}</div>
</div>
""", unsafe_allow_html=True)

# Metric chips — native Streamlit columns (avoids raw HTML rendering issues)
unc_label = "low uncertainty" if ent < 1.0 else "high uncertainty"
chip_cols = st.columns(4)
for i, col in enumerate(chip_cols):
    with col:
        if i < 3:
            idx = top3_idx[i]
            st.markdown(
                f"""
                <div class="kv-chip">
                    <div class="kv-chip-label">Top-{i+1}</div>
                    <div class="kv-chip-val">{class_names[idx]}</div>
                    <div style="font-size:0.75rem;color:#8b949e;">{probs[idx]*100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="kv-chip">
                    <div class="kv-chip-label">Entropy</div>
                    <div class="kv-chip-val">{ent:.3f}</div>
                    <div style="font-size:0.75rem;color:#8b949e;">{unc_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("""
<div class="kv-explain">
    <strong>How to read this:</strong> Confidence is the softmax probability of the top class.
    Entropy measures overall uncertainty — low entropy means the model is peaked on one class;
    high entropy means the probability mass is spread across many classes (the model is unsure).
</div>
""", unsafe_allow_html=True)

# Probability bar chart
fig, ax = plt.subplots(figsize=(9, 3.2))
colors = ["#3fb950" if i == pred_idx else "#388bfd" for i in range(10)]
bars = ax.barh(class_names, probs, color=colors, edgecolor="none", height=0.7)
ax.set_xlim(0, 1)
ax.set_xlabel("Probability", fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#30363d")
ax.spines["bottom"].set_color("#30363d")
ax.tick_params(colors="#8b949e", labelsize=9)
for bar, p in zip(bars, probs):
    if p > 0.04:
        ax.text(p + 0.015, bar.get_y() + bar.get_height() / 2, f"{p*100:.1f}%",
                va="center", fontsize=8, color="#c9d1d9")
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# ── Kronecker Embeddings ────────────────────────────────────────────────────

if model_key == "KViT":
    st.markdown('<div class="kv-section">Kronecker Patch Embeddings <span>CORE CONTRIBUTION</span></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="kv-explain">
        Below is the actual output of the proposed <strong>Kronecker Patch Embedding</strong> layer.
        Each of the 64 rows is one patch; each of the 128 columns is one dimension of the embedding.
        These embeddings are formed by summing Kronecker products of local features and position vectors
        <em>inside</em> every patch — not by a simple linear projection.
    </div>
    """, unsafe_allow_html=True)

    with torch.no_grad():
        patch_emb = model.get_patch_embeddings(img_tensor)  # (1, 64, 128)
        emb_np = patch_emb.squeeze(0).cpu().numpy()

    # Stats
    emb_mean = float(emb_np.mean())
    emb_std = float(emb_np.std())
    emb_norm = float(np.linalg.norm(emb_np, axis=1).mean())

    stat_cols = st.columns(4)
    stats = [
        ("Shape", "64 × 128"),
        ("Mean", f"{emb_mean:.4f}"),
        ("Std", f"{emb_std:.4f}"),
        ("Avg L2 Norm", f"{emb_norm:.3f}"),
    ]
    for col, (label, val) in zip(stat_cols, stats):
        with col:
            st.markdown(
                f"""
                <div class="kv-chip">
                    <div class="kv-chip-label">{label}</div>
                    <div class="kv-chip-val">{val}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    fig, ax = plt.subplots(figsize=(12, 4.2))
    im = ax.imshow(emb_np, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_xlabel("Embedding dimension (0 → 127)", fontsize=10)
    ax.set_ylabel("Patch index (0 → 63)", fontsize=10)
    ax.set_title("Kronecker Patch Embedding Matrix", fontsize=12, color="#f0f6fc", pad=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="#8b949e")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#8b949e")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("""
    <div class="kv-explain">
        <strong>Interpretation:</strong> Bright vertical bands indicate dimensions that are consistently
        activated across many patches (shared structure). Horizontal variation shows how different
        spatial regions of the image produce distinct embedding signatures. Because of the Kronecker
        composition, these patterns are constrained to be factorisable into content × position.
    </div>
    """, unsafe_allow_html=True)

    # Cosine similarity
    st.markdown('<div class="kv-card-title" style="margin-top:1.2rem;">Pairwise Cosine Similarity of Patch Embeddings</div>', unsafe_allow_html=True)

    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(emb_np)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(sim, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title("Cosine Similarity (64 × 64)", fontsize=11, color="#f0f6fc", pad=8)
    ax.set_xlabel("Patch index")
    ax.set_ylabel("Patch index")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="#8b949e")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    st.markdown("""
    <div class="kv-explain">
        <strong>What this shows:</strong> Diagonal is always 1 (a patch is identical to itself).
        Bright off-diagonal blocks reveal groups of patches that the embedding has made similar —
        often corresponding to spatially adjacent or semantically related regions (e.g. sky patches,
        object body patches). Dark regions indicate dissimilar patches. This matrix is a direct
        view of the geometry induced by the Kronecker composition.
    </div>
    """, unsafe_allow_html=True)

# ── Attention ───────────────────────────────────────────────────────────────

if attn_maps is not None:
    st.markdown('<div class="kv-section">Transformer Attention <span>STEP 3</span></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="kv-explain">
        After the patch embeddings are formed, a stack of Transformer blocks lets every token
        attend to every other token. The maps below show how much the <strong>[CLS] token</strong>
        (used for classification) attends to each of the 64 spatial patches. Brighter = higher attention.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 3])
    with c1:
        layer_idx = st.slider("Layer", 0, len(attn_maps) - 1, len(attn_maps) - 1,
                              help="Deeper layers usually show more semantic focus.")
        head_idx = st.slider("Attention Head", 0, attn_maps[0].shape[1] - 1, 0)

    attn = attn_maps[layer_idx][0, head_idx].cpu().numpy()
    cls_attn = attn[0, 1:]  # CLS → patches
    n = int(np.sqrt(len(cls_attn)))
    grid = cls_attn.reshape(n, n)

    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(grid, cmap="inferno", interpolation="nearest")
    ax.set_title(f"CLS Attention  ·  Layer {layer_idx}  ·  Head {head_idx}",
                 fontsize=11, color="#f0f6fc", pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="#8b949e")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    st.markdown("""
    <div class="kv-explain">
        <strong>How to read attention maps:</strong><br>
        • Early layers tend to attend more uniformly or to local neighbourhoods.<br>
        • Later layers often focus on the most discriminative object parts.<br>
        • Different heads can specialise (one head on texture, another on shape boundaries).<br>
        • The [CLS] token aggregates information; high attention weight means that patch strongly
          influenced the final classification decision.
    </div>
    """, unsafe_allow_html=True)

# ── Architecture reminder ───────────────────────────────────────────────────

st.markdown('<div class="kv-section">Architecture Overview</div>', unsafe_allow_html=True)

st.markdown("""
<div class="kv-card">
<div style="font-family:'JetBrains Mono',monospace; font-size:0.82rem; color:#8b949e; line-height:1.7;">
<span style="color:#58a6ff;">Image</span> (3×32×32)<br>
&nbsp;&nbsp;│<br>
&nbsp;&nbsp;▼<br>
<span style="color:#a371f7;">Kronecker Patch Embedding</span> &nbsp;→&nbsp; 64 tokens × 128 dims<br>
&nbsp;&nbsp;│ &nbsp;&nbsp;<span style="color:#484f58;">E = Σ (eᵢ ⊗ pᵢ)</span><br>
&nbsp;&nbsp;▼<br>
+ [CLS] token + Absolute Positional Embedding<br>
&nbsp;&nbsp;│<br>
&nbsp;&nbsp;▼<br>
<span style="color:#58a6ff;">Transformer Encoder × 6</span> &nbsp;(4 heads, MLP ratio 4)<br>
&nbsp;&nbsp;│<br>
&nbsp;&nbsp;▼<br>
[CLS] → Linear head → 10-class logits
</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="kv-divider"></div>
<div style="text-align:center; font-size:0.75rem; color:#484f58; padding: 0.5rem 0 1rem 0;">
    Kronecker Vision Lab &nbsp;·&nbsp; Research Prototype &nbsp;·&nbsp; Not an implementation of an existing paper
</div>
""", unsafe_allow_html=True)
