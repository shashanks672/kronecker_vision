"""
Kronecker Patch Embedding — Premium Research Demo
A high-end neural / research-lab style interface.
"""

from __future__ import annotations

import sys
from pathlib import Path

from io import BytesIO

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
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

# Embedding analysis helpers (live next to this file)
sys.path.insert(0, str(ROOT / "app"))
from embedding_analysis import (
    extract_cnn_embedding,
    extract_vit_embedding,
    compute_stats,
    plot_embedding_bar,
    plot_embedding_heatmap,
    plot_histogram,
    plot_pca_patches,
    plot_tsne_patches,
    format_raw_values,
)


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


@st.cache_resource
def load_all_models(device_str: str):
    """Load CNN, ViT, and Kronecker-ViT for side-by-side embedding comparison."""
    cfg = load_config()
    device = torch.device(device_str)
    weights = {
        "CNN": ROOT / "weights" / "cnn_best.pt",
        "ViT": ROOT / "weights" / "vit_best.pt",
        "KViT": ROOT / "weights" / "kvit_best.pt",
    }
    models = {}
    loaded_flags = {}
    for name, path in weights.items():
        if name == "CNN":
            m = BaselineCNN(CNNConfig(num_classes=10))
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
            m = build_vit(vit_cfg) if name == "ViT" else build_kvit(vit_cfg)
        ok = path.exists()
        if ok:
            ckpt = torch.load(path, map_location=device)
            m.load_state_dict(ckpt["model_state_dict"])
        m = m.to(device).eval()
        models[name] = m
        loaded_flags[name] = ok
    return models, device, loaded_flags


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
# Sidebar collapse state
# ─────────────────────────────────────────────────────────────────────────────

if "sidebar_collapsed" not in st.session_state:
    st.session_state.sidebar_collapsed = False

# CSS: collapsed rail (~60px) + fixed expand button that always stays visible
if st.session_state.sidebar_collapsed:
    st.markdown("""
    <style>
    /* Narrow the Streamlit sidebar to a thin rail */
    section[data-testid="stSidebar"] {
        min-width: 60px !important;
        max-width: 60px !important;
        width: 60px !important;
        overflow: hidden !important;
    }
    section[data-testid="stSidebar"] > div {
        width: 60px !important;
        padding: 0 !important;
    }
    /* Hide all normal sidebar content while collapsed */
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] > div {
        opacity: 0 !important;
        pointer-events: none !important;
        height: 0 !important;
        overflow: hidden !important;
    }
    /* Hide Streamlit's built-in collapse control (we use our own) */
    button[kind="header"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    /* Fixed expand (>>) button — always on top at left edge */
    div.st-key-expand_sidebar,
    div[data-testid="stButton"]:has(button[aria-label="Expand sidebar"]),
    .kv-expand-wrap {
        position: fixed !important;
        left: 8px !important;
        top: 80px !important;
        z-index: 2147483647 !important;
        width: 44px !important;
    }
    div.st-key-expand_sidebar button,
    .kv-expand-wrap button {
        width: 44px !important;
        height: 44px !important;
        border-radius: 10px !important;
        background: linear-gradient(145deg, #1f6feb, #388bfd) !important;
        color: #fff !important;
        border: 1px solid rgba(88,166,255,0.5) !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 18px rgba(56,139,253,0.35) !important;
        padding: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    /* Ensure built-in control doesn't fight our custom toggle when expanded */
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Floating expand button (main area) — only when collapsed; always visible via fixed CSS
if st.session_state.sidebar_collapsed:
    # Marker + button; CSS targets .st-key-expand_sidebar (Streamlit ≥1.33) and sibling patterns
    exp_col = st.container()
    with exp_col:
        if st.button("≫", key="expand_sidebar", help="Expand sidebar", type="primary"):
            st.session_state.sidebar_collapsed = False
            st.rerun()
    st.markdown("""
    <style>
    /* Stronger fixed positioning for the expand control */
    div.st-key-expand_sidebar {
        position: fixed !important;
        left: 8px !important;
        top: 80px !important;
        z-index: 2147483647 !important;
        width: 44px !important;
        margin: 0 !important;
    }
    div.st-key-expand_sidebar button {
        width: 44px !important;
        min-height: 44px !important;
        border-radius: 10px !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        padding: 0 !important;
        box-shadow: 0 4px 18px rgba(56,139,253,0.4) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    # Collapse control — visible only when expanded
    if not st.session_state.sidebar_collapsed:
        c1, c2 = st.columns([1, 3])
        with c1:
            if st.button("≪", key="collapse_sidebar", help="Collapse sidebar"):
                st.session_state.sidebar_collapsed = True
                st.rerun()
        st.markdown("""
        <style>
        div.st-key-collapse_sidebar button {
            width: 40px !important;
            min-height: 36px !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
            padding: 0 !important;
            background: #21262d !important;
            border: 1px solid #30363d !important;
            color: #c9d1d9 !important;
        }
        div.st-key-collapse_sidebar button:hover {
            border-color: #58a6ff !important;
            color: #58a6ff !important;
        }
        </style>
        """, unsafe_allow_html=True)

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
    st.markdown(
        '<div style="font-size:0.7rem;color:#8b949e;text-transform:uppercase;'
        'letter-spacing:0.06em;margin-bottom:0.5rem;">Input Image</div>',
        unsafe_allow_html=True,
    )

    # ── Session-state image store (survives tab navigation) ─────────────────
    if "upload_key" not in st.session_state:
        st.session_state.upload_key = 0
    if "uploaded_image_bytes" not in st.session_state:
        st.session_state.uploaded_image_bytes = None
    if "uploaded_image_name" not in st.session_state:
        st.session_state.uploaded_image_name = None
    if "uploaded_image_size" not in st.session_state:
        st.session_state.uploaded_image_size = None

    if st.session_state.uploaded_image_bytes is None:
    # Empty state → show uploader in sidebar
        with st.sidebar:
            st.markdown("### 📷 Upload Image")

            uploaded = st.file_uploader(
                "Upload PNG / JPG",
                 type=["png", "jpg", "jpeg"],
                 key=f"uploader_{st.session_state.upload_key}",
        )
        if uploaded is not None:
            st.session_state.uploaded_image_bytes = uploaded.getvalue()
            st.session_state.uploaded_image_name = uploaded.name
            st.session_state.uploaded_image_size = len(st.session_state.uploaded_image_bytes)
            st.rerun()
    else:
        # Filled state → thumbnail + caption + Change Image
        try:
            thumb = Image.open(BytesIO(st.session_state.uploaded_image_bytes)).convert("RGB")
            thumb.thumbnail((180, 180))
            st.image(thumb, use_container_width=True)
        except Exception:
            st.warning("Could not preview image.")

        name = st.session_state.uploaded_image_name or "image"
        size_b = st.session_state.uploaded_image_size or 0
        size_str = f"{size_b / 1024:.1f} KB" if size_b < 1024 * 1024 else f"{size_b / (1024 * 1024):.2f} MB"
        st.caption(f"{name}  ·  {size_str}")

        if st.button("Change Image", use_container_width=True, key="btn_change_image"):
            st.session_state.uploaded_image_bytes = None
            st.session_state.uploaded_image_name = None
            st.session_state.uploaded_image_size = None
            st.session_state.upload_key += 1  # forces a brand-new uploader widget
            st.rerun()

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

# Image handling (from session_state — survives tab changes)
if st.session_state.uploaded_image_bytes is not None:
    try:
        pil_img = Image.open(BytesIO(st.session_state.uploaded_image_bytes)).convert("RGB")
    except Exception:
        st.warning("Failed to load the stored image — using a random sample.")
        arr = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
        pil_img = Image.fromarray(arr)
else:
    st.markdown("""
    <div class="kv-explain">
        <strong>No image uploaded.</strong> A random noise sample is shown below.
        Upload any image in the sidebar — it will be resized to 32×32 and processed by the selected model.
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
    with st.expander("Advanced Analysis · Kronecker Patch Matrix & Cosine Similarity", expanded=False):
        st.markdown("""
        <div class="kv-explain">
            Actual output of the proposed <strong>Kronecker Patch Embedding</strong> layer.
            Rows = patches, columns = dims. Formed by Σ (eᵢ ⊗ pᵢ) inside every patch.
        </div>
        """, unsafe_allow_html=True)

        with torch.no_grad():
            patch_emb = model.get_patch_embeddings(img_tensor)
            emb_np = patch_emb.squeeze(0).cpu().numpy()

        emb_mean = float(emb_np.mean())
        emb_std = float(emb_np.std())
        emb_norm = float(np.linalg.norm(emb_np, axis=1).mean())

        stat_cols = st.columns(4)
        for col, (label, val) in zip(stat_cols, [
            ("Shape", "64 × 128"),
            ("Mean", f"{emb_mean:.4f}"),
            ("Std", f"{emb_std:.4f}"),
            ("Avg L2", f"{emb_norm:.3f}"),
        ]):
            with col:
                st.markdown(
                    f'<div class="kv-chip"><div class="kv-chip-label">{label}</div>'
                    f'<div class="kv-chip-val">{val}</div></div>',
                    unsafe_allow_html=True,
                )

        fig, ax = plt.subplots(figsize=(10, 3.2))
        im = ax.imshow(emb_np, aspect="auto", cmap="magma", interpolation="nearest")
        ax.set_xlabel("Dimension", fontsize=8)
        ax.set_ylabel("Patch", fontsize=8)
        ax.set_title("Kronecker Patch Embedding Matrix", fontsize=10, color="#f0f6fc", pad=6)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        from sklearn.metrics.pairwise import cosine_similarity
        sim = cosine_similarity(emb_np)
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        im = ax.imshow(sim, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title("Cosine Similarity (64 × 64)", fontsize=10, color="#f0f6fc", pad=6)
        ax.set_xlabel("Patch")
        ax.set_ylabel("Patch")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

# ── Attention ───────────────────────────────────────────────────────────────

if attn_maps is not None:
    with st.expander("Advanced Analysis · Transformer Attention (CLS → patches)", expanded=False):
        st.markdown("""
        <div class="kv-explain">
            After the patch embeddings are formed, Transformer blocks let every token attend to
            every other token. Maps show how much the <strong>[CLS] token</strong> attends to
            each of the 64 spatial patches. Brighter = higher attention.
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns([1, 3])
        with c1:
            layer_idx = st.slider("Layer", 0, len(attn_maps) - 1, len(attn_maps) - 1,
                                  help="Deeper layers usually show more semantic focus.")
            head_idx = st.slider("Attention Head", 0, attn_maps[0].shape[1] - 1, 0)

        attn = attn_maps[layer_idx][0, head_idx].cpu().numpy()
        cls_attn = attn[0, 1:]
        n = int(np.sqrt(len(cls_attn)))
        grid = cls_attn.reshape(n, n)

        fig, ax = plt.subplots(figsize=(4.5, 3.6))
        im = ax.imshow(grid, cmap="inferno", interpolation="nearest")
        ax.set_title(f"CLS Attention  ·  Layer {layer_idx}  ·  Head {head_idx}",
                     fontsize=10, color="#f0f6fc", pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color="#8b949e")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

        st.caption("Early layers ≈ local / uniform · Late layers ≈ semantic focus · Heads specialise")

# ── Embedding Analysis (compact research dashboard) ─────────────────────────

st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 1.5rem; max-width: 1480px;}
.ea-hero {
    background: linear-gradient(135deg, rgba(56,139,253,0.10) 0%, rgba(163,113,247,0.08) 100%);
    border: 1px solid #21262d;
    border-radius: 14px;
    padding: 0.9rem 1.2rem;
    margin: 0.2rem 0 0.9rem 0;
}
.ea-hero h2 {
    margin: 0 0 0.2rem 0;
    font-size: 1.25rem;
    font-weight: 700;
    color: #f0f6fc;
    letter-spacing: -0.02em;
}
.ea-hero p {
    margin: 0;
    color: #8b949e;
    font-size: 0.82rem;
    line-height: 1.4;
}
.ea-metric {
    background: linear-gradient(160deg, rgba(22,27,34,0.95), rgba(13,17,23,0.98));
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 0.75rem 0.85rem;
    transition: transform 0.15s ease, border-color 0.15s ease;
    height: 100%;
}
.ea-metric:hover {
    transform: translateY(-1px);
    border-color: #58a6ff;
}
.ea-metric .label {
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #8b949e;
    font-weight: 600;
    margin-bottom: 0.3rem;
}
.ea-metric .value {
    font-size: 1.2rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    color: #f0f6fc;
    line-height: 1.1;
}
.ea-metric .icon {
    font-size: 0.85rem;
    margin-bottom: 0.2rem;
    opacity: 0.9;
}
.ea-panel {
    background: linear-gradient(160deg, rgba(22,27,34,0.92), rgba(13,17,23,0.96));
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 0.7rem 0.8rem 0.85rem 0.8rem;
    margin-bottom: 0.65rem;
    height: 100%;
}
.ea-panel-title {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #8b949e;
    margin-bottom: 0.45rem;
}
.ea-panel-sub {
    font-size: 0.68rem;
    color: #6e7681;
    margin: -0.25rem 0 0.4rem 0;
}
.ea-strip {
    height: 8px;
    border-radius: 5px;
    background: linear-gradient(90deg, #f85149 0%, #8b949e 50%, #3fb950 100%);
    margin: 0.45rem 0 0.15rem 0;
}
.ea-strip-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.65rem;
    color: #8b949e;
    font-family: 'JetBrains Mono', monospace;
}
.ea-stats-box {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #c9d1d9;
    line-height: 1.65;
}
.ea-stats-box span.label { color: #8b949e; display: inline-block; width: 7.2rem; }
.ea-stats-box span.val { color: #f0f6fc; font-weight: 500; }
div[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #30363d;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="kv-section">Embedding Analysis <span>LAB</span></div>', unsafe_allow_html=True)

# Extract embedding for the currently selected model (UNCHANGED LOGIC)
with torch.no_grad():
    if model_key == "CNN":
        primary_result = extract_cnn_embedding(model, img_tensor)
    elif model_key == "ViT":
        primary_result = extract_vit_embedding(model, img_tensor, name="ViT")
    else:
        primary_result = extract_vit_embedding(model, img_tensor, name="Kronecker-ViT")

primary_stats = compute_stats(primary_result.embedding)

# Hero
st.markdown(f"""
<div class="ea-hero">
    <h2>{primary_result.model_name} Embedding</h2>
    <p>{primary_result.source}</p>
</div>
""", unsafe_allow_html=True)

# ── Six metric cards ────────────────────────────────────────────────────────
mcols = st.columns(6)
_metric_specs = [
    (mcols[0], "#", "Dimension", f"{primary_stats['dim']}"),
    (mcols[1], "μ", "Mean", f"{primary_stats['mean']:.4f}"),
    (mcols[2], "σ", "Std Dev", f"{primary_stats['std']:.4f}"),
    (mcols[3], "‖·‖", "L2 Norm", f"{primary_stats['l2_norm']:.3f}"),
    (mcols[4], "↓", "Minimum", f"{primary_stats['min']:.3f}"),
    (mcols[5], "↑", "Maximum", f"{primary_stats['max']:.3f}"),
]
for col, icon, label, val in _metric_specs:
    with col:
        st.markdown(f"""
        <div class="ea-metric">
            <div class="icon">{icon}</div>
            <div class="label">{label}</div>
            <div class="value">{val}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height:0.55rem'></div>", unsafe_allow_html=True)

# ── Raw embedding table (column-wise) ───────────────────────────────────────
st.markdown("""
<div class="ea-panel" style="margin-bottom:0.7rem;">
<div class="ea-panel-title">Raw Embedding Values (Column-wise)</div>
<div class="ea-panel-sub">Each column is one dimension of the embedding vector</div>
</div>
""", unsafe_allow_html=True)

max_dim = int(primary_stats["dim"])
choices = [n for n in (16, 32, 64, 128, 256) if n <= max_dim]
if max_dim not in choices:
    choices.append(max_dim)
default_idx = 1 if 32 in choices else 0

ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1.2, 4])
with ctrl1:
    n_show = st.selectbox("Show", options=choices, index=default_idx, key="ea_n_dims", label_visibility="collapsed")
with ctrl2:
    st.caption(f"Showing {n_show} of {max_dim} dims")
with ctrl3:
    full_df = pd.DataFrame({
        "dimension": list(range(max_dim)),
        "value": primary_result.embedding.astype(float),
    })
    st.download_button(
        label="⬇ Download CSV",
        data=full_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{primary_result.model_name.lower().replace(' ', '_')}_embedding.csv",
        mime="text/csv",
    )

row_df = pd.DataFrame(
    [primary_result.embedding[:n_show].astype(float)],
    columns=[f"Dim {i}" for i in range(n_show)],
    index=["Value"],
)
vmax = float(abs(primary_result.embedding[:n_show]).max()) or 1.0
st.dataframe(
    row_df.style.format("{:+.4f}").background_gradient(
        cmap="RdYlGn", axis=1, vmin=-vmax, vmax=vmax,
    ),
    use_container_width=True,
    height=78,
)

st.markdown("""
<div class="ea-strip"></div>
<div class="ea-strip-labels"><span>negative</span><span>0</span><span>positive</span></div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ── Visualization grid: row 1 (4 panels) ────────────────────────────────────
r1c1, r1c2, r1c3, r1c4 = st.columns(4)

with r1c1:
    st.markdown('<div class="ea-panel"><div class="ea-panel-title">Embedding Bar Chart</div><div class="ea-panel-sub">First N dimensions</div></div>', unsafe_allow_html=True)
    fig = plot_embedding_bar(
        primary_result.embedding, n_show=n_show,
        title="", figsize=(3.8, 2.4),
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with r1c2:
    st.markdown('<div class="ea-panel"><div class="ea-panel-title">Embedding Heatmap</div><div class="ea-panel-sub">Full vector</div></div>', unsafe_allow_html=True)
    fig = plot_embedding_heatmap(
        primary_result.embedding.reshape(1, -1),
        title="", figsize=(3.8, 2.4),
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with r1c3:
    st.markdown('<div class="ea-panel"><div class="ea-panel-title">Distribution (Histogram)</div><div class="ea-panel-sub">Value distribution</div></div>', unsafe_allow_html=True)
    fig = plot_histogram(primary_result.embedding, title="", figsize=(3.8, 2.4), bins=40)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with r1c4:
    st.markdown('<div class="ea-panel"><div class="ea-panel-title">Embedding Statistics</div><div class="ea-panel-sub">Summary of values</div></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="ea-panel" style="margin-top:-0.4rem;">
    <div class="ea-stats-box">
        <div><span class="label">Mean</span><span class="val">{primary_stats['mean']:+.4f}</span></div>
        <div><span class="label">Std Deviation</span><span class="val">{primary_stats['std']:.4f}</span></div>
        <div><span class="label">Variance</span><span class="val">{primary_stats['std']**2:.4f}</span></div>
        <div><span class="label">Min Value</span><span class="val">{primary_stats['min']:+.4f}</span></div>
        <div><span class="label">Max Value</span><span class="val">{primary_stats['max']:+.4f}</span></div>
        <div><span class="label">Sparsity</span><span class="val">{primary_stats['sparsity']*100:.2f}%</span></div>
        <div><span class="label">L1 Norm</span><span class="val">{primary_stats['l1_norm']:.4f}</span></div>
        <div><span class="label">L2 Norm</span><span class="val">{primary_stats['l2_norm']:.4f}</span></div>
    </div>
    </div>
    """, unsafe_allow_html=True)

# ── Visualization grid: row 2 (PCA / t-SNE / Patch matrix) ──────────────────
r2c1, r2c2, r2c3 = st.columns(3)

with r2c1:
    st.markdown('<div class="ea-panel"><div class="ea-panel-title">PCA Projection (2D)</div></div>', unsafe_allow_html=True)
    if primary_result.patch_matrix is not None:
        fig = plot_pca_patches(primary_result.patch_matrix, title="", figsize=(4.2, 2.8))
        if fig is not None:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    else:
        st.caption("PCA available for ViT / Kronecker-ViT only")

with r2c2:
    st.markdown('<div class="ea-panel"><div class="ea-panel-title">t-SNE Projection (2D)</div></div>', unsafe_allow_html=True)
    if primary_result.patch_matrix is not None:
        with st.spinner("t-SNE…"):
            fig = plot_tsne_patches(primary_result.patch_matrix, title="", figsize=(4.2, 2.8))
        if fig is not None:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.caption("t-SNE unavailable")
    else:
        st.caption("t-SNE available for ViT / Kronecker-ViT only")

with r2c3:
    st.markdown('<div class="ea-panel"><div class="ea-panel-title">Patch Embedding Matrix</div><div class="ea-panel-sub">Before CLS aggregation</div></div>', unsafe_allow_html=True)
    if primary_result.patch_matrix is not None:
        fig = plot_embedding_heatmap(
            primary_result.patch_matrix,
            title="",
            figsize=(4.5, 2.8),
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.caption("Patch matrix available for ViT / Kronecker-ViT only")

# ── Model Comparison ────────────────────────────────────────────────────────
st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)
st.markdown('<div class="ea-panel-title" style="margin-bottom:0.4rem;">Model Comparison (Same Input Image)</div>', unsafe_allow_html=True)

all_models, cmp_device, flags = load_all_models(device_str)
img_cmp = img_tensor.to(cmp_device)

results = {}
with torch.no_grad():
    results["CNN"] = extract_cnn_embedding(all_models["CNN"], img_cmp)
    results["ViT"] = extract_vit_embedding(all_models["ViT"], img_cmp, name="ViT")
    results["Kronecker-ViT"] = extract_vit_embedding(
        all_models["KViT"], img_cmp, name="Kronecker-ViT"
    )

cmp_rows = []
best_conf = -1.0
best_name = None
for name, res in results.items():
    s = compute_stats(res.embedding)
    pred_name = class_names[res.pred_idx] if res.pred_idx is not None else "—"
    conf = float(res.confidence) if res.confidence is not None else 0.0
    key = "KViT" if name == "Kronecker-ViT" else name
    loaded = "●" if flags.get(key, False) else "○"
    if conf > best_conf:
        best_conf = conf
        best_name = name
    cmp_rows.append({
        "Model": f"{loaded} {name}",
        "Embedding Dim": s["dim"],
        "Mean": round(s["mean"], 4),
        "Std": round(s["std"], 4),
        "Min": round(s["min"], 3),
        "Max": round(s["max"], 3),
        "L2 Norm": round(s["l2_norm"], 3),
        "Prediction": pred_name,
        "Confidence": f"{conf * 100:.1f}%",
    })

cmp_df = pd.DataFrame(cmp_rows)

def _highlight_best(row):
    if best_name and best_name in str(row["Model"]):
        return ["background-color: rgba(63,185,80,0.14)"] * len(row)
    return [""] * len(row)

st.dataframe(
    cmp_df.style.apply(_highlight_best, axis=1),
    use_container_width=True,
    hide_index=True,
)
st.caption("● trained weights loaded  ·  ○ random init  ·  green row = highest confidence")

# Overlay histogram (compact)
with st.expander("Distribution overlay across models", expanded=False):
    fig, ax = plt.subplots(figsize=(9, 2.6))
    colors = {"CNN": "#f85149", "ViT": "#388bfd", "Kronecker-ViT": "#a371f7"}
    for name, res in results.items():
        ax.hist(
            res.embedding, bins=40, alpha=0.45, label=name,
            color=colors.get(name, "#8b949e"), density=True,
        )
    ax.set_xlabel("Embedding value", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.markdown("""
<div class="kv-explain" style="margin-top:0.5rem;">
    <strong>Why these embeddings differ</strong> —
    <strong>CNN</strong>: 256-d global conv features ·
    <strong>ViT</strong>: linear patch proj + CLS ·
    <strong>Kronecker-ViT</strong>: Σ (content ⊗ position) per patch, then same Transformer
</div>
""", unsafe_allow_html=True)

# ── Architecture reminder ───────────────────────────────────────────────────

with st.expander("Architecture Overview", expanded=False):
    st.markdown("""
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
    """, unsafe_allow_html=True)

st.markdown("""
<div class="kv-divider"></div>
<div style="text-align:center; font-size:0.75rem; color:#484f58; padding: 0.5rem 0 1rem 0;">
    Kronecker Vision Lab &nbsp;·&nbsp; Research Prototype &nbsp;·&nbsp; Not an implementation of an existing paper
</div>
""", unsafe_allow_html=True)
