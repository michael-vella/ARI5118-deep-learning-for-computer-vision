"""
ARI5118 - Deep Learning for Computer Vision
Topic 2: Pooling, Normalisation and Activation Functions
Interactive Principle Simulator

Run with:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import cv2
from PIL import Image
import io

# ─────────────────────────────────────────
#  PAGE CONFIG & GLOBAL STYLE
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Pooling · Norm · Activations",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a clean, dark-academic aesthetic
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
  }
  code, .stCode, pre {
    font-family: 'DM Mono', monospace !important;
  }

  /* Dark sidebar */
  [data-testid="stSidebar"] {
    background: #0f0f14 !important;
    border-right: 1px solid #2a2a3a;
  }
  [data-testid="stSidebar"] * { color: #e8e8f0 !important; }
  [data-testid="stSidebar"] .stSlider label,
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stRadio label { color: #a0a0c0 !important; font-size: 0.82rem; }

  /* Main background */
  .main { background: #12121a; }
  .block-container { padding-top: 1.5rem; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background: #1a1a26;
    border-radius: 8px;
    padding: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 6px;
    color: #7070a0;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    padding: 8px 20px;
  }
  .stTabs [aria-selected="true"] {
    background: #6c63ff !important;
    color: #fff !important;
  }

  /* Cards */
  .info-card {
    background: #1a1a26;
    border: 1px solid #2a2a40;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    font-size: 0.88rem;
    color: #b0b0d0;
    line-height: 1.6;
  }
  .info-card b { color: #a78bfa; }
  .metric-badge {
    display: inline-block;
    background: #6c63ff22;
    border: 1px solid #6c63ff55;
    color: #a78bfa;
    border-radius: 6px;
    padding: 2px 10px;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    margin: 2px;
  }

  /* Section headers */
  h1 { color: #e8e8f8 !important; font-weight: 800; letter-spacing: -1px; }
  h2 { color: #c8c8e8 !important; font-weight: 700; }
  h3 { color: #a8a8d0 !important; font-weight: 600; font-size: 1rem; }

  /* Matplotlib figures */
  [data-testid="stImage"] img { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  MATPLOTLIB THEME
# ─────────────────────────────────────────
DARK_BG   = "#12121a"
CARD_BG   = "#1a1a26"
GRID_CLR  = "#2a2a40"
TEXT_CLR  = "#c8c8e8"
ACCENT1   = "#6c63ff"   # violet
ACCENT2   = "#ff6584"   # rose
ACCENT3   = "#43e97b"   # green
ACCENT4   = "#f7971e"   # amber

def apply_dark_theme(fig, axes_list=None):
    """Apply the dark theme to a matplotlib figure."""
    fig.patch.set_facecolor(DARK_BG)
    if axes_list is None:
        axes_list = fig.get_axes()
    for ax in axes_list:
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors=TEXT_CLR, labelsize=8)
        ax.xaxis.label.set_color(TEXT_CLR)
        ax.yaxis.label.set_color(TEXT_CLR)
        ax.title.set_color(TEXT_CLR)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_CLR)
        ax.grid(True, color=GRID_CLR, linewidth=0.5, linestyle="--", alpha=0.6)
    return fig

def fig_to_img(fig):
    """Convert matplotlib figure to a PIL Image for st.image()."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=DARK_BG)
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)

# ─────────────────────────────────────────
#  SYNTHETIC IMAGE HELPERS
# ─────────────────────────────────────────
@st.cache_data
def make_test_feature_map(style: str, size: int = 16) -> np.ndarray:
    """
    Generate a synthetic 2-D feature map for pooling demonstrations.
    Returns a (size, size) float32 array in [0, 1].
    """
    rng = np.random.default_rng(42)
    if style == "Gradient":
        x = np.linspace(0, 1, size)
        fm = np.outer(x, x)
    elif style == "Checkerboard":
        fm = np.indices((size, size)).sum(axis=0) % 2 * 0.9
    elif style == "Gaussian Blobs":
        fm = np.zeros((size, size), dtype=np.float32)
        for _ in range(4):
            cx, cy = rng.integers(2, size - 2, size=2)
            sigma = rng.uniform(1.0, 3.0)
            for i in range(size):
                for j in range(size):
                    fm[i, j] += np.exp(-((i - cx)**2 + (j - cy)**2) / (2 * sigma**2))
    elif style == "Random Noise":
        fm = rng.random((size, size)).astype(np.float32)
    else:  # "Edges"
        fm = np.zeros((size, size), dtype=np.float32)
        fm[size//4:3*size//4, size//4:3*size//4] = 0.8
        fm = cv2.Laplacian(fm, cv2.CV_32F) + 0.5
    fm = fm.astype(np.float32)
    fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
    return fm

# ─────────────────────────────────────────
#  POOLING FUNCTIONS
# ─────────────────────────────────────────
def apply_pooling(fm: np.ndarray, mode: str, kernel: int, stride: int) -> np.ndarray:
    """
    Apply 2-D pooling to a feature map.
    mode: 'Max' | 'Average'
    Returns pooled output as float32 array.
    """
    H, W = fm.shape

    out_h = (H - kernel) // stride + 1
    out_w = (W - kernel) // stride + 1
    out = np.zeros((out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            patch = fm[i*stride:i*stride+kernel, j*stride:j*stride+kernel]
            out[i, j] = patch.max() if mode == "Max" else patch.mean()
    return out

def output_size_formula(H: int, kernel: int, stride: int, padding: int = 0) -> int:
    """Standard formula: ⌊(H + 2P − K) / S⌋ + 1"""
    return (H + 2 * padding - kernel) // stride + 1

# ─────────────────────────────────────────
#  NORMALISATION FUNCTIONS
# ─────────────────────────────────────────
def batch_norm(x: np.ndarray, eps: float = 1e-5) -> tuple:
    """
    Batch normalisation over (N, C, H, W) or (N, D).
    Normalises over the batch (N) dimension per feature.
    Returns (normalised, mean, var).
    """
    axes = tuple(i for i in range(x.ndim) if i != 1)  # all axes except channel
    mu  = x.mean(axis=axes, keepdims=True)
    var = x.var(axis=axes, keepdims=True)
    x_hat = (x - mu) / np.sqrt(var + eps)
    return x_hat, mu.squeeze(), var.squeeze()

def layer_norm(x: np.ndarray, eps: float = 1e-5) -> tuple:
    """
    Layer normalisation: normalises over all features for each sample.
    """
    axes = tuple(range(1, x.ndim))  # all axes except batch
    mu  = x.mean(axis=axes, keepdims=True)
    var = x.var(axis=axes, keepdims=True)
    x_hat = (x - mu) / np.sqrt(var + eps)
    return x_hat, mu.squeeze(), var.squeeze()

def instance_norm(x: np.ndarray, eps: float = 1e-5) -> tuple:
    """
    Instance normalisation: normalises over spatial dims (H, W) per sample per channel.
    Expects (N, C, H, W).
    """
    axes = (2, 3)
    mu  = x.mean(axis=axes, keepdims=True)
    var = x.var(axis=axes, keepdims=True)
    x_hat = (x - mu) / np.sqrt(var + eps)
    return x_hat, mu, var

def group_norm(x: np.ndarray, G: int, eps: float = 1e-5) -> tuple:
    """
    Group normalisation: splits channels into G groups, normalises within each group.
    """
    N, C, H, W = x.shape
    x_r = x.reshape(N, G, C // G, H, W)
    axes = (2, 3, 4)
    mu  = x_r.mean(axis=axes, keepdims=True)
    var = x_r.var(axis=axes, keepdims=True)
    x_hat = (x_r - mu) / np.sqrt(var + eps)
    return x_hat.reshape(N, C, H, W), mu, var

# ─────────────────────────────────────────
#  ACTIVATION FUNCTIONS
# ─────────────────────────────────────────
def relu(x):        return np.maximum(0, x)
def leaky_relu(x, alpha=0.01): return np.where(x >= 0, x, alpha * x)
def gelu(x):        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
def sigmoid(x):     return 1 / (1 + np.exp(-np.clip(x, -30, 30)))
def tanh_fn(x):     return np.tanh(x)

ACTIVATIONS = {
    "ReLU":        (relu,        {},             ACCENT1, "max(0, x)"),
    "Leaky ReLU":  (leaky_relu,  {"alpha": 0.1}, ACCENT2, "x if x≥0 else αx"),
    "GELU":        (gelu,        {},             ACCENT4, "0.5x·(1+tanh(√(2/π)(x+0.045x³)))"),
    "Sigmoid":     (sigmoid,     {},             "#38bdf8", "1/(1+e⁻ˣ)"),
    "Tanh":        (tanh_fn,     {},             "#f472b6", "tanh(x)"),
}

# ─────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────
st.markdown("""
<h1 style='font-size:2.4rem;margin-bottom:0;'>
  Pooling · Norm · Activations
  <span style='font-size:1.2rem;font-weight:400;color:#6060a0;margin-left:12px;'>
    Interactive Simulator
  </span>
</h1>
<p style='color:#5050a0;font-family:DM Mono,monospace;font-size:0.82rem;margin-top:4px;'>
  ARI5118 · Topic 2 · Streamlit · Author: michael.vella.20@um.edu.mt
</p>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "🟦  Pooling Explorer",
    "📊  Normalisation Lab",
    "⚡  Activation Functions",
])

# ═══════════════════════════════════════════════════════════════
#  TAB 1 ── POOLING EXPLORER
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Spatial Pooling — Visualise and Compare")
    st.markdown("""
<div class='info-card'>
Pooling downsamples a feature map by summarising local regions.
<b>Max pooling</b> retains the strongest activation while <b>average pooling</b> blends them.
</div>
""", unsafe_allow_html=True)

    st.markdown("#### ① Primary Configuration")
    fm_style = st.selectbox(
        "Feature map pattern",
        ["Random Noise", "Gaussian Blobs", "Gradient", "Checkerboard", "Edges"],
        key="p_style"
    )
    fm_size = st.slider("Feature map size (H=W)",  8, 32, 32, 2, key="p_size")
    pool_mode = st.selectbox("Pooling type", ["Max", "Average"], key="p_mode")
    kernel_sz = st.slider("Kernel size K", 2, 6, 2, key="p_kernel")
    stride = st.slider("Stride S", 1, 4, 2, key="p_stride")

    st.markdown("#### ② Comparison Configuration")
    pool_mode2 = st.selectbox("Pooling type (B)", ["Average", "Max"], key="p_mode2")
    kernel_sz2 = st.slider("Kernel size K (B)", 2, 6, 2, key="p_kernel2")
    stride2 = st.slider("Stride S (B)", 1, 4, 2, key="p_stride2")

    # ── Compute ───────────────────────────────────────────────
    fm = make_test_feature_map(fm_style, fm_size)

    out_a = apply_pooling(fm, pool_mode, kernel_sz, stride)
    out_b = apply_pooling(fm, pool_mode2, kernel_sz2, stride2)

    out_h_a = output_size_formula(fm_size, kernel_sz, stride)
    out_h_b = output_size_formula(fm_size, kernel_sz2, stride2)

    # ── Metrics strip ─────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Input shape",   f"{fm_size}×{fm_size}")
    m2.metric("Output A shape", f"{out_a.shape[0]}×{out_a.shape[1]}")
    m3.metric("Output B shape", f"{out_b.shape[0]}×{out_b.shape[1]}")
    reduction_a = (1 - out_a.size / fm.size) * 100
    reduction_b = (1 - out_b.size / fm.size) * 100
    m4.metric("Spatial reduction A", f"{reduction_a:.0f}%")
    m5.metric("Spatial reduction B", f"{reduction_b:.0f}%")

    # ── Figure ────────────────────────────────────────────────
    viridis_dark = plt.cm.viridis

    fig = plt.figure(figsize=(13, 4.5), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    def draw_heatmap(ax, data, title, cmap=viridis_dark):
        im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9, color=TEXT_CLR, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_CLR)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(
            labelsize=7, colors=TEXT_CLR)
        # Annotate small maps
        if data.shape[0] <= 8:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center",
                            fontsize=6.5, color="white",
                            fontweight="bold" if data[i,j] > 0.5 else "normal")

    ax0 = fig.add_subplot(gs[0, 0])
    draw_heatmap(ax0, fm, f"Input  {fm_size}×{fm_size}")

    ax1 = fig.add_subplot(gs[0, 1])
    draw_heatmap(ax1, out_a, f"Pooling A: {pool_mode}\n{out_a.shape[0]}×{out_a.shape[1]}")

    ax2 = fig.add_subplot(gs[0, 2])
    draw_heatmap(ax2, out_b, f"Pooling B: {pool_mode2}\n{out_b.shape[0]}×{out_b.shape[1]}")

    apply_dark_theme(fig)
    st.image(fig_to_img(fig), use_container_width=True)

# ═══════════════════════════════════════════════════════════════
#  TAB 2 ── NORMALISATION LAB
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Normalisation — Why, How, and Which?")
    st.markdown("""
<div class='info-card'>
Normalisation stabilises training by controlling the distribution of activations.
Different schemes normalise over <b>different axes</b> of the (N, C, H, W) tensor.
Adjust ε (epsilon) and observe how it guards against division by zero when variance collapses.
</div>
""", unsafe_allow_html=True)

    cn1, cn2 = st.columns([1, 1])
    with cn1:
        st.markdown("#### ① Primary Normalisation")
        norm_a = st.selectbox("Method A",
                              ["Batch Norm", "Layer Norm", "Instance Norm", "Group Norm"],
                              key="n_a")
        eps_a  = st.select_slider("ε (epsilon) A",
                                  options=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1],
                                  value=1e-5, key="n_eps_a",
                                  format_func=lambda v: f"{v:.0e}")
        if norm_a == "Group Norm":
            n_groups_a = st.slider("Num groups G (A)", 1, 8, 4, key="n_ga")
        dist_a = st.selectbox("Input distribution A",
                              ["Standard Normal", "Skewed", "Bimodal",
                               "High Variance", "Near-Zero"], key="n_dist_a")

    with cn2:
        st.markdown("#### ② Comparison Normalisation")
        norm_b = st.selectbox("Method B",
                              ["Layer Norm", "Batch Norm", "Instance Norm", "Group Norm"],
                              key="n_b")
        eps_b  = st.select_slider("ε (epsilon) B",
                                  options=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1],
                                  value=1e-5, key="n_eps_b",
                                  format_func=lambda v: f"{v:.0e}")
        if norm_b == "Group Norm":
            n_groups_b = st.slider("Num groups G (B)", 1, 8, 2, key="n_gb")
        batch_sz = st.slider("Batch size N", 2, 32, 8, key="n_bs")

    # ── Synthesise data ──────────────────────────────────────
    rng = np.random.default_rng(7)
    N, C, H, W = batch_sz, 8, 4, 4

    def make_dist(style, shape):
        rng2 = np.random.default_rng(99)
        if style == "Standard Normal":
            return rng2.normal(0, 1, shape).astype(np.float32)
        elif style == "Skewed":
            return (rng2.exponential(1, shape) - 1).astype(np.float32)
        elif style == "Bimodal":
            h = shape[0] // 2
            return np.concatenate([
                rng2.normal(-2, 0.5, (h,) + shape[1:]),
                rng2.normal(+2, 0.5, (shape[0]-h,) + shape[1:])
            ], axis=0).astype(np.float32)
        elif style == "High Variance":
            return rng2.normal(0, 5, shape).astype(np.float32)
        else:  # Near-Zero
            return rng2.normal(0, 0.01, shape).astype(np.float32)

    x_data = make_dist(dist_a, (N, C, H, W))

    # Apply normalisation A
    if norm_a == "Batch Norm":
        x_a, mu_a, var_a = batch_norm(x_data, eps=eps_a)
    elif norm_a == "Layer Norm":
        x_a, mu_a, var_a = layer_norm(x_data, eps=eps_a)
    elif norm_a == "Instance Norm":
        x_a, mu_a, var_a = instance_norm(x_data, eps=eps_a)
    else:
        ng = n_groups_a if 'n_groups_a' in dir() else 4
        ng = min(ng, C)
        while C % ng != 0: ng -= 1
        x_a, mu_a, var_a = group_norm(x_data, ng, eps=eps_a)

    # Apply normalisation B
    if norm_b == "Batch Norm":
        x_b, mu_b, var_b = batch_norm(x_data, eps=eps_b)
    elif norm_b == "Layer Norm":
        x_b, mu_b, var_b = layer_norm(x_data, eps=eps_b)
    elif norm_b == "Instance Norm":
        x_b, mu_b, var_b = instance_norm(x_data, eps=eps_b)
    else:
        ng2 = n_groups_b if 'n_groups_b' in dir() else 2
        ng2 = min(ng2, C)
        while C % ng2 != 0: ng2 -= 1
        x_b, mu_b, var_b = group_norm(x_data, ng2, eps=eps_b)

    # ── Metrics ───────────────────────────────────────────────
    ma, mb, mc, md = st.columns(4)
    ma.metric("Input  μ",  f"{x_data.mean():.3f}")
    mb.metric("Input  σ",  f"{x_data.std():.3f}")
    mc.metric("Output A μ", f"{x_a.mean():.4f}")
    md.metric("Output A σ", f"{x_a.std():.4f}")

    # ── Figure ────────────────────────────────────────────────
    fig2, axes = plt.subplots(2, 4, figsize=(13, 6), facecolor=DARK_BG)
    fig2.suptitle("Normalisation Comparison", color=TEXT_CLR, fontsize=11, y=1.01)
    apply_dark_theme(fig2, axes.ravel())

    # Row 0: input and A
    # Row 1: B and stats

    def plot_dist_row(ax_hist, ax_heat, data, label, color):
        flat = data.ravel()
        ax_hist.hist(flat, bins=40, color=color, alpha=0.8, density=True)
        ax_hist.axvline(flat.mean(), color="white", lw=1.2, linestyle="--",
                        label=f"μ={flat.mean():.2f}")
        ax_hist.axvline(flat.mean()+flat.std(), color="#f9ca24", lw=0.8,
                        linestyle=":", label=f"σ={flat.std():.2f}")
        ax_hist.axvline(flat.mean()-flat.std(), color="#f9ca24", lw=0.8, linestyle=":")
        ax_hist.set_title(label, fontsize=9, color=TEXT_CLR)
        ax_hist.legend(fontsize=7, facecolor=CARD_BG, edgecolor=GRID_CLR,
                       labelcolor=TEXT_CLR)
        ax_hist.set_xlabel("Activation value", fontsize=8)

        # Show spatial heatmap for sample 0, channel 0
        im = ax_heat.imshow(data[0, 0], cmap="RdBu_r", aspect="auto")
        ax_heat.set_title(f"{label}  [N=0, C=0]", fontsize=8, color=TEXT_CLR)
        ax_heat.set_xticks([]); ax_heat.set_yticks([])
        plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04).ax.tick_params(
            labelsize=7, colors=TEXT_CLR)

    plot_dist_row(axes[0, 0], axes[0, 1], x_data, "Input Distribution", TEXT_CLR)
    plot_dist_row(axes[0, 2], axes[0, 3], x_a,    f"After {norm_a}", ACCENT1)
    plot_dist_row(axes[1, 0], axes[1, 1], x_b,    f"After {norm_b}", ACCENT3)

    # Residual distribution (A vs B)
    diff_norm = x_a - x_b
    axes[1, 2].hist(diff_norm.ravel(), bins=40, color=ACCENT2, alpha=0.8, density=True)
    axes[1, 2].set_title(f"Δ {norm_a} − {norm_b}", fontsize=9, color=TEXT_CLR)
    axes[1, 2].axvline(0, color="white", lw=1, linestyle="--")
    axes[1, 2].set_xlabel("Difference", fontsize=8)

    # Channel-wise variance after norm A and B
    var_per_ch_a = x_a.var(axis=(0, 2, 3))  # (C,)
    var_per_ch_b = x_b.var(axis=(0, 2, 3))
    ch_idx = np.arange(C)
    axes[1, 3].bar(ch_idx - 0.2, var_per_ch_a, 0.35, color=ACCENT1, alpha=0.85,
                   label=norm_a)
    axes[1, 3].bar(ch_idx + 0.2, var_per_ch_b, 0.35, color=ACCENT3, alpha=0.85,
                   label=norm_b)
    axes[1, 3].set_title("Per-channel σ² (after norm)", fontsize=9, color=TEXT_CLR)
    axes[1, 3].set_xlabel("Channel", fontsize=8)
    axes[1, 3].legend(fontsize=7, facecolor=CARD_BG, edgecolor=GRID_CLR,
                      labelcolor=TEXT_CLR)

    fig2.tight_layout()
    apply_dark_theme(fig2, axes.ravel())
    st.image(fig_to_img(fig2), use_container_width=True)

    # ── Normalisation axis diagram ─────────────────────────────
    st.markdown("#### Which axes does each method normalise over?")
    fig3, ax_diag = plt.subplots(1, 1, figsize=(10, 2.5), facecolor=DARK_BG)
    ax_diag.set_facecolor(DARK_BG)
    ax_diag.axis("off")

    methods_info = [
        ("Batch Norm",    "N axis\n(per C,H,W)", ACCENT1, "Small ε OK\nBad: small N"),
        ("Layer Norm",    "C,H,W axes\n(per N)", ACCENT2, "Good: small N\nNLP/Transformers"),
        ("Instance Norm", "H,W axes\n(per N,C)", ACCENT3, "Style transfer\nSmall batches"),
        ("Group Norm",    "C/G,H,W\n(per N,G)", ACCENT4, "Detection/Seg\nAny batch size"),
    ]
    for k, (name, axes_lbl, clr, note) in enumerate(methods_info):
        xpos = 0.1 + k * 0.23
        rect = plt.Rectangle((xpos, 0.2), 0.19, 0.6,
                              facecolor=clr + "22", edgecolor=clr, lw=1.5,
                              transform=ax_diag.transAxes)
        ax_diag.add_patch(rect)
        ax_diag.text(xpos + 0.095, 0.85, name, ha="center", va="top",
                     color=clr, fontsize=9, fontweight="bold",
                     transform=ax_diag.transAxes)
        ax_diag.text(xpos + 0.095, 0.58, axes_lbl, ha="center", va="top",
                     color=TEXT_CLR, fontsize=8, transform=ax_diag.transAxes,
                     linespacing=1.5)
        ax_diag.text(xpos + 0.095, 0.26, note, ha="center", va="bottom",
                     color="#6060a0", fontsize=7, transform=ax_diag.transAxes,
                     linespacing=1.4)

    st.image(fig_to_img(fig3), use_container_width=True)

    # ── Info cards ────────────────────────────────────────────
    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown("""
<div class='info-card'>
<b>Batch Norm</b> computes statistics over the <b>batch dimension</b> per channel.
It works best with large batch sizes (≥16). With small batches, estimates of
μ and σ² become noisy, degrading performance.
At inference it uses <b>running statistics</b> collected during training.
</div>""", unsafe_allow_html=True)
    with ic2:
        st.markdown("""
<div class='info-card'>
<b>Layer Norm</b> normalises over the feature dimensions <em>for each sample independently</em>,
making it immune to batch-size effects. This is why it became the default in Transformers
(BERT, GPT, ViT). <b>Instance Norm</b> is the per-channel version used in style transfer,
preserving intra-channel spatial statistics.
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  TAB 3 ── ACTIVATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Activation Functions — Shape, Derivatives, and Dead Neurons")
    st.markdown("""
<div class='info-card'>
Activation functions introduce non-linearity that allows deep networks to learn complex mappings.
The <b>derivative</b> governs gradient flow — flat regions (saturation) cause vanishing gradients.
<b>Dying ReLU</b> occurs when a neuron permanently outputs zero because its weights push it
into the negative half-plane. Modern activations (GELU) are smooth approximations
designed to avoid this while keeping fast computation.
</div>
""", unsafe_allow_html=True)

 
    st.markdown("#### ① Primary Activation")
    act_a_name = st.selectbox("Activation A",
                            list(ACTIVATIONS.keys()), index=0, key="a_a")
    if act_a_name == "Leaky ReLU":
        alpha_a = st.slider("α (Leaky ReLU)", 0.001, 0.5, 0.1, 0.001, key="a_alpha_a")


    st.markdown("#### ② Comparison Activation")
    act_b_name = st.selectbox("Activation B",
                            list(ACTIVATIONS.keys()), index=3, key="a_b")
    if act_b_name == "Leaky ReLU":
        alpha_b = st.slider("α (Leaky ReLU B)", 0.001, 0.5, 0.1, 0.001, key="a_alpha_b")

    st.markdown("#### Dead Neuron Simulator")
    dead_thresh = st.slider("Input shift (dying ReLU demo)",
                            -8.0, 0.0, -2.0, 0.5, key="a_dead")

    # ── Build x axis ──────────────────────────────────────────
    xmin_a, xmax_a = -8, 8
    xs = np.linspace(xmin_a, xmax_a, 800)

    # Resolve functions with current parameters
    fn_map = {
        "ReLU":        lambda x: relu(x),
        "Leaky ReLU":  lambda x: leaky_relu(x, alpha_a if act_a_name == "Leaky ReLU" else 0.1),
        "GELU":        lambda x: gelu(x),
        "Sigmoid":     lambda x: sigmoid(x),
        "Tanh":        lambda x: tanh_fn(x),
    }
    fn_map_b = {
        "ReLU":        lambda x: relu(x),
        "Leaky ReLU":  lambda x: leaky_relu(x, alpha_b if act_b_name == "Leaky ReLU" else 0.1),
        "GELU":        lambda x: gelu(x),
        "Sigmoid":     lambda x: sigmoid(x),
        "Tanh":        lambda x: tanh_fn(x),
    }

    ya  = fn_map[act_a_name](xs)
    yb  = fn_map_b[act_b_name](xs)

    # Numerical derivative
    dx  = xs[1] - xs[0]
    dya = np.gradient(ya, dx)
    dyb = np.gradient(yb, dx)

    # Dead neuron demo: shift inputs by dead_thresh
    xs_dead = xs + dead_thresh
    ya_dead = fn_map["ReLU"](xs_dead)

    # ── Figure ────────────────────────────────────────────────
    n_rows = 3
    fig4, axes4 = plt.subplots(n_rows, 3, figsize=(13, 3.2 * n_rows), facecolor=DARK_BG)
    if n_rows == 2:
        axes4 = np.vstack([axes4, [None, None, None]])

    apply_dark_theme(fig4, [a for a in axes4.ravel() if a is not None])

    _, _, clr_a, formula_a = ACTIVATIONS[act_a_name]
    _, _, clr_b, formula_b = ACTIVATIONS[act_b_name]

    # Row 0: f(x) comparison
    ax_fa = axes4[0, 0]
    ax_fa.plot(xs, ya, color=clr_a, lw=2.5, label=f"{act_a_name}\n{formula_a}")
    ax_fa.axhline(0, color=GRID_CLR, lw=0.8); ax_fa.axvline(0, color=GRID_CLR, lw=0.8)
    ax_fa.set_title(f"A: {act_a_name}  f(x)", fontsize=9, color=TEXT_CLR)
    ax_fa.legend(fontsize=7.5, facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
    ax_fa.set_xlabel("x"); ax_fa.set_ylabel("f(x)")

    ax_fb = axes4[0, 1]
    ax_fb.plot(xs, yb, color=clr_b, lw=2.5, label=f"{act_b_name}\n{formula_b}")
    ax_fb.axhline(0, color=GRID_CLR, lw=0.8); ax_fb.axvline(0, color=GRID_CLR, lw=0.8)
    ax_fb.set_title(f"B: {act_b_name}  f(x)", fontsize=9, color=TEXT_CLR)
    ax_fb.legend(fontsize=7.5, facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
    ax_fb.set_xlabel("x"); ax_fb.set_ylabel("f(x)")

    # Overlay comparison
    ax_ov = axes4[0, 2]
    ax_ov.plot(xs, ya, color=clr_a, lw=2, label=act_a_name)
    ax_ov.plot(xs, yb, color=clr_b, lw=2, linestyle="--", label=act_b_name)
    ax_ov.axhline(0, color=GRID_CLR, lw=0.6); ax_ov.axvline(0, color=GRID_CLR, lw=0.6)
    ax_ov.set_title("Overlay: A vs B", fontsize=9, color=TEXT_CLR)
    ax_ov.legend(fontsize=8, facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
    ax_ov.set_xlabel("x"); ax_ov.set_ylabel("f(x)")

    # Row 1: Distribution transformation
    rng2 = np.random.default_rng(42)
    sample = rng2.normal(0, 1.5, 3000)

    ax_da = axes4[1, 0]
    ax_da.hist(sample, bins=50, color=TEXT_CLR, alpha=0.4, density=True, label=r"$x_i \sim \mathcal{N}(0,\ 1.5^2),\ n=3000$")
    ax_da.hist(fn_map[act_a_name](sample), bins=50, color=clr_a, alpha=0.7,
               density=True, label=f"After {act_a_name}")
    ax_da.set_title("Input → Output distribution (A)", fontsize=9, color=TEXT_CLR)
    ax_da.legend(fontsize=7, facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
    ax_da.set_xlabel("value")

    ax_db = axes4[1, 1]
    ax_db.hist(sample, bins=50, color=TEXT_CLR, alpha=0.4, density=True, label="Input")
    ax_db.hist(fn_map_b[act_b_name](sample), bins=50, color=clr_b, alpha=0.7,
               density=True, label=f"After {act_b_name}")
    ax_db.set_title("Input → Output distribution (B)", fontsize=9, color=TEXT_CLR)
    ax_db.legend(fontsize=7, facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
    ax_db.set_xlabel("value")

    # Dead neuron demo
    ax_dn = axes4[1, 2]
    dead_fraction = (xs_dead < 0).mean()
    dead_x_cutoff = -dead_thresh  # the x value where shifted ReLU wakes up
    ax_dn.axvspan(xmin_a, dead_x_cutoff, color=ACCENT2, alpha=0.15, label=f"Dead zone ({dead_fraction*100:.0f}%)")

    # Plot the two ReLUs on top
    ax_dn.plot(xs, relu(xs),      color=ACCENT3, lw=2, alpha=0.5, label="ReLU (no shift)")
    ax_dn.plot(xs, ya_dead,       color=ACCENT2, lw=2.5, label=f"ReLU (shift={dead_thresh:+.1f})")
    ax_dn.axhline(0, color=GRID_CLR, lw=0.8); ax_dn.axvline(0, color=GRID_CLR, lw=0.8)
    ax_dn.set_title(f"Dying ReLU: {dead_fraction*100:.0f}% dead", fontsize=9, color=TEXT_CLR)
    ax_dn.legend(fontsize=7, facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
    ax_dn.set_xlabel("x"); ax_dn.set_ylabel("ReLU(x + shift)")

    # Row 2: Derivatives
    ax_dfa = axes4[2, 0]
    ax_dfa.plot(xs, dya, color=clr_a, lw=2)
    ax_dfa.axhline(0, color=GRID_CLR, lw=0.8)
    ax_dfa.axhline(1, color=GRID_CLR, lw=0.5, linestyle=":")
    ax_dfa.fill_between(xs, dya, 0, where=(np.abs(dya) < 0.05),
                        color=ACCENT2, alpha=0.3, label="Saturation zone")
    ax_dfa.set_title(f"f′(x)  — {act_a_name}", fontsize=9, color=TEXT_CLR)
    ax_dfa.legend(fontsize=7, facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
    ax_dfa.set_xlabel("x"); ax_dfa.set_ylabel("f′(x)")

    ax_dfb = axes4[2, 1]
    ax_dfb.plot(xs, dyb, color=clr_b, lw=2)
    ax_dfb.axhline(0, color=GRID_CLR, lw=0.8)
    ax_dfb.axhline(1, color=GRID_CLR, lw=0.5, linestyle=":")
    ax_dfb.fill_between(xs, dyb, 0, where=(np.abs(dyb) < 0.05),
                        color=ACCENT2, alpha=0.3, label="Saturation zone")
    ax_dfb.set_title(f"f′(x)  — {act_b_name}", fontsize=9, color=TEXT_CLR)
    ax_dfb.legend(fontsize=7, facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
    ax_dfb.set_xlabel("x"); ax_dfb.set_ylabel("f′(x)")

    # Derivative overlay
    ax_dov = axes4[2, 2]
    ax_dov.plot(xs, dya, color=clr_a, lw=2, label=f"f′  {act_a_name}")
    ax_dov.plot(xs, dyb, color=clr_b, lw=2, linestyle="--", label=f"f′  {act_b_name}")
    ax_dov.axhline(0, color=GRID_CLR, lw=0.8)
    ax_dov.set_title("Derivative overlay", fontsize=9, color=TEXT_CLR)
    ax_dov.legend(fontsize=8, facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
    ax_dov.set_xlabel("x"); ax_dov.set_ylabel("f′(x)")

    fig4.tight_layout(pad=1.5)
    apply_dark_theme(fig4, [a for a in axes4.ravel() if a is not None])
    st.image(fig_to_img(fig4), use_container_width=True)