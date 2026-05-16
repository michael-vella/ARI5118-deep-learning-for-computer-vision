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
ACCENT5 = "#888899" # grey

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
  ARI5118 · Topic 2 · Author: michael.vella.20@um.edu.mt
</p>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "🟦  Pooling Explorer",
    "📊  Batch Normalisation Lab",
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
    st.markdown("### Batch Normalisation")
    st.markdown("""
<div class='info-card'>
<b>Batch Normalisation</b> stabilises training by normalising each channel's activations
across the batch dimension. For a tensor of shape <b>(N, C, H, W)</b>, it computes the
mean μ and variance σ² over the <b>N, H, W</b> axes — independently per channel C —
then applies <b>x̂ = (x − μ) / √(σ² + ε)</b>. The learnable parameters γ (scale)
and β (shift) restore expressive power after normalisation. Below, we work with a
<b>single channel's values across the batch</b> — the simplest case — so you can trace
each number through every step.
</div>
""", unsafe_allow_html=True)
 
    
    st.markdown("#### Input Batch")
    preset = st.selectbox(
        "Preset distribution",
        ["Standard Normal", "Custom sliders", "High Variance", "Skewed", "Near-Zero"],
        key="n_preset"
    )

    N = 4
    st.metric("Batch size N", N)

    rng_default = np.random.default_rng(7)
    if preset == "Custom sliders":
        st.markdown("**Set each value manually:**")
        defaults = np.round(rng_default.uniform(-5, 5, N), 1).tolist()
        raw = np.array([
            st.slider(f"x_{i+1}", -10.0, 10.0, defaults[i], 0.1, key=f"n_v{i}")
            for i in range(N)
        ], dtype=np.float32)
    else:
        rng2 = np.random.default_rng(7)
        if preset == "Standard Normal":
            raw = rng2.normal(0, 1, N).astype(np.float32)
        elif preset == "High Variance":
            raw = rng2.normal(0, 5, N).astype(np.float32)
        elif preset == "Skewed":
            raw = (rng2.exponential(2, N) - 1).astype(np.float32)
        else:  # Near-Zero
            raw = rng2.normal(0, 0.05, N).astype(np.float32)
        st.info(f"Values: {np.round(raw, 2).tolist()}")
 
    
    st.markdown("#### Parameters")
    eps = st.select_slider("ε (epsilon)",
                            options=[1e-8, 1e-5, 1e-3, 0.01, 0.1, 0.5, 1.0],
                            value=1e-5, key="n_eps",
                            format_func=lambda v: f"{v:.0e}" if v < 0.01 else str(v),
                            help="Added to variance before dividing — prevents ÷0")
    gamma = st.slider("γ (scale)",  0.1, 3.0, 1.0, 0.1, key="n_gamma",
                        help="Learnable scale applied after normalisation")
    beta  = st.slider("β (shift)", -3.0, 3.0, 0.0, 0.1, key="n_beta",
                        help="Learnable shift applied after normalisation")
    
    highlight = st.selectbox("Highlight one value",
                                [f"x_{i+1}" for i in range(N)],
                                key="n_hi",
                                help="Trace this value through all four steps")
    hi = int(highlight.split("_")[1]) - 1  # 0-based index
 
 
    # ── Batch Norm — step by step ──────────────────────────────
    mu     = raw.mean()                        # scalar
    var    = raw.var()                         # scalar (biased estimator)
    x_norm = (raw - mu) / np.sqrt(var + eps)  # normalised: x̂
    x_out  = gamma * x_norm + beta            # scaled and shifted: y
 
    # ── Metrics strip ─────────────────────────────────────────
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("μ (mean)",      f"{mu:.3f}")
    m2.metric("σ² (variance)", f"{var:.3f}")
    m3.metric("σ (std)",       f"{np.sqrt(var):.3f}")
    m4.metric("ε",             f"{eps:.0e}")
    m5.metric("γ",             f"{gamma:.1f}")
    m6.metric("β",             f"{beta:.1f}")
 
    # ── Figure: 4-step bar charts ─────────────────────────────
    labels = [f"x_{i+1}" for i in range(N)]
    x_pos  = np.arange(N)
 
    def annotate_bars_local(ax, values, fontsize=8):
        """Write the numeric value above/below each bar."""
        y_range = max(ax.get_ylim()[1] - ax.get_ylim()[0], 0.1)
        offset  = y_range * 0.04
        for i, v in enumerate(values):
            ax.text(i, v + offset if v >= 0 else v - offset * 2,
                    f"{v:.2f}", ha="center",
                    va="bottom" if v >= 0 else "top",
                    color="white", fontsize=fontsize, fontweight="bold")
 
    def draw_step(ax, values, title, formula, base_color, step_num):
        """One step as an annotated bar chart. Pink bar = highlighted value."""
        colors     = [base_color] * N
        colors[hi] = "#ff6584"
        bars = ax.bar(x_pos, values, color=colors, width=0.6,
                      edgecolor=GRID_CLR, linewidth=0.8, zorder=3)
        ax.axhline(0, color=TEXT_CLR, lw=0.8, linestyle="-", alpha=0.4)
 
        if step_num == 1:
            ax.axhline(mu, color=ACCENT4, lw=1.5, linestyle="--",
                       label=f"μ = {mu:.2f}", zorder=4)
            ax.legend(fontsize=7.5, facecolor=CARD_BG, edgecolor=GRID_CLR,
                      labelcolor=TEXT_CLR, loc="upper right")
 
        if step_num == 3:
            ax.axhspan(-1, 1, color=ACCENT3, alpha=0.08, zorder=0, label="±1σ")
            ax.axhline( 1, color=ACCENT3, lw=0.8, linestyle=":", alpha=0.6)
            ax.axhline(-1, color=ACCENT3, lw=0.8, linestyle=":", alpha=0.6)
            ax.legend(fontsize=7.5, facecolor=CARD_BG, edgecolor=GRID_CLR,
                      labelcolor=TEXT_CLR)
 
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=8, color=TEXT_CLR)
        ax.set_title(f"Step {step_num}: {title}\n{formula}",
                     fontsize=8.5, color=TEXT_CLR, pad=8)
        annotate_bars_local(ax, values)
        bars[hi].set_edgecolor("white")
        bars[hi].set_linewidth(2.5)
 
    fig2, axes2 = plt.subplots(1, 4, figsize=(14, 4.5), facecolor=DARK_BG)
    apply_dark_theme(fig2, axes2)
 
    draw_step(axes2[0], raw,       "Raw input",      "x",
              ACCENT5, 1)
    draw_step(axes2[1], raw - mu,  "Subtract mean",  f"x − μ      μ={mu:.2f}",
              ACCENT1,  2)
    draw_step(axes2[2], x_norm,    "Divide by std",  f"(x−μ)/√(σ²+ε)   σ={np.sqrt(var):.2f}",
              ACCENT3, 3)
    draw_step(axes2[3], x_out,     "Scale & shift",  f"γ·x̂ + β      γ={gamma}, β={beta}",
              ACCENT4, 4)
    
    fig2.suptitle(
        f"{highlight}:   {raw[hi]:.2f}  →  {raw[hi]-mu:.2f}"
        f"  →  {x_norm[hi]:.2f}  →  {x_out[hi]:.2f}",
        color="#ff6584", fontsize=10, y=1.02
    )
    fig2.tight_layout(pad=1.5)
    apply_dark_theme(fig2, axes2)
    st.image(fig_to_img(fig2), use_container_width=True)
 
    # ── Numeric trace cards ────────────────────────────────────
    st.markdown(f"#### Tracing **{highlight}** = `{raw[hi]:.4f}` through each step")
    t1, t2, t3, t4 = st.columns(4)
 
    t1.markdown(f"""<div class='info-card'>
<b>① Raw</b><br>
<code>{raw[hi]:.4f}</code>
</div>""", unsafe_allow_html=True)
 
    t2.markdown(f"""<div class='info-card'>
<b>② Subtract μ</b><br>
<code>{raw[hi]:.4f} − {mu:.4f}</code><br>
= <code>{raw[hi] - mu:.4f}</code>
</div>""", unsafe_allow_html=True)
 
    t3.markdown(f"""<div class='info-card'>
<b>③ Divide by √(σ²+ε)</b><br>
<code>{raw[hi]-mu:.4f} / √({var:.4f} + {eps:.0e})</code><br>
= <code>{raw[hi]-mu:.4f} / {np.sqrt(var+eps):.4f}</code><br>
= <code>{x_norm[hi]:.4f}</code>
</div>""", unsafe_allow_html=True)
 
    t4.markdown(f"""<div class='info-card'>
<b>④ Scale & shift</b><br>
<code>{gamma} × {x_norm[hi]:.4f} + {beta}</code><br>
= <code>{x_out[hi]:.4f}</code>
</div>""", unsafe_allow_html=True)
 
    # ── Epsilon explainer ──────────────────────────────────────
    st.markdown("#### What does ε actually do?")
    st.markdown(
        "When all batch values are nearly identical, σ²≈0 and dividing by √σ² "
        "would cause a numerical explosion. ε sets a safe floor on the denominator."
    )
 
    eps_vals      = np.logspace(-8, 0, 300)
    near_zero_var = 1e-6
    denom         = np.sqrt(near_zero_var + eps_vals)
 
    fig4, ax4 = plt.subplots(figsize=(8, 3), facecolor=DARK_BG)
    apply_dark_theme(fig4, [ax4])
    ax4.semilogx(eps_vals, denom, color=ACCENT3, lw=2,
                 label=r"$\sqrt{\sigma^2 + \varepsilon}$")
    ax4.axhline(np.sqrt(near_zero_var), color=ACCENT1, lw=1.5, linestyle="--",
                label=f"√σ² alone ← dangerous (very close to 0)")
    ax4.set_xlabel("ε  (log scale)", fontsize=9)
    ax4.set_ylabel(f"Denominator Result", fontsize=9)
    ax4.set_title("Effect of ε on denominator stability  (σ²=10⁻⁶)",
                  fontsize=9, color=TEXT_CLR)
    ax4.legend(fontsize=8, facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
    fig4.tight_layout()
    apply_dark_theme(fig4, [ax4])
    st.image(fig_to_img(fig4), use_container_width=True)


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