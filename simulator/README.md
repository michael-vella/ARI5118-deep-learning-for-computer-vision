# Pooling · Normalisation · Activation Functions — Interactive Simulator

**ARI5118 Deep Learning for Computer Vision — Topic 2**

An interactive Streamlit application that lets you explore, visualise, and
compare the three foundational building blocks covered in Topic 2.

The simulator was deployed and is being hosted on Streamlit. For access, visit https://mvella-uom-ari5118-deep-learning.streamlit.app/.

## Features

| Tab | What you can explore |
|---|---|
| 🟦 **Pooling Explorer** | Max and Average pooling on synthetic feature maps. Adjust kernel size and stride, see output shapes update in real time, compare two pooling configurations side-by-side. |
| 📊 **Normalisation Lab** | An in-depth guide on Batch Norm works, allowing you tune different parameters for different experimentation scenarios. |
| ⚡ **Activation Functions** | 5 major activations (ReLU, Leaky ReLU, GELU, Sigmoid, Tanh). View f(x), f′(x), distribution transformations, dying-neuron simulation. |

## Requirements

- Python ≥ 3.12
- No GPU required (CPU-only)

## Setup & Launch

```bash
# 1. Clone / download the repository
# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the simulator
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

## File Structure

```
simulator/
├── app.py           # Main Streamlit application
├── requirements.txt # Python dependencies
└── README.md        # This file
```