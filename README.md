# SBTS Advanced: Levy Processes & Schrödinger Bridge for Time Series Generation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![École Polytechnique](https://img.shields.io/badge/Institution-École%20Polytechnique-red)](https://www.polytechnique.edu/)

> **Projet Scientifique Collectif (PSC)**  
> **Institution:** École Polytechnique (IP Paris)  
> **Academic Year:** 2024-2025
---

## 👥 Authors & Supervision

This project was developed by **Lizhan HONG** (*École Polytechnique*) under the supervision and guidance of **Prof. Huyên PHAM**.

> **Context:** Performed within the framework of the *Projet Scientifique Collectif (PSC)*, 2024-2025.
---

## 📖 Overview

**SBTS Advanced** is a research-oriented implementation focusing on generative modeling for financial time series using **Schrödinger Bridge (SB)** frameworks. Unlike standard diffusion models that rely on Brownian motion, this project extends the methodology to **Levy Processes** and **Jump-Diffusion SDEs**, allowing for the generation of realistic financial data with heavy tails and discontinuous jumps.

This repository implements a complete pipeline from jump detection in raw market data to the calibration of stochastic volatility models and the generation of synthetic trajectories via Entropic Optimal Transport.

### Key Mathematical Framework
The core dynamics are modeled by a Jump-Diffusion Stochastic Differential Equation (SDE):

$$ dX_t = \mu(X_t, t)dt + \sigma(X_t, t)dW_t + dJ_t $$

Where $W_t$ is a Brownian motion and $J_t$ represents the jump component (Poisson/Levy), handled via specific solvers and calibration techniques.

---

## 🗂 Project Structure

The codebase is organized into modular components for scalability and research experimentation:

```plaintext
sbts_advanced/
├── core/                   # Core algorithms and SDE solvers
│   ├── bandwidth.py        # Adaptive CV Bandwidth selection for density estimation
│   ├── lightsb.py          # LightSB implementation (GMM Parameterization)
│   ├── reference.py        # Reference measures (Levy Processes, Stoch Vol)
│   └── solver.py           # Euler-Maruyama solver with Jump-Diffusion support
├── models/                 # Financial modeling and Calibration
│   ├── jumps.py            # Jump Detection algorithms (e.g., Ait-Sahalia, Lee-Mykland)
│   └── calibration.py      # Volatility Surface & Model Calibration
├── utils/                  # Utilities for analysis
│   ├── metrics.py          # Evaluation metrics (Wasserstein Distance, MSE, discriminative score)
│   └── visualization.py    # Plotting routines for trajectories and distributions
└── main.py                 # Main entry point for the training/generation pipeline
```

## ✨ Key Features

This project integrates advanced stochastic calculus with modern generative modeling:

*   🧮 **Advanced SDE Solvers**
    *   Implementation of **Euler-Maruyama schemes** specifically adapted for jump-diffusion processes.
*   📉 **Robust Jump Detection**
    *   Algorithms designed to identify discontinuities in high-frequency or daily financial data with high precision.
*   ⚡ **LightSB Integration**
    *   Efficient **Schrödinger Bridge** training utilizing Gaussian Mixture Model (GMM) parameterization for fast convergence.
*   🎯 **Adaptive Bandwidth Selection**
    *   **Cross-validation** based bandwidth selection for Kernel Density Estimation (KDE) during the diffusion process.
*   📊 **Quantitative Evaluation**
    *   Comprehensive suite of metrics (e.g., Wasserstein distance) to rigorously evaluate the fidelity of generated time series against historical references.

---

## 🚀 Getting Started

### 🛠 Prerequisites
*   **Python 3.10+** environment.
*   Scientific computing stack: `numpy`, `scipy`, `torch`, `matplotlib`.
*   See more on `requirements.txt`.

### 📥 Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/ApolloHong/sbts_advanced.git
    cd sbts_advanced
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

### 🏃 Usage

Execute the full pipeline (Data Calibration $\rightarrow$ Jump Detection $\rightarrow$ Generation):

```bash
python main.py