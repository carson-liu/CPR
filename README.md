# Circular Phase Representation and Geometry-Aware Optimization for Ptychographic Image Reconstruction

Official Python implementation of the paper:

> **Circular Phase Representation and Geometry-Aware Optimization for Ptychographic Image Reconstruction**

---

## Overview

Ptychography is a powerful lensless computational imaging technique that reconstructs the complex-valued transmission function of a specimen from a set of far-field diffraction patterns collected at overlapping illumination positions. A central challenge is the reliable estimation of the **phase** of the complex object field, which is inherently wrapped in the interval $(-\pi, \pi]$.

This work introduces two core contributions:

1. **Circular Phase Representation (CPR):** Instead of parameterizing the phase directly as a real scalar, we represent each phase value as a point on the unit circle $S^1$ via the complex exponential $e^{i\phi}$. This intrinsic representation eliminates phase-wrapping discontinuities and enables the optimizer to move freely along the phase manifold without artificial boundaries.

2. **Geometry-Aware Optimization:** We exploit the Riemannian structure of the phase manifold and incorporate geometry-aware gradient updates (Riemannian gradient descent / natural gradient) that respect the curvature of $S^1$. This leads to faster convergence and better-conditioned optimization landscapes compared to naive Euclidean updates applied to wrapped phase variables.

Together, CPR and geometry-aware optimization achieve state-of-the-art reconstruction quality in both noiseless and noisy settings across synthetic and real-world ptychographic datasets.

---

## Method

### Circular Phase Representation

For a complex object $O = A \cdot e^{i\Phi}$, the phase $\Phi$ is traditionally stored as a real matrix subject to wrapping. CPR instead maintains a **unit-modulus complex matrix** $Z = e^{i\Phi}$ as the phase variable, so that the constraint $|Z_{mn}| = 1$ is encoded in the parameterization itself. Amplitude $A$ is kept as a separate non-negative variable.

### Geometry-Aware Gradient Updates

Given a loss $\mathcal{L}$ (e.g., amplitude-based or intensity-based least squares), the Riemannian gradient with respect to $Z$ on the manifold $\mathcal{M} = (S^1)^{M \times N}$ is:

$$\nabla_{\mathcal{M}} \mathcal{L} = \nabla_{\mathbb{C}} \mathcal{L} - \text{Re}\!\left(\nabla_{\mathbb{C}} \mathcal{L} \odot \bar{Z}\right) \odot Z$$

The retraction back onto the manifold after a gradient step is simply element-wise normalization:

$$Z \leftarrow \frac{Z - \alpha \, \nabla_{\mathcal{M}} \mathcal{L}}{|Z - \alpha \, \nabla_{\mathcal{M}} \mathcal{L}|}$$

This ensures that phase variables always remain on the unit circle without projection steps or penalty terms.

---

## Requirements

- Python >= 3.8
- NumPy
- SciPy
- Matplotlib
- (Optional) PyTorch >= 1.10 — for GPU-accelerated variants

Install dependencies:

```bash
pip install numpy scipy matplotlib
# Optional GPU support
pip install torch
```

---

## Repository Structure

```
CPR/
├── README.md
├── cpr/
│   ├── __init__.py
│   ├── representation.py   # Circular Phase Representation utilities
│   ├── optimizer.py        # Geometry-aware gradient update rules
│   └── loss.py             # Ptychographic forward model and loss functions
├── data/
│   └── ...                 # Synthetic / example datasets
├── experiments/
│   ├── run_synthetic.py    # Reproduce synthetic experiments
│   └── run_realdata.py     # Run on real ptychographic data
└── tests/
    └── test_cpr.py
```

---

## Usage

### Quickstart

```python
from cpr.representation import CircularPhase
from cpr.optimizer import GeometryAwareOptimizer
from cpr.loss import PtychographicLoss

# Initialize the circular phase object
obj = CircularPhase(shape=(256, 256))

# Define the forward model / loss
loss_fn = PtychographicLoss(diffraction_patterns, probe, scan_positions)

# Run geometry-aware optimization
optimizer = GeometryAwareOptimizer(step_size=1e-2, n_iter=500)
reconstructed_object = optimizer.run(obj, loss_fn)
```

### Reproducing Paper Results

```bash
# Synthetic benchmark
python experiments/run_synthetic.py --snr 30 --n_iter 1000 --method CPR

# Real dataset
python experiments/run_realdata.py --data_path data/example.npz --n_iter 500
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{liu2024cpr,
  title   = {Circular Phase Representation and Geometry-Aware Optimization
             for Ptychographic Image Reconstruction},
  author  = {Liu, Carson and others},
  journal = {arXiv preprint},
  year    = {2024}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
