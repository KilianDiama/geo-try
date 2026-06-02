# VectorMonolith — Deterministic Idea Clustering Engine

VectorMonolith is a fully deterministic, monolithic, vector‑based pipeline designed to encode text, deform its geometry based on a query, cluster ideas with a reinforced deterministic KMeans, score them using multi‑criteria metrics, and synthesize pivot‑aware summaries.

This engine is built for reproducibility, auditability, and scientific robustness.  
Zero randomness. Zero hidden state. Zero external ML dependencies.

---

## 🚀 Features

- **Deterministic text encoding** using n‑gram hashing + orthonormal QR projection  
- **Query‑aware geometric deformation** (SLERP or linear interpolation)  
- **Deterministic KMeans** with lexsort init, farthest‑first expansion, and empty‑cluster recovery  
- **Multi‑criteria scoring** (distance, compactness, separation, balance, size)  
- **Pivot‑aware synthesis** for human‑readable cluster summaries  
- **Fully vectorized** with NumPy  
- **Reproducible end‑to‑end** thanks to strict invariants and controlled seeds

---

## 📦 Installation

```bash
pip install numpy
