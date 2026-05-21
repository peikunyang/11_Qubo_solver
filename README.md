# Comparative Evaluation of PyTorch, JAX, SciPy, and Neal for Dense QUBO Optimization at Large Matrix Scales

This repository contains the source code and benchmarking framework for the paper:

**Comparative Evaluation of PyTorch, JAX, SciPy, and Neal for Dense QUBO Optimization at Large Matrix Scales**  
**Author**: Pei-Kun Yang  
📧 E-mail: [peikun@isu.edu.tw](mailto:peikun@isu.edu.tw)  
🆔 ORCID: [0000-0003-1840-6204](https://orcid.org/0000-0003-1840-6204)

---

## 🧩 Overview

This project evaluates several software-based solvers for **Quadratic Unconstrained Binary Optimization (QUBO)**, focusing on performance, solution quality, and scalability. The solvers tested include:

- **Neal** (simulated annealing)
- **PyTorch (CPU)**
- **PyTorch (GPU)**
- **JAX**
- **SciPy**

## 📦 Installation

Install all required Python packages:

```bash
pip install numpy torch scipy optax jax jaxlib pyqubo dwave-ocean-sdk


.
├── 1_gen_coe         # Generate QUBO matrices (Q)
├── 2_neal            # Solve QUBO using Neal (CPU-based simulated annealing)
├── 3_pytorch_cpu     # Solve QUBO using PyTorch on CPU
├── 4_pytorch_gpu     # Solve QUBO using PyTorch on GPU
├── 5_JAX_gpu         # Solve QUBO using JAX (GPU accelerated)
├── 6_SciPy           # Solve QUBO using SciPy (L-BFGS-B optimizer)
