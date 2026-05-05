# E2D2: Explanation-Based Drift Detection for Unsupervised NIDS

This repository contains the reference implementation, experiments, and results for the paper:

> **E2D2: A Modular Framework for Explanation-Based Drift Detection in Unsupervised Network Intrusion Detection**
> Beny Nugraha and Thomas Bauschert
> *TX4Nets workshop, IFIP Networking 2026*, Lugano, Switzerland, 24–27 May 2026.

E2D2 detects concept drift in unsupervised intrusion-detection systems (IDS) by monitoring how the *explanations* produced for an IDS's anomaly scores evolve over time, rather than monitoring the input distribution or the score distribution alone. From per-window explanation fingerprints, E2D2 computes a continuous **Explanation Drift Intensity (XDI)** signal, raises threshold-based alarms, and returns compact **1–2 feature drift summaries** to help analysts interpret each alarm.

The framework is modular along four design axes — IDS model, attribution method, population view, and aggregation strategy — yielding 48 configurations evaluated across five network-traffic benchmarks and four drift scenarios.

## Repository contents

```
.
├── e2d2.ipynb              # Main notebook — reproduces all paper results
├── requirements.txt        # Pinned Python dependencies
├── data/
│   └── README.md           # How to obtain and prepare the five datasets
├── E2D2_outputs/           # Created at runtime (figures, tables, CSV exports)
├── LICENSE
└── README.md               # This file
```

The notebook is organized into named sections that map directly to the paper:

| Notebook section            | Purpose                                                       | Paper reference   |
| --------------------------- | ------------------------------------------------------------- | ----------------- |
| Cell 0.1 – 0.2              | Environment setup, dataset loading, leakage-safe preprocessing | §IV-A             |
| Cell 1.1 – 1.2              | VAE / AE backbone definitions and training                    | §III-A, §IV-A     |
| Cell 2.1                    | Four attribution methods and MAD normalization                | §III-B, Eq. (1)   |
| Cell 3.1                    | Sliding windows, fingerprints, XDI engine                     | §III-C, Eqs. (2)–(5) |
| Cell 4.1 – 4.2              | Drift-scenario generators and baseline detectors              | §IV-A (Table I), §IV-B |
| Cell 5.1 – 5.2              | Full experiment matrix and metrics                            | §IV-B             |
| Cell 6.1 – 6.5              | RQ1 – RQ4 analyses + computational-cost comparison            | §V                |
| Cell 7.1 – 7.2              | Publication-ready tables/figures and alarm-filter sweep       | Tables II–V, §V-C |

## Quick start

```bash
git clone https://github.com/DLTeamTUC/E2D2.git
cd E2D2

# Recommended: create an isolated environment
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# Place the five preprocessed datasets under the layout described in
# data/README.md, then launch the notebook:
jupyter notebook e2d2.ipynb
```

Run the cells in order. Each cell prints a short progress summary and a `Cell X.Y complete.` line when it finishes. The five tables and nine figures from the paper are produced by Cells 6.1 – 7.2; final exports (CSV, LaTeX, PNG, PDF) are written to `E2D2_outputs/`.

## Datasets

The notebook expects five preprocessed CSV files corresponding to the five benchmarks evaluated in the paper:

* CIC-DDoS2019 NTP, Portmap, and Syn subsets
* CICIoT2023
* 5GNIDD

The original datasets are publicly available from their respective authors. Because each benchmark requires non-trivial preprocessing (column dropping, balancing, and merging), the exact filenames and directory layout the notebook expects are documented separately. **Please see [`data/README.md`](data/README.md) before running the notebook.**

## Hardware and runtime

The paper results were produced on:

* **GPU**: NVIDIA GeForce RTX 3060 (12 GB)
* **CUDA / cuDNN**: CUDA 12.1
* **OS**: Linux

Approximate end-to-end runtime on this machine: about **42 minutes** for the full experiment matrix in Cell 5.1 (the dominant cost is computing Integrated Gradients and SmoothGrad attributions, since each requires 50 gradient evaluations per sample). The remaining cells together add a few minutes.

A CUDA-capable GPU is strongly recommended. The notebook works on CPU, but Integrated Gradients and SmoothGrad become prohibitively slow.

## Reproducibility

The notebook seeds Python's NumPy, PyTorch (CPU and CUDA), and XGBoost using a single `RANDOM_SEED = 42`. Drift-scenario feature selection uses fixed per-scenario seed offsets, so the sampled drift features are deterministic across runs.

The notebook also enables PyTorch's deterministic cuDNN mode (`torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`) so that successive runs on the same machine produce the same numbers.

**Cross-machine reproducibility caveat**: bit-exact reproduction of the paper's specific numerical values additionally requires the same hardware and library stack. Small numerical deviations (typically < 5%) are expected on different GPUs, CUDA/cuDNN versions, or PyTorch builds. The qualitative claims in the paper — the dominance of the population axis, the best-composition `VAE × Integrated Gradients × all × max`, and the gain from alarm filtering — reproduce reliably across runs.

## Configuration

All hyperparameters are centralized in `E2D2_CONFIG` at the top of Cell 0.1. The settings used in the paper:

| Parameter                | Value           | Reference  |
| ------------------------ | --------------- | ---------- |
| Window size `W`          | 500             | §III-C     |
| Stride `S`               | 250             | §III-C     |
| Baseline windows `B`     | 5               | §III-C     |
| Top-p fraction `p`       | 0.25            | Eq. (3)    |
| Threshold multiplier `κ` | 2.0             | Eq. (5)    |
| MAD floor `m_min`        | 0.05            | §III-C     |
| Z-score clip             | ±10             | §III-C     |
| Drift-explanation cutoff | 0.80            | §III-C     |
| Encoder hidden dims      | [64, 32]        | §IV-A      |
| Latent dim               | 16              | §IV-A      |
| Training                 | 50 epochs, batch 256, lr 10⁻³, patience 10 | §IV-A      |
| Feature selection        | Top 20 by 0.6·SHAP + 0.4·ANOVA | §IV-A      |
| Attribution methods (4)  | raw_gradient, input_x_gradient, integrated_gradients, smoothgrad | §III-B     |
| Population views (3)     | all, ben, sus   | §III-C     |
| Aggregations (2)         | mean_top_p, max | Eqs. (3)–(4) |

These values are kept fixed across all datasets and scenarios. Per-dataset tuning is intentionally avoided.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{nugraha2026e2d2,
  author    = {Beny Nugraha and Thomas Bauschert},
  title     = {{E2D2}: A Modular Framework for Explanation-Based Drift Detection in
               Unsupervised Network Intrusion Detection},
  booktitle = {Proceedings of the IFIP Networking 2026 Workshops (TX4Nets)},
  year      = {2026},
  address   = {Lugano, Switzerland},
  month     = {May},
  publisher = {IFIP},
  isbn      = {978-3-903176-82-9}
}
```

## Acknowledgement

This work was performed in the framework of the **SUSTAINET-Advance** project, funded by the German Federal Ministry of Research, Technology, and Space (BMFTR), grant **16KIS2280**.

## License

This repository is released under the [Apache License 2.0](LICENSE). See the `LICENSE` file for the full terms.

## Contact

Questions, issues, and pull requests are welcome via [GitHub Issues](https://github.com/DLTeamTUC/E2D2/issues).

For other inquiries:

* Beny Nugraha — `beny.nugraha@etit.tu-chemnitz.de`
* Thomas Bauschert — `thomas.bauschert@etit.tu-chemnitz.de`

Chair of Communication Networks, Chemnitz University of Technology, Germany.
