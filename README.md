# CASP: Covariance-Aware Simplex Projection

This repository accompanies the manuscript *Covariance-Aware Simplex Projection for Cardinality-Constrained Portfolio Optimization*. It provides the full experimental pipeline, figures, and compiled manuscript.

## Overview
- Problem: repair infeasible portfolios under cardinality and box constraints by projecting onto the feasible simplex while respecting the covariance geometry.
- Method: CASP replaces Euclidean projection with a covariance-weighted (Omega-metric) projection; RA-CASP further couples return selection and projection.
- Empirical result (S&P 500, 2020–2024): CASP-Basic lowers portfolio variance by 15.7% relative to Euclidean repair. A selection-only variant (VolNorm+Euc) yields 14.3%, isolating an incremental ~1.4% variance reduction attributable to the covariance-weighted projection.

## Environment
```bash
pip install -r requirements.txt
```
All experiments use only the included data. If desired, refresh the dataset:
```bash
python3 src/download_data.py
```

## Reproducing Results
Run the end-to-end experiments (ablation, out-of-sample validation, statistical testing):
```bash
python3 src/analysis.py
```
Generate publication figures:
```bash
python3 src/create_figures.py
```
Build the paper PDF:
```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error CASP_Paper_TwoColumn.tex
pdflatex -interaction=nonstopmode -halt-on-error CASP_Paper_TwoColumn.tex
```

## Outputs
- `results/ablation_summary.csv`, `results/oos_summary.csv`, `results/walk_forward_oos_summary.csv` plus raw distributions (`*.csv`)
- `paper/figures/*.pdf`
- `paper/CASP_Paper_TwoColumn.pdf`

## Citation
```bibtex
@misc{iliopoulos2025casp,
  title        = {Covariance-Aware Simplex Projection for Cardinality-Constrained Portfolio Optimization},
  author       = {Iliopoulos, Nikolaos},
  year         = {2025},
  eprint       = {2512.19986},
  archivePrefix= {arXiv},
  primaryClass = {q-fin.PM},
  url          = {https://arxiv.org/abs/2512.19986}
}
```

## License
MIT License
