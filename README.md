# LPBF-SDSS-6Ni-FEM

Reproducible finite-element analysis and experimental–numerical comparison for LPBF-fabricated 6 wt.% Ni-modified 2507 super duplex stainless steel.

## Scope

This repository contains:

- experimental tensile mean curves;
- condition-specific FEM tensile histories;
- experimental–FEM comparison;
- spatial stress/plastic-strain field evolution;
- strain-decomposition analysis;
- provenance and SHA-256 integrity records for large Exodus outputs.

Conditions:

`AS`, `SR350`, `SR425`, `SR525`, and `SA1150`.

## Repository structure

```text
.
├── data/
│   ├── experimental/
│   ├── fem_histories/
│   └── exodus/
├── scripts/
│   ├── SDSS_exp_vs_FEM_current_conditions.py
│   ├── field_evolution.py
│   └── strain_decomposition.py
├── results/
│   ├── exp_vs_FEM/
│   ├── field_evolution/
│   └── strain_decomposition/
├── requirements.txt
└── LICENSE
