# LPBF-SDSS-6Ni-FEM

Reproducible finite-element analysis and experimental–numerical comparison for LPBF-fabricated 6 wt.% Ni-modified 2507 super duplex stainless steel.

## Scope

This repository contains:

- experimental tensile mean curves;
- condition-specific FEM tensile histories;
- experimental–FEM comparison;
- spatial stress/plastic-strain field-evolution analysis;
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
```

## Installation

```bash
git clone https://github.com/mengedagnaw/LPBF-SDSS-6Ni-FEM.git
cd LPBF-SDSS-6Ni-FEM

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Analysis workflows

### Experimental–FEM comparison

```bash
python scripts/SDSS_exp_vs_FEM_current_conditions.py
```

### Strain decomposition

```bash
python scripts/strain_decomposition.py
```

### Spatial field evolution

The field-evolution workflow additionally requires the condition-specific Exodus databases.

After reconstructing and extracting the Exodus archive, point the workflow to the directory containing the required production data:

```bash
export SDSS_PRODUCTION_RUN=/path/to/production_data
python scripts/field_evolution.py
```

## Large Exodus data

The condition-specific Exodus spatial-field databases are distributed through
the `v1.0.0` GitHub Release as three split assets:

- `Exodus_files.zip.part00`
- `Exodus_files.zip.part01`
- `Exodus_files.zip.part02`

Reconstruct the original archive on Linux/macOS:

```bash
cat Exodus_files.zip.part00 \
    Exodus_files.zip.part01 \
    Exodus_files.zip.part02 \
    > Exodus_files.zip
```

Verify the reconstructed archive:

```bash
sha256sum Exodus_files.zip
```

Expected SHA-256:

`ed78fc7f994bcad5f49e5dd74e5f1a1d95a4cd6a53b05f40321b92c1e1b1932b`

The reconstructed archive contains:

- `AS.e`
- `SR350.e`
- `SR425.e`
- `SR525.e`
- `SA1150.e`

Individual Exodus-file hashes are documented in:

`data/exodus/EXODUS_DATA_MANIFEST.csv`

The release also includes `SHA256SUMS.txt` for integrity verification of the three split assets.

## Reproducibility boundary

The repository separates source inputs, executable post-processing scripts, derived numerical outputs, figures, and provenance metadata.

The Exodus databases are numerical FEM outputs. Their preservation enables reproduction of the reported spatial-field post-processing but does not, by itself, establish experimental validation, constitutive-model uniqueness, or predictive validity outside the evaluated conditions.

## License

The source code is released under the MIT License. See `LICENSE`.
