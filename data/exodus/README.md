# Exodus spatial-field data

This directory contains provenance and integrity metadata for the condition-specific Exodus FEM databases used by the spatial field-evolution workflow.

The full `.e` files are not stored in normal Git history because each exceeds GitHub's 100 MB per-file limit.

The authoritative Exodus databases are:

- `AS.e`
- `SR350.e`
- `SR425.e`
- `SR525.e`
- `SA1150.e`

Their SHA-256 hashes, sizes, timestep counts, and selected field-evolution indices are recorded in `EXODUS_DATA_MANIFEST.csv`.

The complete `Exodus_files.zip` archive will be distributed separately as a GitHub Release asset.
