# Exodus spatial-field data

These five Exodus databases are the authoritative spatial FEM outputs used by
`field_evolution.py`.

## Preservation decision

Preserve all five:

- `AS.e`
- `SR350.e`
- `SR425.e`
- `SR525.e`
- `SA1150.e`

They are not duplicates. They use the same FE mesh topology and variable schema,
but contain different condition-specific time histories and spatial fields.

The workflow uses element fields including `equivalent_plastic_strain`,
`strain_xx`, `stress_xx`, and `von_mises_stress`, plus nodal displacement/
reaction fields and global stress-strain/gauge histories.

All selected field-evolution indices were verified to exist and contain finite
stress and plastic-strain data.

## Recommended GitHub handling

Do not commit the `.e` files to ordinary Git history. Store `Exodus_files.zip`
as a GitHub Release asset or in a research-data repository.

Suggested repository layout:

```text
data/
  exodus/
    EXODUS_DATA_MANIFEST.csv
    README_EXODUS_DATA.md
scripts/
  field_evolution.py
results/
  field_evolution/
```

After downloading and extracting the release asset, point the workflow to the
data directory:

```bash
export SDSS_PRODUCTION_RUN=/path/to/production_data
python scripts/field_evolution.py
```

## Archive integrity

`Exodus_files.zip`

SHA-256:

`ed78fc7f994bcad5f49e5dd74e5f1a1d95a4cd6a53b05f40321b92c1e1b1932b`

Verify with:

```bash
sha256sum Exodus_files.zip
```

Verify individual `.e` files against `EXODUS_DATA_MANIFEST.csv`.

## Scientific boundary

These files are numerical FEM outputs. Preserving them supports reproducibility
of the reported spatial-field post-processing. It does not by itself establish
experimental validation or uniqueness of the FEM model.
