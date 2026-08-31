#!/usr/bin/env python3
"""GitHub-ready export of SDSS_field_evolution_equal_maps_cell_by_cell.ipynb.

Outputs were stripped from the notebook; repository-relative/environment-configurable paths are used.
"""


# %% [markdown]
# # SDSS field-evolution workflow — equal-size maps with per-sample colour scales

# %% [markdown]
# ## Cell 1 — Imports and Exodus reader

# %% Cell 1
from pathlib import Path
import hashlib
from io import BytesIO
import json
import os
import platform
import shutil
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.collections import LineCollection, PolyCollection
from PIL import Image
try:
    from netCDF4 import Dataset
except ImportError:
    # MOOSE Exodus files in this workflow are NetCDF classic 64-bit-offset
    # files, which SciPy can read without the optional netCDF4 package.
    from scipy.io import netcdf_file

    def Dataset(path, mode="r"):
        return netcdf_file(path, mode=mode, mmap=False)

print("Cell 1 complete: imports available.")


# %% [markdown]
# ## Cell 2 — Paths, state definitions, and publication settings

# %% Cell 2
# -----------------------------------------------------------------------------
# USER CONFIGURATION: change ROOT/RUN_DIR only if your project is elsewhere.
# The transient coordinate is reported in seconds for this workflow. This is
# solver time; it is not an independently measured experimental test duration.
ROOT = Path(os.environ.get(
    "SDSS_MOOSE_ROOT",
    Path.cwd(),
)).expanduser().resolve()

# Directory containing authoritative per-condition CSV histories and Exodus files.
# Override with SDSS_PRODUCTION_RUN when the large Exodus files live elsewhere.
RUN_DIR = Path(os.environ.get(
    "SDSS_PRODUCTION_RUN",
    ROOT / "data" / "production",
)).expanduser().resolve()

OUTPUT_DIR = Path(os.environ.get(
    "SDSS_FIELD_OUTPUT",
    ROOT / "results" / "field_evolution",
)).expanduser().resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["AS", "SR350", "SR425", "SR525", "SA1150"]

# The first two states are sampled output records nearest the stated
# gauge-averaged equivalent-plastic-strain targets. They are not exact target
# states and the 1e-4 state is not asserted to be the onset of local yielding.
STATE_SPECS = [
    {
        "key": "nearest_gauge_ep_1e-4",
        "short_label": "nearest sampled ⟨ε̄ₚ⟩g = 10⁻⁴",
        "table_label": (
            "sampled state nearest gauge-averaged equivalent plastic strain "
            "of 1e-4"
        ),
        "target": 1.0e-4,
        "selection_method": "nearest_sample",
    },
    {
        "key": "nearest_gauge_ep_2e-3",
        "short_label": "nearest sampled ⟨ε̄ₚ⟩g = 2×10⁻³",
        "table_label": (
            "sampled state nearest gauge-averaged equivalent plastic strain "
            "of 2e-3"
        ),
        "target": 2.0e-3,
        "selection_method": "nearest_sample",
    },
    {
        "key": "terminal_modeled_pre_necking",
        "short_label": "terminal modeled pre-necking state",
        "table_label": "terminal modeled pre-necking state",
        "target": None,
        "selection_method": "last_complete_record",
    },
]

FIELD_ALIASES = {
    "stress": ["von_mises_stress", "von_mises", "mises"],
    "plastic": ["equivalent_plastic_strain", "effective_plastic_strain"],
}
DISP_ALIASES = {
    "x": ["disp_x", "displacement_x", "u_x", "ux"],
    "y": ["disp_y", "displacement_y", "u_y", "uy"],
    "z": ["disp_z", "displacement_z", "u_z", "uz"],
}

COLORMAP = "turbo"
# Field values are read at the selected deformed states, but every field map is
# drawn on the common undeformed coupon mesh. This deliberately standardizes
# displayed specimen dimensions and prevents deformation amplitude from being
# encoded by map length.
DISPLAY_COORDINATE_MODE = "reference"
WARP_FACTOR = 1.0
SOLVER_TIME_UNIT = "s"
TIME_MATCH_ATOL = 1.0e-10
EXPECTED_VOLUME_CELLS = 3840
EXPECTED_TOP_FACES = 960
SURFACE_AXIS = "global z"
SURFACE_SIDE = "maximum"
EARLY_STATE_KEYS = [spec["key"] for spec in STATE_SPECS[:2]]

BOLD_FONT_PATH = Path(fm.findfont(
    fm.FontProperties(family="DejaVu Sans", weight="bold"),
    fallback_to_default=False,
)).resolve()
if not BOLD_FONT_PATH.is_file() or "Bold" not in BOLD_FONT_PATH.stem:
    raise RuntimeError(f"A genuine DejaVu Sans bold face was not resolved: {BOLD_FONT_PATH}")

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10,
    "font.weight": "bold",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

print(f"Verified bold font: {BOLD_FONT_PATH}")


# Exact authoritative production inputs. Cell 3 stops if an incomplete or
# alternate CSV/Exodus copy is selected.
EXPECTED_INPUT_SHA256 = {
    ("AS", ".csv"): "fe1ddcb14dab1f95799d071131c8abe662d828f4219feade9895d3042b0a5fef",
    ("AS", ".e"): "d813fd31a6507cc8fcee5e260ac9f0408954b119196ee97e4d0e85c80e34a4f1",
    ("SR350", ".csv"): "0e30289f926d29d014aafdf5bcbac395efa22531430d2d7faa13be5de55a51cf",
    ("SR350", ".e"): "4a019e6ee5eb63143d73c5c632a6a88ce289535ec480c6a42004ef5f50911be9",
    ("SR425", ".csv"): "ce53be632dc7d4fde51f3653e50cdabd752aac12cf8123b84ea7b73e84482898",
    ("SR425", ".e"): "8333941878465301109c57e9d1e3729b2ad0f7361ec7c02811f1aa96cdc1efd4",
    ("SR525", ".csv"): "12d1a4b7b5489204a4f518a97e13b71fddf406978589b690a23c0eac06c36bd6",
    ("SR525", ".e"): "aef90272f6640bdf9c0375f97b54a1bec683c74a2c8dae34894a7066b9df9e16",
    ("SA1150", ".csv"): "4571b86af109ff9aad68d7162d54e7b7d74dbea8c737921b086b2c8df848ad02",
    ("SA1150", ".e"): "82cfc65ddedcc0badda97161c7659b42c2800e8233ce90b3656358cfe69b5556",
}

print(f"ROOT: {ROOT}")
print(f"RUN_DIR: {RUN_DIR}")
print(f"OUTPUT_DIR: {OUTPUT_DIR}")


# %% [markdown]
# ## Cell 3 — Validate histories and hash all ten source files
# 
# This cell checks monotonicity, finite values, exact final solver time, and records SHA-256 hashes. Hashing the large Exodus files can take several seconds.

# %% Cell 3
def resolve_file(condition: str, suffix: str) -> Path:
    """Resolve nested or flat inputs and reject distinct ambiguous copies."""
    candidates = [
        RUN_DIR / condition / f"{condition}{suffix}",
        RUN_DIR / condition / f"fem_{condition}{suffix}",
        RUN_DIR / f"{condition}{suffix}",
        RUN_DIR / f"fem_{condition}{suffix}",
        ROOT / f"{condition}{suffix}",
        ROOT / f"fem_{condition}{suffix}",
    ]
    matches = []
    resolved_targets = set()
    for path in candidates:
        if path.is_file():
            target = path.resolve()
            if target not in resolved_targets:
                matches.append(path)
                resolved_targets.add(target)
    if not matches:
        raise FileNotFoundError(
            f"No {suffix} file found for {condition}.\n"
            + "\n".join(f"Checked: {path}" for path in candidates)
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous distinct {suffix} files for {condition}:\n"
            + "\n".join(str(path) for path in matches)
        )
    return matches[0]

def load_history(condition: str) -> pd.DataFrame:
    path = resolve_file(condition, ".csv")
    df = pd.read_csv(path)
    required = {
        "time", "engineering_strain", "engineering_stress",
        "gauge_plastic_strain_average", "end_displacement",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    df = df.copy()
    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if not np.isfinite(df[list(required)].to_numpy(dtype=float)).all():
        raise ValueError(f"{path} contains non-finite required values")
    if (np.diff(df["time"].to_numpy(dtype=float)) < -1.0e-12).any():
        raise ValueError(f"Time is not monotonic in {path}")
    plastic = df["gauge_plastic_strain_average"].to_numpy(dtype=float)
    if (np.diff(plastic) < -1.0e-12).any():
        raise ValueError(
            f"Gauge-averaged equivalent plastic strain is not monotonic in {path}"
        )
    if np.min(plastic) < -1.0e-12:
        raise ValueError(f"Negative gauge-averaged plastic strain in {path}")
    if not np.isclose(
        float(df["time"].iloc[-1]),
        1.0,
        atol=TIME_MATCH_ATOL,
        rtol=0.0,
    ):
        raise ValueError(f"Incomplete run for {condition}: final time != 1")
    return df


histories = {condition: load_history(condition) for condition in CONDITIONS}
exodus_files = {condition: resolve_file(condition, ".e") for condition in CONDITIONS}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_input_path(path: Path) -> str:
    for label, base in (("RUN_DIR", RUN_DIR), ("ROOT", ROOT)):
        try:
            return f"{label}/{path.relative_to(base)}"
        except ValueError:
            pass
    return str(path)


input_records = []
for condition in CONDITIONS:
    for role, suffix, path in (
        ("global_history_csv", ".csv", resolve_file(condition, ".csv")),
        ("spatial_field_exodus", ".e", exodus_files[condition]),
    ):
        actual_hash = sha256_file(path)
        expected_hash = EXPECTED_INPUT_SHA256[(condition, suffix)]
        if actual_hash != expected_hash:
            raise ValueError(
                f"Authoritative-input hash mismatch for {condition}{suffix}:\n"
                f"selected: {path}\n"
                f"expected: {expected_hash}\n"
                f"actual:   {actual_hash}"
            )
        input_records.append({
            "condition": condition,
            "role": role,
            "filename": path.name,
            "relative_path_from_run_dir": portable_input_path(path),
            "size_bytes": path.stat().st_size,
            "sha256": actual_hash,
        })

input_manifest = pd.DataFrame(input_records)
input_manifest.to_csv(OUTPUT_DIR / "input_file_manifest_sha256.csv", index=False)

print(input_manifest[["condition", "role", "filename", "size_bytes"]].to_string(index=False))
print("Cell 3 complete: histories validated and source hashes recorded.")


# %% [markdown]
# ## Cell 4 — Select and document the 15 sampled states
# 
# The table reports absolute and relative target error and bracketing records.

# %% Cell 4
def select_states(condition: str) -> pd.DataFrame:
    df = histories[condition]
    plastic = df["gauge_plastic_strain_average"].to_numpy(dtype=float)
    records = []
    used_indices = set()
    for spec in STATE_SPECS:
        target = spec["target"]
        index = len(df) - 1 if target is None else int(np.abs(plastic - target).argmin())
        if index in used_indices:
            raise ValueError(f"Duplicate selected state for {condition}: row {index}")
        used_indices.add(index)
        row = df.iloc[index]
        selected_plastic = float(row["gauge_plastic_strain_average"])
        absolute_error = (
            np.nan if target is None else selected_plastic - float(target)
        )
        relative_error_percent = (
            np.nan
            if target is None
            else 100.0 * absolute_error / float(target)
        )

        if target is None:
            lower_index = upper_index = index
        else:
            upper_candidates = np.flatnonzero(plastic >= target)
            upper_index = (
                int(upper_candidates[0])
                if len(upper_candidates)
                else len(plastic) - 1
            )
            lower_index = max(0, upper_index - 1)

        records.append({
            "condition": condition,
            "state_key": spec["key"],
            "state_label": spec["table_label"],
            "selection_method": spec["selection_method"],
            "target_gauge_effective_plastic_strain": target,
            "csv_row": index,
            "csv_time_seconds": float(row["time"]),
            "end_displacement_mm": float(row["end_displacement"]),
            "engineering_strain_percent": 100.0 * float(row["engineering_strain"]),
            "engineering_stress_mpa": float(row["engineering_stress"]),
            "gauge_effective_plastic_strain": selected_plastic,
            "target_absolute_error": absolute_error,
            "target_relative_error_percent": relative_error_percent,
            "lower_bracketing_csv_row": lower_index,
            "lower_bracketing_gauge_effective_plastic_strain": float(
                plastic[lower_index]
            ),
            "upper_bracketing_csv_row": upper_index,
            "upper_bracketing_gauge_effective_plastic_strain": float(
                plastic[upper_index]
            ),
        })
    return pd.DataFrame(records)


snapshot_table = pd.concat(
    [select_states(condition) for condition in CONDITIONS], ignore_index=True
)
snapshot_table.to_csv(OUTPUT_DIR / "selected_loading_states_all_conditions.csv", index=False)
print(snapshot_table.to_string(index=False))
print("Cell 4 complete: 15 sampled states selected with explicit errors.")


# %% [markdown]
# ## Cell 5 — Exodus coordinate, name, and dimension utilities

# %% Cell 5
def normalized_name(value: str) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def decode_names(variable) -> list[str]:
    names = []
    for row in np.asarray(variable[:]):
        if row.dtype.kind == "S":
            text = b"".join(row.tolist()).decode("utf-8", errors="replace")
        else:
            text = "".join(chr(int(value)) for value in row)
        names.append(text.split("\x00", 1)[0].strip())
    return names


def resolve_name(names: list[str], aliases: list[str], label: str) -> str:
    lookup = {normalized_name(name): name for name in names}
    for alias in aliases:
        key = normalized_name(alias)
        if key in lookup:
            return lookup[key]
    raise KeyError(f"Cannot resolve {label}. Available names: {names}")


def coordinates(dataset: Dataset) -> np.ndarray:
    if all(name in dataset.variables for name in ("coordx", "coordy", "coordz")):
        return np.column_stack([
            np.asarray(dataset.variables[name][:], dtype=float)
            for name in ("coordx", "coordy", "coordz")
        ])
    xyz = np.asarray(dataset.variables["coord"][:], dtype=float)
    if xyz.shape[0] == 3:
        return xyz.T
    if xyz.shape[1] == 3:
        return xyz
    raise ValueError(f"Unexpected coordinate-array shape: {xyz.shape}")


def dimension_size(dataset, name: str) -> int:
    """Support both netCDF4.Dimension and SciPy's integer dimensions."""
    dimension = dataset.dimensions[name]
    return int(dimension if isinstance(dimension, (int, np.integer)) else len(dimension))

print("Cell 5 complete: Exodus utilities defined.")


# %% [markdown]
# ## Cell 6 — Define validated global-maximum-z top-surface extraction
# 
# The function requires 3,840 HEX8 volume cells and 960 top faces and also reconstructs the undeformed boundary outline.

# %% Cell 6
def read_spatial_state(condition: str, requested_time_seconds: float) -> dict:
    path = exodus_files[condition]
    with Dataset(path, "r") as dataset:
        times = np.asarray(dataset.variables["time_whole"][:], dtype=float)
        time_index = int(np.abs(times - requested_time_seconds).argmin())
        time_error = abs(float(times[time_index]) - requested_time_seconds)
        if time_error > TIME_MATCH_ATOL:
            raise ValueError(
                f"CSV/Exodus time mismatch for {condition}: error={time_error:.3e}"
            )

        point_names = decode_names(dataset.variables["name_nod_var"])
        cell_names = decode_names(dataset.variables["name_elem_var"])
        disp_names = [
            resolve_name(point_names, DISP_ALIASES[axis], f"disp_{axis}")
            for axis in ("x", "y", "z")
        ]
        stress_name = resolve_name(cell_names, FIELD_ALIASES["stress"], "von Mises stress")
        plastic_name = resolve_name(
            cell_names, FIELD_ALIASES["plastic"], "equivalent plastic strain"
        )
        disp_indices = [point_names.index(name) + 1 for name in disp_names]
        stress_index = cell_names.index(stress_name) + 1
        plastic_index = cell_names.index(plastic_name) + 1

        undeformed = coordinates(dataset)
        displacement = np.column_stack([
            np.asarray(dataset.variables[f"vals_nod_var{index}"][time_index], dtype=float)
            for index in disp_indices
        ])
        if displacement.shape != undeformed.shape or not np.isfinite(displacement).all():
            raise ValueError(f"Invalid displacement array for {condition}")
        deformed = undeformed + WARP_FACTOR * displacement
        if DISPLAY_COORDINATE_MODE == "reference":
            display_coordinates = undeformed
        elif DISPLAY_COORDINATE_MODE == "deformed":
            display_coordinates = deformed
        else:
            raise ValueError(
                "DISPLAY_COORDINATE_MODE must be 'reference' or 'deformed'"
            )

        top_z = float(np.max(undeformed[:, 2]))
        z_tolerance = max(float(np.ptp(undeformed[:, 2])) * 1.0e-9, 1.0e-12)
        polygons, stress_values, plastic_values = [], [], []
        boundary_edge_counts = {}
        total_cells = 0

        number_blocks = dimension_size(dataset, "num_el_blk")
        for block_index in range(1, number_blocks + 1):
            connectivity = np.asarray(
                dataset.variables[f"connect{block_index}"][:], dtype=int
            ) - 1
            if connectivity.ndim != 2 or connectivity.shape[1] != 8:
                raise ValueError(
                    f"{condition}, block {block_index} is not HEX8: {connectivity.shape}"
                )
            total_cells += len(connectivity)
            stress_block = np.asarray(
                dataset.variables[f"vals_elem_var{stress_index}eb{block_index}"][time_index],
                dtype=float,
            )
            plastic_block = np.asarray(
                dataset.variables[f"vals_elem_var{plastic_index}eb{block_index}"][time_index],
                dtype=float,
            )
            if len(stress_block) != len(connectivity) or len(plastic_block) != len(connectivity):
                raise ValueError(f"Element-field length mismatch in {condition}, block {block_index}")

            for element_index, nodes in enumerate(connectivity):
                element_z = undeformed[nodes, 2]
                if abs(float(np.max(element_z)) - top_z) > z_tolerance:
                    continue
                top_nodes = nodes[np.abs(element_z - top_z) <= z_tolerance]
                if len(top_nodes) != 4:
                    raise ValueError(
                        f"Expected four top-face nodes; got {len(top_nodes)} in {condition}"
                    )
                undeformed_xy = undeformed[top_nodes, :2]
                center = np.mean(undeformed_xy, axis=0)
                angles = np.arctan2(
                    undeformed_xy[:, 1] - center[1],
                    undeformed_xy[:, 0] - center[0],
                )
                ordered_nodes = top_nodes[np.argsort(angles)]
                polygons.append(display_coordinates[ordered_nodes, :2])

                for edge_start, edge_end in zip(
                    ordered_nodes,
                    np.roll(ordered_nodes, -1),
                ):
                    edge = tuple(sorted((int(edge_start), int(edge_end))))
                    boundary_edge_counts[edge] = boundary_edge_counts.get(edge, 0) + 1

                stress_values.append(float(stress_block[element_index]))
                plastic_values.append(float(plastic_block[element_index]))

    if total_cells != EXPECTED_VOLUME_CELLS:
        raise ValueError(
            f"Unexpected cell count for {condition}: {total_cells}; "
            f"expected {EXPECTED_VOLUME_CELLS}"
        )
    polygons = np.asarray(polygons, dtype=float)
    stress_values = np.asarray(stress_values, dtype=float)
    plastic_values = np.asarray(plastic_values, dtype=float)
    if polygons.ndim != 3 or polygons.shape[1:] != (4, 2) or len(polygons) == 0:
        raise ValueError(f"Invalid top-surface polygons for {condition}: {polygons.shape}")
    if len(polygons) != EXPECTED_TOP_FACES:
        raise ValueError(
            f"Unexpected top-surface face count for {condition}: {len(polygons)}; "
            f"expected {EXPECTED_TOP_FACES}"
        )
    if not np.isfinite(stress_values).all() or not np.isfinite(plastic_values).all():
        raise ValueError(f"Non-finite fields for {condition}")
    if np.min(plastic_values) < -1.0e-10:
        raise ValueError(f"Negative equivalent plastic strain for {condition}")

    boundary_edges = [
        edge for edge, count in boundary_edge_counts.items() if count == 1
    ]
    if not boundary_edges:
        raise ValueError(f"No top-surface boundary edges resolved for {condition}")

    undeformed_boundary_segments = np.asarray([
        undeformed[list(edge), :2] for edge in boundary_edges
    ], dtype=float)

    return {
        "condition": condition,
        "polygons": polygons,
        "stress_values": stress_values,
        "plastic_values": plastic_values,
        "undeformed_boundary_segments": undeformed_boundary_segments,
        "requested_time_seconds": requested_time_seconds,
        "exodus_time_seconds": float(times[time_index]),
        "time_index": time_index,
        "stress_array": stress_name,
        "plastic_array": plastic_name,
        "top_surface_face_count": len(polygons),
        "top_surface_boundary_edge_count": len(boundary_edges),
        "top_surface_z": top_z,
        "top_surface_z_tolerance": z_tolerance,
        "display_coordinate_mode": DISPLAY_COORDINATE_MODE,
    }

print("Cell 6 complete: top-surface extraction defined.")


# %% [markdown]
# ## Cell 7 — Read all 15 Exodus states
# 
# This is the expensive read. Plot-only changes do not require rerunning this cell.

# %% Cell 7
states = []
for row in snapshot_table.itertuples(index=False):
    state = read_spatial_state(row.condition, row.csv_time_seconds)
    state["plastic_values_percent"] = 100.0 * state["plastic_values"]
    state.update({
        "state_key": row.state_key,
        "state_label": row.state_label,
        "selection_method": row.selection_method,
        "target_gauge_effective_plastic_strain": (
            row.target_gauge_effective_plastic_strain
        ),
        "target_absolute_error": row.target_absolute_error,
        "target_relative_error_percent": row.target_relative_error_percent,
        "engineering_strain_percent": row.engineering_strain_percent,
        "end_displacement_mm": row.end_displacement_mm,
        "gauge_effective_plastic_strain": row.gauge_effective_plastic_strain,
    })
    states.append(state)

print(f"Cell 7 complete: extracted {len(states)} spatial states.")


# %% [markdown]
# ## Cell 8 — Per-sample field scales, equal reference geometry, and plotting primitive
# 
# Each sample uses one stress scale and one plastic-strain scale shared across
# its three loading states. Every primary panel uses identical reference-mesh
# dimensions and identical x/y limits.

# %% Cell 8
def limits(state_subset: list[dict], value_key: str) -> tuple[float, float]:
    values = np.concatenate([state[value_key] for state in state_subset])
    values = values[np.isfinite(values)]
    low, high = float(np.min(values)), float(np.max(values))
    if value_key.startswith("plastic_values"):
        low = 0.0
    if high <= low:
        high = low + max(abs(low), 1.0) * 1.0e-12
    return low, high


condition_states = {
    condition: [state for state in states if state["condition"] == condition]
    for condition in CONDITIONS
}
condition_stress_limits = {
    condition: limits(condition_states[condition], "stress_values")
    for condition in CONDITIONS
}
condition_plastic_limits_percent = {
    condition: limits(condition_states[condition], "plastic_values_percent")
    for condition in CONDITIONS
}
condition_stress_norms = {
    condition: mpl.colors.Normalize(*condition_stress_limits[condition])
    for condition in CONDITIONS
}
condition_plastic_norms = {
    condition: mpl.colors.Normalize(
        *condition_plastic_limits_percent[condition]
    )
    for condition in CONDITIONS
}

per_sample_scale_table = pd.DataFrame([
    {
        "condition": condition,
        "stress_min_mpa": condition_stress_limits[condition][0],
        "stress_max_mpa": condition_stress_limits[condition][1],
        "plastic_strain_min_percent": (
            condition_plastic_limits_percent[condition][0]
        ),
        "plastic_strain_max_percent": (
            condition_plastic_limits_percent[condition][1]
        ),
    }
    for condition in CONDITIONS
])


all_display_points = np.concatenate([
    state["polygons"].reshape(-1, 2) for state in states
])
all_undeformed_points = np.concatenate([
    state["undeformed_boundary_segments"].reshape(-1, 2) for state in states
])
all_spatial_points = np.vstack([all_display_points, all_undeformed_points])

display_bounds = np.asarray([
    [
        np.min(state["polygons"][..., 0]),
        np.max(state["polygons"][..., 0]),
        np.min(state["polygons"][..., 1]),
        np.max(state["polygons"][..., 1]),
    ]
    for state in states
], dtype=float)
if not np.allclose(display_bounds, display_bounds[0], rtol=0.0, atol=1.0e-12):
    raise ValueError(
        "Reference-geometry standardization failed: map bounds differ across states"
    )

x_min, y_min = np.min(all_spatial_points, axis=0)
x_max, y_max = np.max(all_spatial_points, axis=0)
x_pad = 0.015 * max(x_max - x_min, 1.0)
y_pad = 0.08 * max(y_max - y_min, 1.0)
shared_xlim = (float(x_min - x_pad), float(x_max + x_pad))
shared_ylim = (float(y_min - y_pad), float(y_max + y_pad))


def add_field(axis, state: dict, value_key: str, norm) -> None:
    collection = PolyCollection(
        state["polygons"],
        array=state[value_key],
        cmap=COLORMAP,
        norm=norm,
        edgecolors="none",
        linewidths=0.0,
        antialiaseds=False,
    )
    axis.add_collection(collection)
    outline = LineCollection(
        state["undeformed_boundary_segments"],
        colors="black",
        linewidths=0.55,
        linestyles="dashed",
        alpha=0.75,
        zorder=3,
    )
    axis.add_collection(outline)
    axis.set_xlim(*shared_xlim)
    axis.set_ylim(*shared_ylim)
    axis.set_aspect("equal", adjustable="box")
    axis.set_axis_off()


state_lookup = {
    (state["condition"], state["state_key"]): state for state in states
}

print(per_sample_scale_table.to_string(index=False))
print(f"Shared x limits: {shared_xlim}")
print(f"Shared y limits: {shared_ylim}")
print("Verified identical displayed specimen bounds across all 15 maps.")
print("Cell 8 complete: per-sample scales and equal reference frames defined.")


# %% [markdown]
# ## Cell 9 — Define vertically grouped publication plotting functions

# %% Cell 9
def enforce_exact_bold_text(figure) -> None:
    offenders = []
    for text_artist in figure.findobj(match=mpl.text.Text):
        text = text_artist.get_text()
        if "$" in text:
            offenders.append(text)
        size = text_artist.get_fontsize()
        text_artist.set_fontproperties(
            fm.FontProperties(fname=str(BOLD_FONT_PATH), size=size)
        )
    if offenders:
        raise ValueError(
            "Mathtext remained in a publication figure: " + repr(offenders)
        )


def save_verified_figure(figure, output_path: Path, attempts: int = 3) -> None:
    """Force exact bold text, then reject truncated PNG image streams."""
    enforce_exact_bold_text(figure)
    temporary_path = output_path.with_name(f".{output_path.stem}.writing.png")
    last_error = None
    for _ in range(attempts):
        temporary_path.unlink(missing_ok=True)
        buffer = BytesIO()
        figure.savefig(
            buffer,
            format="png",
            dpi=600,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.08,
        )
        try:
            buffer.seek(0)
            with Image.open(buffer) as image:
                image.verify()
        except Exception as error:
            last_error = error
            continue

        temporary_path.write_bytes(buffer.getvalue())
        os.replace(temporary_path, output_path)
        temporary_path.unlink(missing_ok=True)
        return

    temporary_path.unlink(missing_ok=True)
    raise OSError(
        f"Could not create a valid PNG after {attempts} attempts: {output_path}"
    ) from last_error


def add_condition_colorbars(stress_cax, plastic_cax, condition: str) -> None:
    """Add the two scales corresponding to one sample block."""
    stress_bar = mpl.colorbar.ColorbarBase(
        stress_cax,
        cmap=mpl.colormaps[COLORMAP],
        norm=condition_stress_norms[condition],
        orientation="horizontal",
    )
    plastic_bar = mpl.colorbar.ColorbarBase(
        plastic_cax,
        cmap=mpl.colormaps[COLORMAP],
        norm=condition_plastic_norms[condition],
        orientation="horizontal",
    )
    stress_bar.set_label(
        "σᵥₘ (MPa)",
        fontsize=10,
        fontweight="bold",
    )
    plastic_bar.set_label(
        "ε̄ₚ (%)",
        fontsize=10,
        fontweight="bold",
    )
    stress_bar.ax.tick_params(labelsize=8.2, length=2.5, pad=1.5)
    plastic_bar.ax.tick_params(labelsize=8.2, length=2.5, pad=1.5)
    stress_bar.locator = mpl.ticker.MaxNLocator(nbins=4)
    plastic_bar.locator = mpl.ticker.MaxNLocator(nbins=4)
    stress_bar.update_ticks()
    plastic_bar.update_ticks()
    plt.setp(stress_bar.ax.get_xticklabels(), fontweight="bold")
    plt.setp(plastic_bar.ax.get_xticklabels(), fontweight="bold")


def plot_vertical_conditions(
    plotted_conditions: list[str],
    output_name: str,
    figure_size: tuple[float, float],
    panel_letters: dict[str, str],
) -> Path:
    """Stack states vertically and show the scales corresponding to each sample."""
    rows_per_condition = len(STATE_SPECS)
    figure = plt.figure(figsize=figure_size, facecolor="white")
    outer_grid = figure.add_gridspec(
        len(plotted_conditions),
        1,
        left=0.135,
        right=0.985,
        bottom=0.025,
        top=0.965,
        hspace=0.22,
    )

    figure.text(
        0.31,
        0.982,
        "σᵥₘ",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
    )
    figure.text(
        0.73,
        0.982,
        "ε̄ₚ",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
    )

    for condition_index, condition in enumerate(plotted_conditions):
        condition_grid = outer_grid[condition_index].subgridspec(
            rows_per_condition + 1,
            2,
            height_ratios=[1.0, 1.0, 1.0, 0.13],
            wspace=0.035,
            hspace=0.30,
        )
        condition_axes = np.empty((rows_per_condition, 2), dtype=object)

        for state_index, spec in enumerate(STATE_SPECS):
            state = state_lookup[(condition, spec["key"])]
            stress_axis = figure.add_subplot(condition_grid[state_index, 0])
            plastic_axis = figure.add_subplot(condition_grid[state_index, 1])
            condition_axes[state_index, 0] = stress_axis
            condition_axes[state_index, 1] = plastic_axis

            add_field(
                stress_axis,
                state,
                "stress_values",
                condition_stress_norms[condition],
            )
            add_field(
                plastic_axis,
                state,
                "plastic_values_percent",
                condition_plastic_norms[condition],
            )

            state_line = (
                f"t = {state['exodus_time_seconds']:.4f} {SOLVER_TIME_UNIT}"
            )
            stress_axis.set_title(
                state_line,
                fontsize=9.2,
                fontweight="bold",
                pad=2.0,
            )
            # Stress and plastic-strain maps in a row are the same saved state;
            # time is therefore printed once, above the stress map only.
            plastic_axis.set_title("")

        condition_axes[0, 0].text(
            -0.17,
            1.36,
            f"{panel_letters[condition]}) {condition}",
            transform=condition_axes[0, 0].transAxes,
            ha="left",
            va="bottom",
            fontsize=11.5,
            fontweight="bold",
        )

        stress_cax = figure.add_subplot(condition_grid[rows_per_condition, 0])
        plastic_cax = figure.add_subplot(condition_grid[rows_per_condition, 1])
        add_condition_colorbars(stress_cax, plastic_cax, condition)

        figure.canvas.draw()
        axis_boxes = [
            axis.get_position()
            for axis in condition_axes.reshape(-1)
        ]
        widths = np.asarray([box.width for box in axis_boxes])
        heights = np.asarray([box.height for box in axis_boxes])
        if not (
            np.allclose(widths, widths[0], rtol=0.0, atol=1.0e-10)
            and np.allclose(heights, heights[0], rtol=0.0, atol=1.0e-10)
        ):
            raise RuntimeError(
                f"Unequal stress/plastic-strain map axes detected for {condition}"
            )

    output_path = OUTPUT_DIR / output_name
    save_verified_figure(figure, output_path)
    plt.close(figure)
    return output_path

print("Cell 9 complete: publication plotting functions defined.")


# %% [markdown]
# ## Cell 10 — Create complete and split primary figures
# 
# The split figures are intended for journal-page use; the complete figure is an overview.

# %% Cell 10
panel_letters = dict(zip(CONDITIONS, "abcde"))


def matched_figure_height(number_conditions: int) -> float:
    """Keep each condition block the same physical height in every export."""
    block_height_inches = 3.15
    outer_hspace = 0.22
    usable_fraction = 0.965 - 0.025
    grid_units = number_conditions + outer_hspace * (number_conditions - 1)
    return block_height_inches * grid_units / usable_fraction


vertical_figure_path = plot_vertical_conditions(
    CONDITIONS,
    "field_evolution_vertical_all_conditions.png",
    (7.4, matched_figure_height(len(CONDITIONS))),
    panel_letters,
)
vertical_part1_path = plot_vertical_conditions(
    CONDITIONS[:3],
    "field_evolution_vertical_part1_AS_SR350_SR425.png",
    (7.4, matched_figure_height(3)),
    panel_letters,
)
vertical_part2_path = plot_vertical_conditions(
    CONDITIONS[3:],
    "field_evolution_vertical_part2_SR525_SA1150.png",
    (7.4, matched_figure_height(2)),
    panel_letters,
)

print("Cell 10 complete: vertically aligned primary figures saved.")


# %% [markdown]
# ## Cell 11 — Create stage-specific early plastic-strain detail figures
# 
# Each sample has its own explicitly displayed plastic-strain scale.

# %% Cell 11
def plot_early_plastic_detail(
    state_key: str,
    output_name: str,
) -> tuple[Path, dict[str, tuple[float, float]]]:
    detail_limits = {
        condition: limits(
            [state_lookup[(condition, state_key)]],
            "plastic_values_percent",
        )
        for condition in CONDITIONS
    }
    detail_norms = {
        condition: mpl.colors.Normalize(*detail_limits[condition])
        for condition in CONDITIONS
    }

    figure = plt.figure(
        figsize=(7.4, 9.0),
        facecolor="white",
    )
    outer_grid = figure.add_gridspec(
        len(CONDITIONS),
        1,
        left=0.17,
        right=0.98,
        bottom=0.025,
        top=0.90,
        hspace=0.30,
    )

    figure.suptitle(
        "ε̄ₚ",
        fontsize=11.0,
        fontweight="bold",
        y=0.975,
    )

    for row_index, condition in enumerate(CONDITIONS):
        condition_grid = outer_grid[row_index].subgridspec(
            2,
            1,
            height_ratios=[1.0, 0.14],
            hspace=0.20,
        )
        state = state_lookup[(condition, state_key)]
        axis = figure.add_subplot(condition_grid[0, 0])
        add_field(
            axis,
            state,
            "plastic_values_percent",
            detail_norms[condition],
        )
        axis.set_title(
            f"{panel_letters[condition]}) {condition}",
            fontsize=9.5,
            fontweight="bold",
            pad=2.0,
        )

        color_axis = figure.add_subplot(condition_grid[1, 0])
        colorbar = mpl.colorbar.ColorbarBase(
            color_axis,
            cmap=mpl.colormaps[COLORMAP],
            norm=detail_norms[condition],
            orientation="horizontal",
        )
        colorbar.set_label(
            "ε̄ₚ (%)",
            fontsize=9.5,
            fontweight="bold",
        )
        colorbar.ax.tick_params(labelsize=8.0, length=2.5, pad=1.5)
        colorbar.locator = mpl.ticker.MaxNLocator(nbins=4)
        colorbar.update_ticks()
        plt.setp(colorbar.ax.get_xticklabels(), fontweight="bold")

    output_path = OUTPUT_DIR / output_name
    save_verified_figure(figure, output_path)
    plt.close(figure)
    return output_path, detail_limits


early_1_path, early_1_limits = plot_early_plastic_detail(
    "nearest_gauge_ep_1e-4",
    "plastic_strain_detail_nearest_gauge_1e-4.png",
)
early_2_path, early_2_limits = plot_early_plastic_detail(
    "nearest_gauge_ep_2e-3",
    "plastic_strain_detail_nearest_gauge_2e-3.png",
)

print("Cell 11 complete: stage-specific early-detail figures saved.")


# %% [markdown]
# ## Cell 12 — Export reproducibility metadata and captions

# %% Cell 12
manifest = pd.DataFrame([
    {
        "condition": state["condition"],
        "state_key": state["state_key"],
        "state_label": state["state_label"],
        "selection_method": state["selection_method"],
        "target_gauge_effective_plastic_strain": (
            state["target_gauge_effective_plastic_strain"]
        ),
        "target_absolute_error": state["target_absolute_error"],
        "target_relative_error_percent": state["target_relative_error_percent"],
        "solver_time_seconds": state["exodus_time_seconds"],
        "engineering_strain_percent": state["engineering_strain_percent"],
        "end_displacement_mm": state["end_displacement_mm"],
        "gauge_effective_plastic_strain": state["gauge_effective_plastic_strain"],
        "exodus_time_index": state["time_index"],
        "stress_array": state["stress_array"],
        "plastic_array": state["plastic_array"],
        "stress_limit_min_mpa": condition_stress_limits[
            state["condition"]
        ][0],
        "stress_limit_max_mpa": condition_stress_limits[
            state["condition"]
        ][1],
        "plastic_limit_min_percent": condition_plastic_limits_percent[
            state["condition"]
        ][0],
        "plastic_limit_max_percent": condition_plastic_limits_percent[
            state["condition"]
        ][1],
        "shared_x_min": shared_xlim[0],
        "shared_x_max": shared_xlim[1],
        "shared_y_min": shared_ylim[0],
        "shared_y_max": shared_ylim[1],
        "warp_factor": WARP_FACTOR,
        "display_coordinate_mode": state["display_coordinate_mode"],
        "surface_axis": SURFACE_AXIS,
        "surface_side": SURFACE_SIDE,
        "surface_z": state["top_surface_z"],
        "surface_z_tolerance": state["top_surface_z_tolerance"],
        "top_surface_face_count": state["top_surface_face_count"],
        "top_surface_boundary_edge_count": state[
            "top_surface_boundary_edge_count"
        ],
    }
    for state in states
])
manifest.to_csv(OUTPUT_DIR / "field_evolution_manifest_all_conditions.csv", index=False)
per_sample_scale_table.to_csv(
    OUTPUT_DIR / "per_sample_color_limits.csv",
    index=False,
)

settings = {
    "analysis_scope": "five LPBF 6 wt.% Ni SDSS modeled conditions",
    "conditions": CONDITIONS,
    "state_definitions": STATE_SPECS,
    "state_selection_variable": "CSV gauge_plastic_strain_average",
    "state_selection_scope": "gauge-averaged equivalent plastic strain",
    "plotted_fields_scope": "local element fields on extracted top surface",
    "solver_time_symbol": "t",
    "solver_time_unit": SOLVER_TIME_UNIT,
    "solver_time_scope": (
        "MOOSE transient coordinate interpreted as seconds; not an "
        "independently measured experimental test duration"
    ),
    "time_matching_absolute_tolerance": TIME_MATCH_ATOL,
    "time_matching_relative_tolerance": 0.0,
    "warp_factor": WARP_FACTOR,
    "display_coordinate_mode": DISPLAY_COORDINATE_MODE,
    "display_geometry_scope": (
        "Fields evaluated at each selected state and rendered on the common "
        "undeformed/reference coupon geometry; displayed length does not encode "
        "displacement or elongation"
    ),
    "surface_selection": {
        "coordinate_system": "undeformed global coordinates",
        "axis": SURFACE_AXIS,
        "side": SURFACE_SIDE,
        "criterion": (
            "HEX8 faces having four nodes within max(ptp(z)*1e-9, 1e-12) "
            "of the global maximum undeformed z coordinate"
        ),
        "expected_top_faces": EXPECTED_TOP_FACES,
    },
    "spatial_alignment": {
        "per_axis_autoscaling": False,
        "shared_x_limits": list(shared_xlim),
        "shared_y_limits": list(shared_ylim),
        "identical_display_bounds_verified": True,
        "equal_stress_strain_axes_verified": True,
        "undeformed_outline": "black dashed line",
    },
    "figure_typography": {
        "all_visible_text_bold": True,
        "compact_symbolic_labels": True,
        "font_file": str(BOLD_FONT_PATH),
        "mathtext_used_in_figures": False,
        "primary_field_symbols": [
            "sigma_vm (MPa)",
            "equivalent plastic strain epsilon_bar_p (%)",
        ],
        "state_annotation_symbols": ["t (s), stress column only"],
        "engineering_strain_annotation_in_figure": False,
    },
    "color_normalization": {
        "primary_figures": {
            "type": "linear full-range",
            "scope": "three states within each condition",
            "stress_limits_mpa_by_condition": {
                condition: list(condition_stress_limits[condition])
                for condition in CONDITIONS
            },
            "plastic_strain_limits_percent_by_condition": {
                condition: list(
                    condition_plastic_limits_percent[condition]
                )
                for condition in CONDITIONS
            },
        },
        "early_detail_nearest_gauge_1e-4": {
            "type": "linear full-range",
            "scope": "each condition independently at this state",
            "plastic_strain_limits_percent_by_condition": {
                condition: list(early_1_limits[condition])
                for condition in CONDITIONS
            },
        },
        "early_detail_nearest_gauge_2e-3": {
            "type": "linear full-range",
            "scope": "each condition independently at this state",
            "plastic_strain_limits_percent_by_condition": {
                condition: list(early_2_limits[condition])
                for condition in CONDITIONS
            },
        },
    },
    "limitations": [
        "The 1e-4 and 2e-3 states are nearest sampled records, not exact target states.",
        "The 1e-4 state is not used as proof of the onset of local yielding.",
        "Terminal states are not strain-matched across conditions.",
        "Reference-geometry rendering suppresses visual differences in elongation; use the tabulated displacement or strain histories for deformation magnitude.",
        "Colors are comparable within a condition but not across conditions; use the corresponding numeric colorbar for cross-condition comparison.",
        "The constitutive model does not represent necking, damage, fracture, or defect-driven localization.",
    ],
    "software": {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": mpl.__version__,
        "pillow": Image.__version__,
    },
}
(OUTPUT_DIR / "analysis_settings.json").write_text(
    json.dumps(settings, indent=2) + "\n",
    encoding="utf-8",
)

history_copy_dir = OUTPUT_DIR / "source_histories_csv"
history_copy_dir.mkdir(exist_ok=True)
for condition in CONDITIONS:
    source = resolve_file(condition, ".csv")
    shutil.copy2(source, history_copy_dir / f"{condition}.csv")

caption_text = """FIGURE CAPTIONS

Primary vertical field-evolution figure. Local top-surface von Mises stress and equivalent plastic strain for the five modeled conditions. Within each condition, the stacked rows correspond, from top to bottom, to the sampled output record nearest gauge-averaged equivalent plastic strain 1e-4, the sampled output record nearest 2e-3, and the terminal modeled pre-necking state. Solver time, t, is reported in seconds once per paired stress-plastic-strain state, above the stress map only. Equivalent plastic strain is reported as percent exclusively through the color scale; engineering-strain values are not printed on the maps. All fields are rendered on the same undeformed/reference coupon geometry, so every stress and plastic-strain map has identical displayed dimensions; map length does not represent displacement or elongation. Each condition has its own full-range stress and plastic-strain colorbars, shared by its three states. Therefore, colors are quantitatively comparable within a condition but not across conditions; cross-condition comparisons require the numerical colorbars. Terminal states are not strain-matched.

Early-stage plastic-strain detail figures. Local top-surface equivalent plastic strain at the sampled output record nearest the stated gauge-averaged target. Strain is reported in percent exclusively through the color scale; time is omitted because each map represents the same paired state already identified in the primary figure. Each sample has an independently normalized, explicitly displayed colorbar. Colors must not be compared quantitatively across samples or across the two detail figures.

Model scope. The fields describe the calibrated continuum model before modeled instability; they do not demonstrate necking, damage, fracture, or defect-controlled localization.
"""
(OUTPUT_DIR / "FIGURE_CAPTIONS.txt").write_text(caption_text, encoding="utf-8")

readme_text = """# Corrected five-condition spatial-field outputs

## Publication figures

- `field_evolution_vertical_all_conditions.png`: complete five-condition vertical layout.
- `field_evolution_vertical_part1_AS_SR350_SR425.png`: journal-page part 1.
- `field_evolution_vertical_part2_SR525_SA1150.png`: journal-page part 2.
- `plastic_strain_detail_nearest_gauge_1e-4.png`: early-state plastic-strain detail.
- `plastic_strain_detail_nearest_gauge_2e-3.png`: 0.002-target plastic-strain detail.

Use the two split primary figures for manuscript preparation; the complete vertical figure is an overview. Each sample has one stress scale and one plastic-strain percentage scale shared across its three states; the corresponding colorbars are printed beneath that sample block. Spatial limits and displayed specimen dimensions are identical across all maps. Solver time is shown in seconds above the stress map only; Roman state numbers and separator bars are omitted. The paired plastic-strain map is the same saved state. Engineering-strain percentages are omitted from the maps. Because color limits differ by sample, use numerical colorbar values—not color alone—for cross-sample comparison.

All field values are rendered on the common undeformed/reference geometry. Consequently, map length is standardized and does not encode displacement or elongation. Compare deformation magnitude using `selected_loading_states_all_conditions.csv` or the original CSV histories.

## Reproducibility

- `selected_loading_states_all_conditions.csv`: state-selection records and target errors.
- `field_evolution_manifest_all_conditions.csv`: Exodus indices, fields, scales, geometry, and surface settings.
- `per_sample_color_limits.csv`: stress and plastic-strain limits used for each sample.
- `input_file_manifest_sha256.csv`: exact source-file sizes and SHA-256 hashes.
- `analysis_settings.json`: full state, plotting, geometry, and model-scope settings.
- `output_file_manifest_sha256.csv`: exact hashes of every generated deliverable.
- `source_histories_csv/`: the five CSV global histories used for state selection.
- `FIGURE_CAPTIONS.txt`: publication-ready captions and scope statements.

The Exodus databases are not duplicated in this compact output directory. Verify locally held `.e` files against `input_file_manifest_sha256.csv` before rerunning.
"""
(OUTPUT_DIR / "README_OUTPUTS.md").write_text(readme_text, encoding="utf-8")

publication_figure_paths = [
    vertical_figure_path,
    vertical_part1_path,
    vertical_part2_path,
    early_1_path,
    early_2_path,
]
for path in publication_figure_paths:
    with Image.open(path) as image:
        image.verify()

for partial_path in OUTPUT_DIR.glob(".*.writing.png"):
    partial_path.unlink(missing_ok=True)

output_manifest_path = OUTPUT_DIR / "output_file_manifest_sha256.csv"
output_records = []
for path in sorted(OUTPUT_DIR.rglob("*")):
    if not path.is_file() or path == output_manifest_path:
        continue
    output_records.append({
        "relative_path": str(path.relative_to(OUTPUT_DIR)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    })
pd.DataFrame(output_records).to_csv(output_manifest_path, index=False)

print(f"Saved primary figure: {vertical_figure_path}")
print(f"Saved split figures: {vertical_part1_path}, {vertical_part2_path}")
print(f"Saved early-detail figures: {early_1_path}, {early_2_path}")
print(f"Saved tables and reproducibility metadata in: {OUTPUT_DIR}")
print("Cell 12 complete: manifests, settings, histories, captions, and README saved.")


# %% [markdown]
# ## Output checklist
# 
# The output directory contains:
# 
# - complete and split vertically aligned primary figures;
# - two early-stage plastic-strain detail figures;
# - selected-state and spatial-field manifests;
# - exact source SHA-256 hashes;
# - analysis settings, copied CSV histories, captions, and README.
# 
# Stop at the first traceback. Do not interpret a successfully rendered figure as validation unless Cell 12 completes without error.
