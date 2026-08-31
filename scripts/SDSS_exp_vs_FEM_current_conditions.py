#!/usr/bin/env python3
"""Compare experimental and MOOSE engineering stress-strain curves.

Current LPBF +6 wt.% Ni SDSS conditions:
    AS, SR350, SR425, SR525, SA1150

Flat-folder launcher
--------------------
This version is intended to sit in the same directory as
``SDSS_exp_vs_FEM_current_conditions.ipynb`` and the eleven flat input files:
five ``*_mean_curve.csv`` experimental files, five condition-named FEM CSV
files, and ``comparison_parameters.csv``.  It uses a flat, repository-friendly input layout and requires no manual path
editing when the script and required CSV inputs are kept together.

Scientific scope
----------------
The experimental inputs are condition-level mean engineering curves with one
standard deviation from three calibration replicates.  The MOOSE inputs are
coupon histories containing compatible engineering strain and engineering
stress.  Comparisons are restricted to the condition-specific common uniform-
strain interval used for calibration.  Therefore, the resulting errors are
calibration-replay descriptors, not independent validation statistics.

No smoothing, stress-measure substitution, strain conversion, or manual sample
exclusion is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

# Matplotlib needs a writable cache directory in restricted/HPC environments.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sdss_exp_fem_mpl")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import pandas as pd
from PIL import Image


CONDITIONS = ("AS", "SR350", "SR425", "SR525", "SA1150")

DISPLAY_NAMES = {
    "AS": "As-built",
    "SR350": "350 °C, 2 h, furnace cooled",
    "SR425": "425 °C, 1 h, furnace cooled",
    "SR525": "525 °C, 1 h, furnace cooled",
    "SA1150": "1150 °C, 10 min, water quenched",
}

# Explicit flat-folder mapping: no AS/SR/SA grouping and no heuristic file
# discovery. This is the layout visible in the user's JupyterLab root.
EXPERIMENTAL_FILES = {
    condition: Path(f"{condition}_mean_curve.csv")
    for condition in CONDITIONS
}
FEM_FILES = {
    condition: Path(f"{condition}.csv")
    for condition in CONDITIONS
}
PARAMETER_FILE = Path("comparison_parameters.csv")

EXP_STRAIN = "engineering_strain"
EXP_MEAN_STRESS = "mean_engineering_stress_mpa"
EXP_SD_STRESS = "sd_engineering_stress_mpa"
EXP_N = "n_replicates"
FEM_STRAIN = "engineering_strain"
FEM_STRESS = "engineering_stress"

# Publication-oriented visual grammar with redundant color and line-style encoding.
# The experimental inputs are summary curves. The figure displays only the
# experimental mean; raw trajectories are never reconstructed from summary
# data. The blue/vermillion pair is the Okabe-Ito colorblind-safe pairing,
# reinforced by solid/dashed line styles for grayscale reproduction.
EXPERIMENT_COLOR = "#0072B2"       # deep blue
FEM_COLOR = "#D55E00"              # vivid vermillion
CONDITION_BADGE_COLOR = "#EEE6F7"  # pale lavender
CONDITION_LABEL_COLOR = "#6A1B9A"  # dark purple
GRID_COLOR = "#E3E3E3"

# Independent condition-specific limits preserve the validated comparison
# intervals while reproducing the compact axes of the reference template.
X_AXIS_LIMITS_PCT = {
    "AS": 8.0,
    "SR350": 8.0,
    "SR425": 7.2,
    "SR525": 9.0,
    "SA1150": 22.0,
}
X_TICK_STEPS_PCT = {
    "AS": 1.0,
    "SR350": 1.0,
    "SR425": 1.0,
    "SR525": 1.0,
    "SA1150": 5.0,
}
Y_AXIS_LIMITS_MPA = {
    "AS": 1100.0,
    "SR350": 1100.0,
    "SR425": 1100.0,
    "SR525": 1100.0,
    "SA1150": 850.0,
}
Y_TICK_STEPS_MPA = {
    "AS": 200.0,
    "SR350": 200.0,
    "SR425": 200.0,
    "SR525": 200.0,
    "SA1150": 150.0,
}
POINTS_PER_COMPARISON = 1200
REACH_TOLERANCE_STRAIN = 2.0e-6
PNG_DPI = 600


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, required: tuple[str, ...], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label}: missing required column(s): {missing}")


def clean_curve(
    frame: pd.DataFrame,
    strain_column: str,
    stress_columns: tuple[str, ...],
    label: str,
) -> pd.DataFrame:
    """Return a finite, non-negative, strictly increasing engineering curve."""
    require_columns(frame, (strain_column, *stress_columns), label)
    clean = frame.loc[:, [strain_column, *stress_columns]].copy()
    for column in clean.columns:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    if clean.isna().any().any() or not np.isfinite(clean.to_numpy()).all():
        raise ValueError(f"{label}: non-numeric or non-finite curve values")
    if (clean[strain_column] < -1.0e-12).any():
        raise ValueError(f"{label}: negative engineering strain")
    if (clean.loc[:, list(stress_columns)] < -1.0e-9).any().any():
        raise ValueError(f"{label}: negative engineering stress")

    clean = clean.sort_values(strain_column, kind="mergesort").reset_index(drop=True)
    strain = clean[strain_column].to_numpy(dtype=float)
    if len(clean) < 10:
        raise ValueError(f"{label}: fewer than 10 usable records")
    if abs(strain[0]) > 1.0e-12:
        raise ValueError(f"{label}: curve does not start at zero strain")
    if np.any(np.diff(strain) <= 0.0):
        raise ValueError(f"{label}: engineering strain is not strictly increasing")
    return clean


def load_experimental(path: Path, condition: str) -> tuple[pd.DataFrame, int]:
    frame = pd.read_csv(path)
    required = (EXP_STRAIN, EXP_MEAN_STRESS, EXP_SD_STRESS, EXP_N)
    require_columns(frame, required, f"{condition} experimental")
    clean = clean_curve(
        frame,
        EXP_STRAIN,
        (EXP_MEAN_STRESS, EXP_SD_STRESS),
        f"{condition} experimental",
    )

    n_values = pd.to_numeric(frame[EXP_N], errors="coerce")
    if n_values.isna().any() or not np.allclose(n_values, np.round(n_values)):
        raise ValueError(f"{condition}: invalid replicate count")
    unique_n = np.unique(n_values.astype(int))
    if unique_n.size != 1 or unique_n[0] < 2:
        raise ValueError(f"{condition}: replicate count must be one constant integer >= 2")
    if (clean[EXP_SD_STRESS] < 0.0).any():
        raise ValueError(f"{condition}: negative experimental standard deviation")
    return clean, int(unique_n[0])


def load_fem(path: Path, condition: str) -> pd.DataFrame:
    return clean_curve(
        pd.read_csv(path),
        FEM_STRAIN,
        (FEM_STRESS,),
        f"{condition} FEM",
    )


def load_parameters(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = (
        "condition",
        "heat_treatment",
        "common_uniform_comparison_limit_pct",
        "calibration_replicates",
    )
    require_columns(frame, required, "comparison parameters")
    if frame["condition"].duplicated().any():
        raise ValueError("comparison parameters: duplicate condition rows")
    indexed = frame.set_index("condition")
    missing = [condition for condition in CONDITIONS if condition not in indexed.index]
    extra = [condition for condition in indexed.index if condition not in CONDITIONS]
    if missing or extra:
        raise ValueError(f"comparison parameters: missing={missing}, unexpected={extra}")
    return indexed.loc[list(CONDITIONS)]


def compare_condition(
    condition: str,
    experimental: pd.DataFrame,
    fem: pd.DataFrame,
    comparison_limit: float,
    replicate_count: int,
) -> tuple[pd.DataFrame, dict[str, float | int | str | bool]]:
    exp_max = float(experimental[EXP_STRAIN].iloc[-1])
    fem_max = float(fem[FEM_STRAIN].iloc[-1])
    reached = fem_max + REACH_TOLERANCE_STRAIN >= comparison_limit
    if not reached:
        raise ValueError(
            f"{condition}: FEM maximum strain {fem_max:.9g} does not reach "
            f"the requested comparison limit {comparison_limit:.9g}"
        )
    if exp_max + REACH_TOLERANCE_STRAIN < comparison_limit:
        raise ValueError(
            f"{condition}: experimental maximum strain {exp_max:.9g} does not "
            f"reach the requested comparison limit {comparison_limit:.9g}"
        )

    actual_limit = min(comparison_limit, exp_max, fem_max)
    grid = np.linspace(0.0, actual_limit, POINTS_PER_COMPARISON)
    exp_mean = np.interp(grid, experimental[EXP_STRAIN], experimental[EXP_MEAN_STRESS])
    exp_sd = np.interp(grid, experimental[EXP_STRAIN], experimental[EXP_SD_STRESS])
    fem_stress = np.interp(grid, fem[FEM_STRAIN], fem[FEM_STRESS])
    residual = fem_stress - exp_mean

    grid_frame = pd.DataFrame(
        {
            "engineering_strain": grid,
            "engineering_strain_percent": 100.0 * grid,
            "experimental_mean_engineering_stress_mpa": exp_mean,
            "experimental_sd_engineering_stress_mpa": exp_sd,
            "fem_engineering_stress_mpa": fem_stress,
            "fem_minus_experiment_mpa": residual,
        }
    )

    denominator = float(np.sum((exp_mean - exp_mean.mean()) ** 2))
    r_squared = float(1.0 - np.sum(residual**2) / denominator)
    max_index = int(np.argmax(np.abs(residual)))
    metrics: dict[str, float | int | str | bool] = {
        "condition": condition,
        "calibration_replicates": replicate_count,
        "requested_common_limit_strain": comparison_limit,
        "actual_comparison_limit_strain": actual_limit,
        "fem_maximum_strain": fem_max,
        "reached_requested_limit_with_tolerance": reached,
        "comparison_points": POINTS_PER_COMPARISON,
        "rmse_mpa": float(np.sqrt(np.mean(residual**2))),
        "mae_mpa": float(np.mean(np.abs(residual))),
        "mean_bias_mpa_fem_minus_experiment": float(np.mean(residual)),
        "maximum_absolute_error_mpa": float(np.abs(residual[max_index])),
        "strain_at_maximum_absolute_error_percent": float(100.0 * grid[max_index]),
        "descriptive_r_squared": r_squared,
        "scope": "calibration replay through common uniform strain; not independent validation",
    }
    return grid_frame, metrics


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.weight": "bold",
            "font.size": 10.0,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": 1.10,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "legend.frameon": True,
            "legend.framealpha": 0.90,
            "legend.fancybox": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def style_axis(axis: plt.Axes, condition: str) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.10)
    axis.spines["bottom"].set_linewidth(1.10)
    axis.grid(True, color=GRID_COLOR, linewidth=0.65, alpha=0.48)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", which="major", labelsize=9.2, length=4.5, pad=3.0)
    axis.xaxis.set_major_locator(MultipleLocator(X_TICK_STEPS_PCT[condition]))
    axis.yaxis.set_major_locator(MultipleLocator(Y_TICK_STEPS_MPA[condition]))
    axis.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: f"{value:g}%")
    )
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        label.set_fontweight("bold")


def separate_origin_labels(axis: plt.Axes) -> None:
    """Keep x=0% and y=0 legible where the two axes meet."""
    axis.figure.canvas.draw()
    x_labels = axis.get_xticklabels()
    if x_labels:
        x_labels[0].set_ha("left")


def plot_one_axis(
    axis: plt.Axes,
    condition: str,
    values: pd.DataFrame,
    panel_letter: str | None = None,
) -> None:
    strain_pct = values["engineering_strain_percent"].to_numpy()
    mean = values["experimental_mean_engineering_stress_mpa"].to_numpy()
    fem = values["fem_engineering_stress_mpa"].to_numpy()

    plotted_max_stress = float(max(np.max(mean), np.max(fem)))
    if float(strain_pct[-1]) > X_AXIS_LIMITS_PCT[condition] + 1.0e-9:
        raise RuntimeError(f"{condition}: x-axis clips validated comparison data")
    if plotted_max_stress > Y_AXIS_LIMITS_MPA[condition] + 1.0e-9:
        raise RuntimeError(f"{condition}: y-axis clips validated comparison data")

    axis.plot(
        strain_pct,
        mean,
        color=EXPERIMENT_COLOR,
        linewidth=2.10,
        solid_capstyle="round",
        zorder=3,
    )
    axis.plot(
        strain_pct,
        fem,
        color=FEM_COLOR,
        linewidth=2.05,
        linestyle=(0, (4.6, 2.2)),
        solid_capstyle="round",
        zorder=4,
    )

    axis.text(
        0.50,
        0.225,
        condition,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=CONDITION_LABEL_COLOR,
        zorder=10,
        bbox={
            "boxstyle": "circle,pad=0.52",
            "facecolor": CONDITION_BADGE_COLOR,
            "edgecolor": "none",
            "alpha": 0.96,
        },
    )
    if panel_letter is not None:
        axis.text(
            -0.115,
            1.025,
            f"{panel_letter})",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=12.0,
            fontweight="bold",
            clip_on=False,
        )
    axis.set_xlabel(r"$\varepsilon$ (%)", fontsize=11.0, fontweight="bold")
    axis.set_ylabel(r"$\sigma$ (MPa)", fontsize=11.0, fontweight="bold")
    axis.set_xlim(0.0, X_AXIS_LIMITS_PCT[condition])
    axis.set_ylim(0.0, Y_AXIS_LIMITS_MPA[condition])
    style_axis(axis, condition)
    separate_origin_labels(axis)


def legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=EXPERIMENT_COLOR,
            linewidth=2.10,
            label="Exp.",
        ),
        Line2D(
            [0],
            [0],
            color=FEM_COLOR,
            linewidth=2.05,
            linestyle=(0, (4.6, 2.2)),
            label="FEM",
        ),
    ]


def save_figure(fig: plt.Figure, png_path: Path, pdf_path: Path | None = None) -> None:
    """Save verified figure files via atomic replacement.

    Atomic publication prevents a delayed reader from observing a partially
    written PNG/PDF on networked or synchronized filesystems.
    """
    png_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for _ in range(3):
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{png_path.stem}.", suffix=".png", dir=png_path.parent
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            fig.savefig(
                temporary_path,
                dpi=PNG_DPI,
                bbox_inches="tight",
                pad_inches=0.04,
                format="png",
            )
            with Image.open(temporary_path) as image:
                image.verify()
            os.replace(temporary_path, png_path)
            with Image.open(png_path) as image:
                image.verify()
            last_error = None
            break
        except Exception as error:
            last_error = error
            temporary_path.unlink(missing_ok=True)
    if last_error is not None:
        raise OSError(f"failed to publish a complete PNG: {png_path}") from last_error

    if pdf_path is not None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{pdf_path.stem}.", suffix=".pdf", dir=pdf_path.parent
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            fig.savefig(
                temporary_path,
                bbox_inches="tight",
                pad_inches=0.04,
                format="pdf",
            )
            if temporary_path.stat().st_size < 1000 or temporary_path.read_bytes()[:5] != b"%PDF-":
                raise OSError(f"invalid temporary PDF export: {temporary_path}")
            os.replace(temporary_path, pdf_path)
            if pdf_path.stat().st_size < 1000 or pdf_path.read_bytes()[:5] != b"%PDF-":
                raise OSError(f"invalid published PDF export: {pdf_path}")
        finally:
            temporary_path.unlink(missing_ok=True)


def make_individual_figures(comparisons: dict[str, pd.DataFrame], output_dir: Path) -> None:
    for condition in CONDITIONS:
        fig, axis = plt.subplots(figsize=(4.25, 3.75))
        plot_one_axis(axis, condition, comparisons[condition])
        axis.legend(
            handles=legend_handles(),
            loc="center right",
            fontsize=8.1,
            handlelength=2.5,
            borderpad=0.7,
            facecolor="#ECECEC",
            edgecolor="none",
        )
        fig.tight_layout()
        save_figure(
            fig,
            output_dir / "individual" / f"{condition}_experimental_vs_FEM.png",
            output_dir / "individual" / f"{condition}_experimental_vs_FEM.pdf",
        )
        plt.close(fig)


def make_vertical_figure(comparisons: dict[str, pd.DataFrame], output_dir: Path) -> None:
    fig, axes = plt.subplots(
        len(CONDITIONS),
        1,
        figsize=(5.45, 17.3),
        sharex=False,
        constrained_layout=False,
    )
    for panel_index, (axis, condition) in enumerate(zip(axes, CONDITIONS)):
        plot_one_axis(
            axis,
            condition,
            comparisons[condition],
            panel_letter=chr(ord("a") + panel_index),
        )

    # Match the compact in-panel legend used by the reference template.
    axes[0].legend(
        handles=legend_handles(),
        loc="center right",
        fontsize=8.1,
        handlelength=2.5,
        borderpad=0.7,
        facecolor="#ECECEC",
        edgecolor="none",
    )
    fig.subplots_adjust(left=0.20, right=0.975, top=0.985, bottom=0.045, hspace=0.56)

    # Hard alignment check before export.
    bounds = np.asarray([axis.get_position().bounds for axis in axes])
    if not np.allclose(bounds[:, 0], bounds[0, 0], atol=1.0e-10):
        raise RuntimeError("vertical figure: panel left edges are not aligned")
    if not np.allclose(bounds[:, 2], bounds[0, 2], atol=1.0e-10):
        raise RuntimeError("vertical figure: panel widths are not identical")

    save_figure(
        fig,
        output_dir / "experimental_vs_FEM_all_conditions_vertical.png",
        output_dir / "experimental_vs_FEM_all_conditions_vertical.pdf",
    )
    plt.close(fig)


def make_compact_figure(comparisons: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Create a compact equal-width 3-over-2 journal composite."""
    fig = plt.figure(figsize=(10.4, 6.9), constrained_layout=False)
    grid = fig.add_gridspec(2, 6)
    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]),
        fig.add_subplot(grid[1, 1:3]),
        fig.add_subplot(grid[1, 3:5]),
    ]
    for panel_index, (axis, condition) in enumerate(zip(axes, CONDITIONS)):
        plot_one_axis(
            axis,
            condition,
            comparisons[condition],
            panel_letter=chr(ord("a") + panel_index),
        )

    axes[0].legend(
        handles=legend_handles(),
        loc="center right",
        fontsize=7.4,
        handlelength=2.3,
        borderpad=0.6,
        facecolor="#ECECEC",
        edgecolor="none",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.965, bottom=0.095, wspace=0.90, hspace=0.70)

    bounds = np.asarray([axis.get_position().bounds for axis in axes])
    if not np.allclose(bounds[:, 2], bounds[0, 2], rtol=0.0, atol=1.0e-10):
        raise RuntimeError("compact figure: panel widths are not identical")
    if not np.allclose(bounds[:, 3], bounds[0, 3], rtol=0.0, atol=1.0e-10):
        raise RuntimeError("compact figure: panel heights are not identical")

    save_figure(
        fig,
        output_dir / "experimental_vs_FEM_all_conditions_compact.png",
        output_dir / "experimental_vs_FEM_all_conditions_compact.pdf",
    )
    plt.close(fig)


def run(base_dir: Path) -> None:
    configure_matplotlib()
    required_paths = [
        *(base_dir / EXPERIMENTAL_FILES[c] for c in CONDITIONS),
        *(base_dir / FEM_FILES[c] for c in CONDITIONS),
        base_dir / PARAMETER_FILE,
    ]
    missing_paths = [path.name for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            "Missing required flat-folder input file(s): "
            + ", ".join(missing_paths)
            + ". Keep this Python file, the notebook, and all eleven CSV "
            "inputs together in the same directory."
        )
    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison_grids").mkdir(parents=True, exist_ok=True)

    parameters = load_parameters(base_dir / PARAMETER_FILE)
    comparisons: dict[str, pd.DataFrame] = {}
    metrics_rows: list[dict[str, float | int | str | bool]] = []
    inputs_rows: list[dict[str, str]] = []

    for condition in CONDITIONS:
        experimental_path = base_dir / EXPERIMENTAL_FILES[condition]
        fem_path = base_dir / FEM_FILES[condition]
        experimental, replicate_count = load_experimental(experimental_path, condition)
        fem = load_fem(fem_path, condition)

        expected_n = int(parameters.loc[condition, "calibration_replicates"])
        if replicate_count != expected_n:
            raise ValueError(
                f"{condition}: experimental n={replicate_count}, parameter n={expected_n}"
            )
        limit = float(
            parameters.loc[condition, "common_uniform_comparison_limit_pct"]
        ) / 100.0
        comparison, metrics = compare_condition(
            condition, experimental, fem, limit, replicate_count
        )
        metrics["thermal_treatment"] = str(parameters.loc[condition, "heat_treatment"])
        metrics["experimental_csv_sha256"] = sha256(experimental_path)
        metrics["fem_csv_sha256"] = sha256(fem_path)
        comparisons[condition] = comparison
        metrics_rows.append(metrics)
        comparison.to_csv(
            output_dir / "comparison_grids" / f"{condition}_comparison_grid.csv",
            index=False,
        )
        inputs_rows.extend(
            [
                {
                    "condition": condition,
                    "role": "experimental_mean_curve",
                    "path": str(EXPERIMENTAL_FILES[condition]),
                    "sha256": sha256(experimental_path),
                },
                {
                    "condition": condition,
                    "role": "fem_coupon_history",
                    "path": str(FEM_FILES[condition]),
                    "sha256": sha256(fem_path),
                },
            ]
        )

    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_frame.to_csv(output_dir / "comparison_metrics.csv", index=False)
    pd.DataFrame(inputs_rows).to_csv(output_dir / "input_manifest.csv", index=False)
    with (output_dir / "analysis_settings.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "conditions": list(CONDITIONS),
                "experimental_measure": "mean engineering stress versus engineering strain",
                "experimental_uncertainty_display": "not displayed",
                "fem_measure": "reaction-force engineering stress versus gauge engineering strain",
                "comparison_scope": "condition-specific common uniform-strain interval",
                "comparison_grid_points": POINTS_PER_COMPARISON,
                "smoothing": False,
                "png_dpi": PNG_DPI,
                "visual_template": "publication-style layout with colorblind-safe distinct encodings",
                "experimental_visual_encoding": "deep-blue solid experimental-mean line; no uncertainty band",
                "fem_visual_encoding": "vivid-vermillion dashed FEM line",
                "condition_badge": "pale-lavender circular badge with dark-purple condition label at y=0.225 axes fraction",
                "x_axis_limits_percent": X_AXIS_LIMITS_PCT,
                "y_axis_limits_mpa": Y_AXIS_LIMITS_MPA,
            },
            handle,
            indent=2,
        )

    make_individual_figures(comparisons, output_dir)
    make_vertical_figure(comparisons, output_dir)
    make_compact_figure(comparisons, output_dir)

    print(metrics_frame.loc[:, ["condition", "rmse_mpa", "mae_mpa", "descriptive_r_squared"]].to_string(index=False))
    print(f"\nSaved outputs to: {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the eleven flat CSV inputs and receiving outputs/.",
    )
    args = parser.parse_args()
    run(args.base_dir.resolve())


if __name__ == "__main__":
    main()
