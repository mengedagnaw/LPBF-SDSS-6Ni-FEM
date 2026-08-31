#!/usr/bin/env python3
"""GitHub-ready export of SDSS_strain_decomposition_cell_by_cell.ipynb.

Outputs were stripped from the notebook; repository-relative/environment-configurable paths are used.
"""


# %% Cell 1
# CELL 1 — IMPORTS, CONFIGURATION, AND PUBLICATION STYLE
from pathlib import Path
import hashlib
import math
import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
try:
    from IPython.display import display
except ImportError:
    def display(value):
        print(value.to_string(index=False) if hasattr(value, "to_string") else value)

CONDITIONS = ["AS", "SR350", "SR425", "SR525", "SA1150"]

# Calibrated Young's moduli used in the corresponding MOOSE material models.
# Source: FEM_moose/material_parameters.csv; units: MPa.
E_MPA = {
    "AS": 180350.34940407076,
    "SR350": 176264.34235990746,
    "SR425": 179585.14291039243,
    "SR525": 181408.92263957212,
    "SA1150": 190617.70939403280,
}

GAUGE_LENGTH_MM = 32.0
ROOT = Path.cwd().resolve()
DATA_ROOT = Path(os.environ.get("SDSS_FEM_HISTORY_DIR", ROOT / "data" / "fem_histories")).expanduser().resolve()
OUTPUT_DIR = Path(os.environ.get("SDSS_STRAIN_OUTPUT", ROOT / "results" / "strain_decomposition")).expanduser().resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

bold_font_path = Path(
    fm.findfont(
        fm.FontProperties(family="DejaVu Sans", weight="bold"),
        fallback_to_default=False,
    )
)
if not bold_font_path.is_file() or "Bold" not in bold_font_path.stem:
    raise RuntimeError(f"A genuine bold font was not resolved: {bold_font_path}")
BOLD = fm.FontProperties(fname=str(bold_font_path))

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.linewidth": 1.1,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "savefig.dpi": 600,
})

print("ROOT:", ROOT)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("Verified bold font:", bold_font_path)


# %% Cell 2
# CELL 2 — RESOLVE AND VALIDATE THE FIVE MOOSE HISTORY FILES
REQUIRED_COLUMNS = [
    "time",
    "engineering_strain",
    "engineering_stress",
    "gauge_left_displacement",
    "gauge_right_displacement",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_history(condition: str) -> Path:
    # First entries support the user's flat and condition-folder layouts.
    candidates = [
        DATA_ROOT / f"{condition}.csv",
        DATA_ROOT / condition / f"{condition}.csv",
        DATA_ROOT / "source_histories_csv" / f"{condition}.csv",
        DATA_ROOT / "separate_production_files" / condition / f"{condition}.csv",
        DATA_ROOT / "outputs" / "source_histories_csv" / f"{condition}.csv",
        DATA_ROOT
        / "SDSS_field_evolution_equal_maps_RELEASE_20260810"
        / "outputs"
        / "source_histories_csv"
        / f"{condition}.csv",
    ]
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise FileNotFoundError(
            f"No history CSV found for {condition}. Checked:\n"
            + "\n".join(str(path) for path in candidates)
        )

    # Duplicate copies are accepted only when their bytes are identical.
    hashes = {sha256(path) for path in existing}
    if len(hashes) > 1:
        raise RuntimeError(
            f"Conflicting CSV copies found for {condition}:\n"
            + "\n".join(str(path) for path in existing)
        )
    return existing[0]


history_files = {condition: resolve_history(condition) for condition in CONDITIONS}
raw_histories = {}
status_rows = []

for condition, path in history_files.items():
    frame = pd.read_csv(path).copy()
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    for column in REQUIRED_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame[REQUIRED_COLUMNS].to_numpy(float)).all():
        raise ValueError(f"{path} contains non-finite required values")
    if (np.diff(frame["time"].to_numpy(float)) < -1.0e-12).any():
        raise ValueError(f"Simulation time is not monotonic in {path}")
    if not np.isclose(float(frame["time"].iloc[0]), 0.0, atol=1.0e-12):
        raise ValueError(f"{condition}: first saved time is not zero")
    if not np.isclose(float(frame["time"].iloc[-1]), 1.0, atol=1.0e-10):
        raise ValueError(f"{condition}: simulation is incomplete (final time != 1)")

    raw_histories[condition] = frame
    status_rows.append({
        "condition": condition,
        "rows": len(frame),
        "csv": str(path),
        "sha256": sha256(path),
    })

input_status = pd.DataFrame(status_rows)
display(input_status)
print("Cell 2 complete: all five histories are readable and complete.")


# %% Cell 3
# CELL 3 — DECOMPOSE TOTAL AXIAL ENGINEERING STRAIN
# One-dimensional engineering approximation under monotonic uniaxial loading:
#     epsilon_total = epsilon_elastic + epsilon_plastic
#     epsilon_elastic = engineering_stress / E
# This is deliberately NOT obtained by subtracting MOOSE equivalent plastic
# strain, because an equivalent scalar is not generally an additive axial
# engineering-strain component.

decomposed = {}
summary_rows = []

for condition in CONDITIONS:
    source = raw_histories[condition]
    result = pd.DataFrame({
        "time": source["time"].to_numpy(float),
        "gauge_deformation_mm": (
            source["gauge_right_displacement"].to_numpy(float)
            - source["gauge_left_displacement"].to_numpy(float)
        ),
        "engineering_stress_mpa": source["engineering_stress"].to_numpy(float),
        "total_engineering_strain": source["engineering_strain"].to_numpy(float),
    })

    result["elastic_engineering_strain"] = (
        result["engineering_stress_mpa"] / E_MPA[condition]
    )
    result["plastic_engineering_strain"] = (
        result["total_engineering_strain"]
        - result["elastic_engineering_strain"]
    )

    # Clip only floating-point noise; reject a physically inconsistent result.
    minimum_plastic = float(result["plastic_engineering_strain"].min())
    if minimum_plastic < -1.0e-8:
        raise ValueError(
            f"{condition}: negative plastic strain ({minimum_plastic:.3e}); "
            "check E, stress measure, and strain measure"
        )
    result["plastic_engineering_strain"] = result[
        "plastic_engineering_strain"
    ].clip(lower=0.0)

    # Independent check that the gauge deformation and reported strain agree.
    strain_from_deformation = result["gauge_deformation_mm"] / GAUGE_LENGTH_MM
    gauge_error = float(np.max(np.abs(
        strain_from_deformation - result["total_engineering_strain"]
    )))
    if gauge_error > 5.0e-9:
        raise ValueError(
            f"{condition}: gauge-deformation/engineering-strain mismatch "
            f"({gauge_error:.3e})"
        )

    closure = np.abs(
        result["total_engineering_strain"]
        - result["elastic_engineering_strain"]
        - result["plastic_engineering_strain"]
    )
    closure_error = float(closure.max())
    if closure_error > 1.0e-12:
        raise ValueError(f"{condition}: decomposition closure failed")

    # Convert to percent only after all dimensionless calculations/checks.
    result["total_strain_percent"] = 100.0 * result["total_engineering_strain"]
    result["elastic_strain_percent"] = 100.0 * result["elastic_engineering_strain"]
    result["plastic_strain_percent"] = 100.0 * result["plastic_engineering_strain"]

    # Preserve the MOOSE equivalent-plastic-strain scalar as a diagnostic only.
    if "gauge_plastic_strain_average" in source.columns:
        result["moose_gauge_equivalent_plastic_strain_percent"] = (
            100.0 * pd.to_numeric(
                source["gauge_plastic_strain_average"], errors="coerce"
            ).to_numpy(float)
        )

    decomposed[condition] = result
    last = result.iloc[-1]
    summary_rows.append({
        "condition": condition,
        "E_GPa": E_MPA[condition] / 1000.0,
        "final_gauge_deformation_mm": last["gauge_deformation_mm"],
        "final_total_strain_percent": last["total_strain_percent"],
        "final_elastic_strain_percent": last["elastic_strain_percent"],
        "final_plastic_strain_percent": last["plastic_strain_percent"],
        "maximum_closure_error": closure_error,
        "maximum_gauge_consistency_error": gauge_error,
    })

decomposition_summary = pd.DataFrame(summary_rows)
display(decomposition_summary.round(6))
print("Cell 3 complete: epsilon_total = epsilon_elastic + epsilon_plastic.")


# %% Cell 4
# CELL 4 — EXPORT THE NUMERICAL DECOMPOSITION TABLES
for condition, frame in decomposed.items():
    frame.to_csv(
        OUTPUT_DIR / f"{condition}_strain_decomposition.csv",
        index=False,
        float_format="%.12g",
    )

decomposition_summary.to_csv(
    OUTPUT_DIR / "strain_decomposition_summary.csv",
    index=False,
    float_format="%.12g",
)
input_status.to_csv(OUTPUT_DIR / "input_manifest.csv", index=False)

print("Exported five decomposition histories, the summary, and input manifest.")


# %% Cell 5
# CELL 5 — DEFINE THE PUBLICATION-READY STACKED-PLOT FUNCTIONS
# High-contrast, color-vision-deficiency-aware palette (Paul Tol-inspired).
# One common label colour is used for every condition, as requested.
ELASTIC_FACE = "#66CCEE"       # cyan
ELASTIC_EDGE = "#157A9C"
PLASTIC_FACE = "#EE7733"       # orange
PLASTIC_EDGE = "#9C3E12"
TOTAL_LINE = "#004488"         # deep navy
CONDITION_LABEL = "#AA3377"    # dark magenta; shared by all labels


def next_nice_limit(value: float) -> float:
    """Condition-specific y limit with reproducible rounded headroom."""
    if value <= 0:
        return 1.0
    exponent = 10.0 ** math.floor(math.log10(value))
    scaled = value / exponent
    step = 0.5 if scaled < 5.0 else 1.0
    return math.ceil((1.08 * scaled) / step) * step * exponent


def independent_x_scale(value: float) -> tuple[float, np.ndarray]:
    """Return a condition-specific x limit and readable tick sequence.

    Seven target intervals provide adequate resolution without crowding. The
    selected step follows the conventional 1-2-2.5-5 progression, so the
    ticks are independently calculated from each condition's deformation
    range but remain publication-readable.
    """
    if value <= 0.0:
        return 1.0, np.array([0.0, 0.5, 1.0])

    padded_maximum = 1.025 * value
    raw_step = padded_maximum / 7.0
    exponent = 10.0 ** math.floor(math.log10(raw_step))
    scaled_step = raw_step / exponent
    bases = (1.0, 2.0, 2.5, 5.0, 10.0)
    base = next(candidate for candidate in bases if candidate >= scaled_step)
    step = base * exponent
    upper = math.ceil((padded_maximum - 1.0e-12) / step) * step
    ticks = np.arange(0.0, upper + 0.5 * step, step)
    return float(upper), ticks


def apply_bold_to_axis(axis) -> None:
    for artist in (
        list(axis.get_xticklabels())
        + list(axis.get_yticklabels())
        + [axis.xaxis.label, axis.yaxis.label, axis.title]
    ):
        artist.set_fontproperties(BOLD)


def separate_origin_tick_labels(axis) -> None:
    """Retain both origin values without allowing their labels to overlap."""
    x_ticks = np.asarray(axis.get_xticks(), dtype=float)
    x_labels = axis.get_xticklabels()
    zero_indices = np.flatnonzero(np.isclose(x_ticks, 0.0, atol=1.0e-12))
    if zero_indices.size != 1:
        raise RuntimeError("Expected exactly one x-axis origin tick")

    # Anchor the x=0 label at its left edge so it extends into the plotting
    # region instead of into the y-axis tick-label space.
    x_labels[int(zero_indices[0])].set_horizontalalignment("left")


def plot_condition(axis, condition: str, show_legend: bool = False) -> None:
    frame = decomposed[condition]
    x = frame["gauge_deformation_mm"].to_numpy(float)
    elastic = frame["elastic_strain_percent"].to_numpy(float)
    total = frame["total_strain_percent"].to_numpy(float)

    axis.fill_between(
        x, 0.0, elastic,
        facecolor=ELASTIC_FACE,
        edgecolor=ELASTIC_EDGE,
        linewidth=0.45,
        hatch="...",
        zorder=1,
    )
    axis.fill_between(
        x, elastic, total,
        facecolor=PLASTIC_FACE,
        edgecolor=PLASTIC_EDGE,
        linewidth=0.45,
        hatch="///",
        zorder=2,
    )
    axis.plot(
        x, total,
        color=TOTAL_LINE,
        linewidth=2.0,
        solid_capstyle="round",
        zorder=3,
    )

    axis.text(
        0.025, 0.90, condition,
        transform=axis.transAxes,
        color=CONDITION_LABEL,
        fontsize=12.0,
        va="top",
        ha="left",
        fontproperties=BOLD,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.88,
            "pad": 1.5,
        },
        zorder=5,
    )
    x_upper, x_ticks = independent_x_scale(float(x.max()))
    axis.set_xlim(0.0, x_upper)
    axis.set_xticks(x_ticks)
    axis.set_ylim(0.0, next_nice_limit(float(total.max())))
    axis.margins(x=0.0, y=0.0)
    axis.tick_params(
        axis="both",
        which="major",
        labelsize=9.0,
        length=4.0,
        width=1.0,
        pad=2.5,
        top=False,
        right=False,
    )
    axis.grid(False)
    axis.set_box_aspect(0.40)
    apply_bold_to_axis(axis)
    separate_origin_tick_labels(axis)

    if show_legend:
        handles = [
            Patch(
                facecolor=ELASTIC_FACE,
                edgecolor=ELASTIC_EDGE,
                hatch="...",
                label="εₑ",
            ),
            Patch(
                facecolor=PLASTIC_FACE,
                edgecolor=PLASTIC_EDGE,
                hatch="///",
                label="εₚ",
            ),
            Line2D(
                [0], [0],
                color=TOTAL_LINE,
                linewidth=2.0,
                label="εₜ",
            ),
        ]
        legend = axis.legend(
            handles=handles,
            # One-column order: elastic, plastic, total strain.
            loc="upper left",
            bbox_to_anchor=(0.42, 0.985),
            ncol=1,
            frameon=False,
            handlelength=2.0,
            handletextpad=0.45,
            labelspacing=0.42,
            borderaxespad=0.0,
            prop=BOLD,
            fontsize=8.2,
        )
        for text_artist in legend.get_texts():
            text_artist.set_fontproperties(BOLD)


def save_verified_figure(figure, png_path: Path) -> tuple[Path, Path]:
    """Atomically save a 600-dpi PNG and a vector PDF, then verify both."""
    pdf_path = png_path.with_suffix(".pdf")
    temporary = png_path.with_name(png_path.stem + ".tmp.png")
    temporary.unlink(missing_ok=True)
    figure.savefig(
        temporary,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    from PIL import Image
    with Image.open(temporary) as image:
        image.verify()
    temporary.replace(png_path)

    temporary_pdf = pdf_path.with_name(pdf_path.stem + ".tmp.pdf")
    temporary_pdf.unlink(missing_ok=True)
    figure.savefig(
        temporary_pdf,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    if not temporary_pdf.read_bytes().startswith(b"%PDF-"):
        raise RuntimeError(f"Invalid PDF export: {temporary_pdf}")
    temporary_pdf.replace(pdf_path)
    if temporary.exists() or temporary_pdf.exists():
        raise RuntimeError(f"Temporary export artifact was not removed: {png_path}")
    return png_path, pdf_path


def make_decomposition_figure(conditions, filename: str, panel_letters=True):
    nrows = len(conditions)
    figure, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(7.1, 2.42 * nrows + 0.62),
        constrained_layout=False,
    )
    axes = np.atleast_1d(axes)

    for index, (axis, condition) in enumerate(zip(axes, conditions)):
        plot_condition(axis, condition, show_legend=(index == 0))
        axis.set_ylabel("ε (%)", fontsize=11.0, fontproperties=BOLD)
        # Each panel displays its own independently calculated x-axis ticks.
        axis.tick_params(axis="x", labelbottom=True)
        if panel_letters:
            axis.text(
                -0.105, 1.01, f"({chr(97 + index)})",
                transform=axis.transAxes,
                ha="left", va="bottom",
                fontsize=10.0,
                fontproperties=BOLD,
            )
        apply_bold_to_axis(axis)

    axes[-1].set_xlabel("ΔL (mm)", fontsize=11.0, fontproperties=BOLD)
    figure.align_ylabels(axes)
    figure.subplots_adjust(
        left=0.145,
        right=0.985,
        top=0.975,
        bottom=0.075,
        hspace=0.17,
    )

    # Hard alignment check: every vertically stacked panel must have the same
    # left edge and width in figure coordinates.
    figure.canvas.draw()
    bounds = np.array([axis.get_position().bounds for axis in axes], dtype=float)
    if not np.allclose(bounds[:, 0], bounds[0, 0], atol=1.0e-12):
        raise RuntimeError("Panel left edges are not vertically aligned")
    if not np.allclose(bounds[:, 2], bounds[0, 2], atol=1.0e-12):
        raise RuntimeError("Panel widths are not identical")

    # Retain both numerical origins and verify their rendered labels are
    # spatially separated in every panel.
    renderer = figure.canvas.get_renderer()
    for axis, condition in zip(axes, conditions):
        x_ticks = np.asarray(axis.get_xticks(), dtype=float)
        y_ticks = np.asarray(axis.get_yticks(), dtype=float)
        x_zero = np.flatnonzero(np.isclose(x_ticks, 0.0, atol=1.0e-12))
        y_zero = np.flatnonzero(np.isclose(y_ticks, 0.0, atol=1.0e-12))
        if x_zero.size != 1 or y_zero.size != 1:
            raise RuntimeError(f"{condition}: origin tick is not uniquely defined")
        x_label = axis.get_xticklabels()[int(x_zero[0])]
        y_label = axis.get_yticklabels()[int(y_zero[0])]
        if x_label.get_window_extent(renderer).overlaps(
            y_label.get_window_extent(renderer)
        ):
            raise RuntimeError(f"{condition}: origin tick labels overlap")

    # Hard x-axis check: every panel must retain its condition-specific scale
    # and show the independently calculated tick labels.
    for axis, condition in zip(axes, conditions):
        maximum = float(decomposed[condition]["gauge_deformation_mm"].max())
        expected_upper, expected_ticks = independent_x_scale(maximum)
        if not np.allclose(axis.get_xlim(), (0.0, expected_upper), atol=1.0e-12):
            raise RuntimeError(f"{condition}: incorrect independent x limits")
        if not np.allclose(axis.get_xticks(), expected_ticks, atol=1.0e-12):
            raise RuntimeError(f"{condition}: incorrect independent x ticks")
        if not all(label.get_visible() for label in axis.get_xticklabels()):
            raise RuntimeError(f"{condition}: x tick labels are not all visible")

    output_path = OUTPUT_DIR / filename
    png_path, pdf_path = save_verified_figure(figure, output_path)
    plt.show()
    plt.close(figure)
    print("Saved:", png_path)
    print("Saved:", pdf_path)
    return png_path, pdf_path


print("Cell 5 complete: plotting functions are defined.")

x_axis_rows = []
for condition in CONDITIONS:
    maximum = float(decomposed[condition]["gauge_deformation_mm"].max())
    upper, ticks = independent_x_scale(maximum)
    x_axis_rows.append({
        "condition": condition,
        "maximum_gauge_deformation_mm": maximum,
        "x_axis_upper_limit_mm": upper,
        "x_tick_values_mm": "; ".join(f"{tick:g}" for tick in ticks),
    })
pd.DataFrame(x_axis_rows).to_csv(
    OUTPUT_DIR / "independent_x_axis_ticks.csv",
    index=False,
)
display(pd.DataFrame(x_axis_rows))


# %% Cell 6
# CELL 6 — GENERATE THE FIVE-CONDITION AND JOURNAL-SPLIT FIGURES
figure_all_png, figure_all_pdf = make_decomposition_figure(
    CONDITIONS,
    "strain_decomposition_all_conditions.png",
)

figure_part1_png, figure_part1_pdf = make_decomposition_figure(
    ["AS", "SR350", "SR425"],
    "strain_decomposition_part1_AS_SR350_SR425.png",
)

figure_part2_png, figure_part2_pdf = make_decomposition_figure(
    ["SR525", "SA1150"],
    "strain_decomposition_part2_SR525_SA1150.png",
)

print("Cell 6 complete: all figures generated at 600 dpi.")


# %% Cell 7
# CELL 7 — FINAL OUTPUT AND SCIENTIFIC-CONSISTENCY AUDIT
expected = [
    OUTPUT_DIR / "strain_decomposition_all_conditions.png",
    OUTPUT_DIR / "strain_decomposition_all_conditions.pdf",
    OUTPUT_DIR / "strain_decomposition_part1_AS_SR350_SR425.png",
    OUTPUT_DIR / "strain_decomposition_part1_AS_SR350_SR425.pdf",
    OUTPUT_DIR / "strain_decomposition_part2_SR525_SA1150.png",
    OUTPUT_DIR / "strain_decomposition_part2_SR525_SA1150.pdf",
    OUTPUT_DIR / "strain_decomposition_summary.csv",
    OUTPUT_DIR / "input_manifest.csv",
    OUTPUT_DIR / "independent_x_axis_ticks.csv",
] + [
    OUTPUT_DIR / f"{condition}_strain_decomposition.csv"
    for condition in CONDITIONS
]

missing = [path for path in expected if not path.is_file() or path.stat().st_size == 0]
if missing:
    raise RuntimeError("Missing or empty outputs:\n" + "\n".join(map(str, missing)))

audit_rows = []
for path in expected:
    audit_rows.append({
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    })
output_manifest = pd.DataFrame(audit_rows)
output_manifest.to_csv(OUTPUT_DIR / "output_manifest.csv", index=False)

method_note = """Method note
===========
The plotted quantities are macroscopic axial engineering-strain components.
For each condition, epsilon_e = sigma_eng / E and epsilon_p = epsilon_total - epsilon_e.
The horizontal axis is gauge deformation, Delta L_g = u_right - u_left.
Each condition uses independently calculated Delta L_g tick values and limits.
The decomposition is a one-dimensional engineering approximation for monotonic
uniaxial pre-necking loading. It is not a tensorial finite-strain decomposition.
The MOOSE gauge-averaged equivalent plastic strain is retained only as a separate
diagnostic because it is not generally additive to axial engineering strain.
"""
(OUTPUT_DIR / "METHOD_NOTE.txt").write_text(method_note, encoding="utf-8")

display(output_manifest)
print("All outputs passed existence, size, closure, and SHA-256 checks.")


# %% Cell 8

