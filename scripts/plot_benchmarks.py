#!/usr/bin/env python3
"""
Voxel Engine Benchmark Visualiser
===================================
Reads CSV/JSON results produced by renderer-naive and renderer-optimised
and generates dissertation-ready figures.

Usage:
    python3 scripts/plot_benchmarks.py --results results/
    python3 scripts/plot_benchmarks.py --results results/ --compare results_optimised/
    python3 scripts/plot_benchmarks.py --results results/ --compare results_optimised/ --out figures/
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ── Style ──────────────────────────────────────────────────────────────────────

BLUE   = "#185FA5"
CORAL  = "#D85A30"
GRAY   = "#888780"

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#D3D1C7",
    "axes.linewidth":    0.8,
    "axes.grid":         True,
    "grid.color":        "#E8E6DF",
    "grid.linewidth":    0.6,
    "grid.linestyle":    "-",
    "xtick.color":       "#5F5E5A",
    "ytick.color":       "#5F5E5A",
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.labelsize":    10,
    "axes.labelcolor":   "#3d3d3a",
    "axes.titlesize":    11,
    "axes.titleweight":  "medium",
    "axes.titlecolor":   "#2C2C2A",
    "legend.fontsize":   9,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "#D3D1C7",
    "font.family":       "sans-serif",
    "figure.dpi":        150,
})

# ── Loader ─────────────────────────────────────────────────────────────────────

def load_results(results_dir: Path, renderer: str = None):
    results_dir = Path(results_dir)
    if not results_dir.exists():
        print(f"ERROR: results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    summaries = {}
    frames    = {}

    for json_file in sorted(results_dir.glob("*_summary.json")):
        with open(json_file) as f:
            s = json.load(f)
        scene_id = s["scene_id"]
        if renderer is None:
            renderer = s["renderer"]
        summaries[scene_id] = s

    for csv_file in sorted(results_dir.glob("*_frames.csv")):
        stem = csv_file.stem
        stem = stem[: -len("_frames")]
        for prefix in [f"{renderer}_", "naive_", "optimised_"]:
            if stem.startswith(prefix):
                scene_id = stem[len(prefix):]
                break
        else:
            scene_id = stem
        frames[scene_id] = pd.read_csv(csv_file)

    return renderer, summaries, frames


def load_stress_steps(df: pd.DataFrame, x_col: str):
    """
    Groups stress test frames by x_col (e.g. 'triangle_count' or 'draw_calls')
    to get per-step averages. Handles both single-draw-call optimised renderer
    and multi-draw-call naive renderer correctly.
    """
    # Sort by x_col so steps are in order
    df = df.sort_values(x_col).reset_index(drop=True)

    # Bin into ~20 equal steps to smooth out noise regardless of x_col cardinality
    n_bins = min(20, df[x_col].nunique())
    df["_bin"] = pd.cut(df[x_col], bins=n_bins, labels=False)

    steps = []
    for _, grp in df.groupby("_bin"):
        if grp.empty:
            continue
        steps.append({
            "x":       grp[x_col].mean(),
            "draws":   grp["draw_calls"].mean(),
            "tris":    grp["triangle_count"].mean(),
            "avg_fps": grp["fps"].mean(),
            "min_fps": grp["fps"].min(),
            "avg_ms":  grp["frame_time_ms"].mean(),
        })
    return steps

# ── Plot helpers ───────────────────────────────────────────────────────────────

def plot_frame_time(ax, df, label, color):
    ax.plot(df["frame"], df["frame_time_ms"], color=color, linewidth=0.8,
            alpha=0.6, label=label)
    roll = df["frame_time_ms"].rolling(20, center=True).mean()
    ax.plot(df["frame"], roll, color=color, linewidth=1.8, alpha=1.0,
            label=f"{label} (20-frame avg)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Frame time (ms)")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))


def plot_fps_distribution(ax, df, label, color):
    ax.hist(df["fps"], bins=40, color=color, alpha=0.75, edgecolor="white",
            linewidth=0.4, label=label)
    avg  = df["fps"].mean()
    pct1 = df["fps"].quantile(0.01)
    ax.axvline(avg,  color=color, linewidth=1.5, linestyle="--",
               label=f"Avg {avg:.0f} FPS")
    ax.axvline(pct1, color=color, linewidth=1.5, linestyle=":",
               label=f"1% low {pct1:.0f} FPS")
    ax.set_xlabel("FPS")
    ax.set_ylabel("Frames")

# ── Scene figures ──────────────────────────────────────────────────────────────

def make_static_figure(summaries, frames, renderer, compare_summaries=None,
                       compare_frames=None, compare_renderer=None, out_dir=None):
    scene = "static_high_density"
    if scene not in frames:
        print(f"  skipping static figure — no data for '{scene}'")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Scene 1: Static high-density view", fontsize=12,
                 fontweight="medium", color="#2C2C2A", y=1.01)

    plot_frame_time(axes[0], frames[scene], renderer, BLUE)
    if compare_frames and scene in compare_frames:
        plot_frame_time(axes[0], compare_frames[scene], compare_renderer, CORAL)
    axes[0].set_title("Frame time over 300 measured frames")
    axes[0].legend()

    plot_fps_distribution(axes[1], frames[scene], renderer, BLUE)
    if compare_frames and scene in compare_frames:
        plot_fps_distribution(axes[1], compare_frames[scene], compare_renderer, CORAL)
    axes[1].set_title("FPS distribution")
    axes[1].legend()

    fig.tight_layout()
    _save(fig, out_dir, "scene1_static_high_density.png")


def make_dynamic_figure(summaries, frames, renderer, compare_summaries=None,
                        compare_frames=None, compare_renderer=None, out_dir=None):
    scene = "dynamic_remesh"
    if scene not in frames:
        print(f"  skipping dynamic figure — no data for '{scene}'")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Scene 2: Dynamic remesh", fontsize=12,
                 fontweight="medium", color="#2C2C2A", y=1.01)

    plot_frame_time(axes[0], frames[scene], renderer, BLUE)
    if compare_frames and scene in compare_frames:
        plot_frame_time(axes[0], compare_frames[scene], compare_renderer, CORAL)
    axes[0].set_title("Frame time over 300 measured frames")
    axes[0].legend()

    plot_fps_distribution(axes[1], frames[scene], renderer, BLUE)
    if compare_frames and scene in compare_frames:
        plot_fps_distribution(axes[1], compare_frames[scene], compare_renderer, CORAL)
    axes[1].set_title("FPS distribution")
    axes[1].legend()

    fig.tight_layout()
    _save(fig, out_dir, "scene2_dynamic_remesh.png")


def make_stress_figure(summaries, frames, renderer, compare_summaries=None,
                       compare_frames=None, compare_renderer=None, out_dir=None):
    scene = "stress_test"
    if scene not in frames:
        print(f"  skipping stress figure — no data for '{scene}'")
        return

    df  = frames[scene]
    has_compare = compare_frames and scene in compare_frames
    cdf = compare_frames[scene] if has_compare else None

    # Choose X axis: triangle_count works for both renderers since the optimised
    # renderer always issues 1 draw call — draw_calls is useless as an X axis there.
    x_col = "triangle_count"

    steps  = load_stress_steps(df,  x_col)
    csteps = load_stress_steps(cdf, x_col) if has_compare else []

    def sv(steps, key):
        return [s[key] for s in steps]

    tris_m  = [s["tris"] / 1e6 for s in steps]
    ctris_m = [s["tris"] / 1e6 for s in csteps]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Scene 3: Stress test", fontsize=12,
                 fontweight="medium", color="#2C2C2A", y=1.01)

    # ── Row 1 ─────────────────────────────────────────────────────────────────

    # FPS vs triangles
    ax = axes[0, 0]
    ax.plot(tris_m, sv(steps, "avg_fps"), "o-", color=BLUE,
            linewidth=1.8, markersize=5, label=renderer)
    if has_compare:
        ax.plot(ctris_m, sv(csteps, "avg_fps"), "s-", color=CORAL,
                linewidth=1.8, markersize=5, label=compare_renderer)
    ax.axhline(30, color=GRAY, linewidth=1, linestyle="--", label="30 FPS floor")
    ax.set_xlabel("Triangles (millions)")
    ax.set_ylabel("Avg FPS")
    ax.set_title("FPS vs triangle count")
    ax.legend()

    # Frame time vs triangles
    ax = axes[0, 1]
    ax.plot(tris_m, sv(steps, "avg_ms"), "o-", color=BLUE,
            linewidth=1.8, markersize=5, label=renderer)
    if has_compare:
        ax.plot(ctris_m, sv(csteps, "avg_ms"), "s-", color=CORAL,
                linewidth=1.8, markersize=5, label=compare_renderer)
    ax.set_xlabel("Triangles (millions)")
    ax.set_ylabel("Avg frame time (ms)")
    ax.set_title("Frame time vs triangle count")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.legend()

    # Draw calls vs triangles — shows the key optimisation difference
    ax = axes[0, 2]
    ax.plot(tris_m, sv(steps, "draws"), "o-", color=BLUE,
            linewidth=1.8, markersize=5, label=renderer)
    if has_compare:
        ax.plot(ctris_m, sv(csteps, "draws"), "s-", color=CORAL,
                linewidth=1.8, markersize=5, label=compare_renderer)
    ax.set_xlabel("Triangles (millions)")
    ax.set_ylabel("Avg draw calls")
    ax.set_title("Draw calls vs triangle count")
    ax.legend()

    # ── Row 2 ─────────────────────────────────────────────────────────────────

    # Frame time over all measured frames
    ax = axes[1, 0]
    plot_frame_time(ax, df, renderer, BLUE)
    if has_compare:
        plot_frame_time(ax, cdf, compare_renderer, CORAL)
    ax.set_title("Frame time per frame — all steps")
    ax.legend()

    # FPS distribution
    ax = axes[1, 1]
    plot_fps_distribution(ax, df, renderer, BLUE)
    if has_compare:
        plot_fps_distribution(ax, cdf, compare_renderer, CORAL)
    ax.set_title("FPS distribution — all steps")
    ax.legend()

    # Min vs avg FPS per step
    ax = axes[1, 2]
    ax.plot(tris_m, sv(steps, "min_fps"), "o-", color=BLUE,
            linewidth=1.8, markersize=5, label=f"{renderer} min FPS")
    ax.plot(tris_m, sv(steps, "avg_fps"), "--", color=BLUE,
            linewidth=1, alpha=0.5, label=f"{renderer} avg FPS")
    if has_compare:
        ax.plot(ctris_m, sv(csteps, "min_fps"), "s-", color=CORAL,
                linewidth=1.8, markersize=5, label=f"{compare_renderer} min FPS")
        ax.plot(ctris_m, sv(csteps, "avg_fps"), "--", color=CORAL,
                linewidth=1, alpha=0.5, label=f"{compare_renderer} avg FPS")
    ax.axhline(30, color=GRAY, linewidth=1, linestyle="--", label="30 FPS floor")
    ax.set_xlabel("Triangles (millions)")
    ax.set_ylabel("FPS")
    ax.set_title("Min vs avg FPS per step")
    ax.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, out_dir, "scene3_stress_test.png")


def make_summary_table(summaries, frames, renderer, compare_summaries=None,
                       compare_frames=None, compare_renderer=None, out_dir=None):
    scenes = ["static_high_density", "dynamic_remesh", "stress_test"]
    labels = ["Static", "Dynamic", "Stress test"]
    has_compare = bool(compare_summaries)

    avg_fps = [summaries.get(s, {}).get("avg_fps", 0)         for s in scenes]
    low_fps = [summaries.get(s, {}).get("one_pct_low_fps", 0) for s in scenes]
    avg_ms  = [summaries.get(s, {}).get("avg_frame_ms", 0)    for s in scenes]

    cavg_fps = [compare_summaries.get(s, {}).get("avg_fps", 0)         for s in scenes] if has_compare else []
    clow_fps = [compare_summaries.get(s, {}).get("one_pct_low_fps", 0) for s in scenes] if has_compare else []
    cavg_ms  = [compare_summaries.get(s, {}).get("avg_frame_ms", 0)    for s in scenes] if has_compare else []

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Summary: all scenes", fontsize=12,
                 fontweight="medium", color="#2C2C2A", y=1.01)

    x = np.arange(len(labels))
    w = 0.35 if has_compare else 0.5

    def grouped_bars(ax, vals, cvals, ylabel, title):
        bars = ax.bar(x - (w / 2 if has_compare else 0), vals, w,
                      color=BLUE, label=renderer, alpha=0.85)
        ax.bar_label(bars, fmt="%.0f", fontsize=8, color="#3d3d3a", padding=2)
        if has_compare:
            cbars = ax.bar(x + w / 2, cvals, w,
                           color=CORAL, label=compare_renderer, alpha=0.85)
            ax.bar_label(cbars, fmt="%.0f", fontsize=8, color="#3d3d3a", padding=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if has_compare:
            ax.legend()

    grouped_bars(axes[0], avg_fps, cavg_fps, "FPS", "Avg FPS")
    grouped_bars(axes[1], low_fps, clow_fps, "FPS", "1% low FPS")
    grouped_bars(axes[2], avg_ms,  cavg_ms,  "ms",  "Avg frame time (ms)")

    fig.tight_layout()
    _save(fig, out_dir, "summary_comparison.png")


def _save(fig, out_dir, filename):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  Saved: {path}")
    plt.close(fig)

# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot voxel engine benchmark results")
    parser.add_argument("--results",  required=True)
    parser.add_argument("--compare",  default=None)
    parser.add_argument("--out",      default=None)
    args = parser.parse_args()

    if args.out is None:
        args.out = Path(args.results).parent / "figures"
        print(f"No --out specified, saving to: {args.out}")

    print(f"Loading: {args.results}")
    renderer, summaries, frames = load_results(args.results)
    print(f"  Renderer: {renderer}  |  Scenes: {list(summaries.keys())}")

    compare_renderer  = None
    compare_summaries = None
    compare_frames    = None

    if args.compare:
        print(f"Loading compare: {args.compare}")
        compare_renderer, compare_summaries, compare_frames = load_results(args.compare)
        print(f"  Renderer: {compare_renderer}  |  Scenes: {list(compare_summaries.keys())}")

    print("Generating figures...")
    kw = dict(
        summaries=summaries, frames=frames, renderer=renderer,
        compare_summaries=compare_summaries, compare_frames=compare_frames,
        compare_renderer=compare_renderer, out_dir=args.out,
    )
    make_static_figure(**kw)
    make_dynamic_figure(**kw)
    make_stress_figure(**kw)
    make_summary_table(**kw)
    print("Done.")


if __name__ == "__main__":
    main()