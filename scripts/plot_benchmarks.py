#!/usr/bin/env python3
"""
Voxel Engine Benchmark Visualiser
===================================
Reads CSV/JSON results produced by renderer-naive (and later renderer-optimised)
and generates dissertation-ready figures.

Usage:
    # Single renderer (naive baseline):
    python3 scripts/plot_benchmarks.py --results results/

    # Comparison between naive and optimised:
    python3 scripts/plot_benchmarks.py --results results/ --compare results_optimised/

    # Save to specific directory:
    python3 scripts/plot_benchmarks.py --results results/ --out figures/
"""

import argparse
import json
import sys
from itertools import groupby
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ── Style ──────────────────────────────────────────────────────────────────────

BLUE   = "#185FA5"
CORAL  = "#D85A30"
GREEN  = "#3B6D11"
GRAY   = "#888780"
LIGHT  = "#E6F1FB"
LIGHT2 = "#FAECE7"

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
    """
    Loads all benchmark CSVs and JSONs from results_dir.
    Returns a dict keyed by scene_id.
    Auto-detects renderer name from filenames if not given.
    """
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
        # Infer scene_id from filename: {renderer}_{scene_id}_frames.csv
        stem = csv_file.stem  # e.g. naive_static_high_density_frames
        # Strip trailing _frames
        stem = stem[: -len("_frames")]
        # Strip leading renderer name
        for prefix in [f"{renderer}_", "naive_", "optimised_"]:
            if stem.startswith(prefix):
                scene_id = stem[len(prefix):]
                break
        else:
            scene_id = stem
        frames[scene_id] = pd.read_csv(csv_file)

    return renderer, summaries, frames


def load_stress_steps(df: pd.DataFrame):
    """Groups stress test frames by draw_calls to get per-step averages."""
    steps = []
    for dc, grp in groupby(df.itertuples(), key=lambda r: r.draw_calls):
        rows = list(grp)
        fps_vals = [r.fps for r in rows]
        ms_vals  = [r.frame_time_ms for r in rows]
        steps.append({
            "draws":   dc,
            "tris":    rows[0].triangle_count,
            "avg_fps": np.mean(fps_vals),
            "min_fps": np.min(fps_vals),
            "avg_ms":  np.mean(ms_vals),
        })
    return steps

# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_frame_time(ax, df: pd.DataFrame, label: str, color: str):
    """Line plot of frame time over the measured frames."""
    ax.plot(df["frame"], df["frame_time_ms"], color=color, linewidth=0.8,
            alpha=0.85, label=label)
    # Rolling mean overlay
    roll = df["frame_time_ms"].rolling(20, center=True).mean()
    ax.plot(df["frame"], roll, color=color, linewidth=1.8, alpha=1.0,
            label=f"{label} (20-frame avg)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Frame time (ms)")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))


def plot_fps_distribution(ax, df: pd.DataFrame, label: str, color: str):
    """Histogram of FPS values across all measured frames."""
    ax.hist(df["fps"], bins=40, color=color, alpha=0.75, edgecolor="white",
            linewidth=0.4, label=label)
    # Mark avg and 1% low
    avg = df["fps"].mean()
    pct1 = df["fps"].quantile(0.01)
    ax.axvline(avg,  color=color, linewidth=1.5, linestyle="--",
               label=f"Avg {avg:.0f} FPS")
    ax.axvline(pct1, color=color, linewidth=1.5, linestyle=":",
               label=f"1% low {pct1:.0f} FPS")
    ax.set_xlabel("FPS")
    ax.set_ylabel("Frames")


def make_static_figure(summaries, frames, renderer, compare_summaries=None,
                       compare_frames=None, compare_renderer=None, out_dir=None):
    scene = "static_high_density"
    if scene not in frames:
        print(f"  skipping static figure — no data for '{scene}'")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Scene 1: Static high-density view", fontsize=12, fontweight="medium",
                 color="#2C2C2A", y=1.01)

    # Frame time
    ax = axes[0]
    plot_frame_time(ax, frames[scene], renderer, BLUE)
    if compare_frames and scene in compare_frames:
        plot_frame_time(ax, compare_frames[scene], compare_renderer, CORAL)
    ax.set_title("Frame time over 300 measured frames")
    ax.legend()

    # FPS distribution
    ax = axes[1]
    plot_fps_distribution(ax, frames[scene], renderer, BLUE)
    if compare_frames and scene in compare_frames:
        plot_fps_distribution(ax, compare_frames[scene], compare_renderer, CORAL)
    ax.set_title("FPS distribution")
    ax.legend()

    fig.tight_layout()
    _save(fig, out_dir, "scene1_static_high_density.png")


def make_dynamic_figure(summaries, frames, renderer, compare_summaries=None,
                        compare_frames=None, compare_renderer=None, out_dir=None):
    scene = "dynamic_remesh"
    if scene not in frames:
        print(f"  skipping dynamic figure — no data for '{scene}'")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Scene 2: Dynamic remesh", fontsize=12, fontweight="medium",
                 color="#2C2C2A", y=1.01)

    ax = axes[0]
    plot_frame_time(ax, frames[scene], renderer, BLUE)
    if compare_frames and scene in compare_frames:
        plot_frame_time(ax, compare_frames[scene], compare_renderer, CORAL)
    ax.set_title("Frame time over 300 measured frames")
    ax.legend()

    ax = axes[1]
    plot_fps_distribution(ax, frames[scene], renderer, BLUE)
    if compare_frames and scene in compare_frames:
        plot_fps_distribution(ax, compare_frames[scene], compare_renderer, CORAL)
    ax.set_title("FPS distribution")
    ax.legend()

    fig.tight_layout()
    _save(fig, out_dir, "scene2_dynamic_remesh.png")


def make_stress_figure(summaries, frames, renderer, compare_summaries=None,
                       compare_frames=None, compare_renderer=None, out_dir=None):
    scene = "stress_test"
    if scene not in frames:
        print(f"  skipping stress figure — no data for '{scene}'")
        return

    steps = load_stress_steps(frames[scene])
    draws = [s["draws"]   for s in steps]
    fps   = [s["avg_fps"] for s in steps]
    ms    = [s["avg_ms"]  for s in steps]
    tris  = [s["tris"] / 1_000_000 for s in steps]

    has_compare = compare_frames and scene in compare_frames
    if has_compare:
        csteps = load_stress_steps(compare_frames[scene])
        cdraws = [s["draws"]   for s in csteps]
        cfps   = [s["avg_fps"] for s in csteps]
        cms    = [s["avg_ms"]  for s in csteps]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Scene 3: Stress test — FPS vs draw call count",
                 fontsize=12, fontweight="medium", color="#2C2C2A", y=1.01)

    # FPS vs draw calls
    ax = axes[0]
    ax.plot(draws, fps, "o-", color=BLUE, linewidth=1.8, markersize=5,
            label=renderer)
    if has_compare:
        ax.plot(cdraws, cfps, "s-", color=CORAL, linewidth=1.8, markersize=5,
                label=compare_renderer)
    ax.axhline(30, color=GRAY, linewidth=1, linestyle="--", label="30 FPS floor")
    ax.set_xlabel("Draw calls")
    ax.set_ylabel("Avg FPS")
    ax.set_title("FPS vs draw calls")
    ax.legend()

    # Frame time vs draw calls
    ax = axes[1]
    ax.plot(draws, ms, "o-", color=BLUE, linewidth=1.8, markersize=5,
            label=renderer)
    if has_compare:
        ax.plot(cdraws, cms, "s-", color=CORAL, linewidth=1.8, markersize=5,
                label=compare_renderer)
    ax.set_xlabel("Draw calls")
    ax.set_ylabel("Avg frame time (ms)")
    ax.set_title("Frame time vs draw calls")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.legend()

    # FPS vs triangle count
    ax = axes[2]
    ax.plot(tris, fps, "o-", color=BLUE, linewidth=1.8, markersize=5,
            label=renderer)
    if has_compare:
        ctris = [s["tris"] / 1_000_000 for s in csteps]
        ax.plot(ctris, cfps, "s-", color=CORAL, linewidth=1.8, markersize=5,
                label=compare_renderer)
    ax.axhline(30, color=GRAY, linewidth=1, linestyle="--", label="30 FPS floor")
    ax.set_xlabel("Triangles (millions)")
    ax.set_ylabel("Avg FPS")
    ax.set_title("FPS vs triangle count")
    ax.legend()

    fig.tight_layout()
    _save(fig, out_dir, "scene3_stress_test.png")


def make_summary_table(summaries, frames, renderer, compare_summaries=None,
                       compare_frames=None, compare_renderer=None, out_dir=None):
    """Bar chart comparing key metrics side by side across all three scenes."""
    scenes = ["static_high_density", "dynamic_remesh", "stress_test"]
    labels = ["Static", "Dynamic", "Stress test"]

    has_compare = bool(compare_summaries)

    avg_fps   = [summaries.get(s, {}).get("avg_fps", 0)         for s in scenes]
    low_fps   = [summaries.get(s, {}).get("one_pct_low_fps", 0) for s in scenes]
    avg_ms    = [summaries.get(s, {}).get("avg_frame_ms", 0)    for s in scenes]

    if has_compare:
        cavg_fps  = [compare_summaries.get(s, {}).get("avg_fps", 0)         for s in scenes]
        clow_fps  = [compare_summaries.get(s, {}).get("one_pct_low_fps", 0) for s in scenes]
        cavg_ms   = [compare_summaries.get(s, {}).get("avg_frame_ms", 0)    for s in scenes]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Summary: all scenes", fontsize=12, fontweight="medium",
                 color="#2C2C2A", y=1.01)

    x = np.arange(len(labels))
    w = 0.35 if has_compare else 0.5

    def grouped_bars(ax, vals, cvals, ylabel, title):
        bars = ax.bar(x - (w/2 if has_compare else 0), vals, w, color=BLUE,
                      label=renderer, alpha=0.85)
        ax.bar_label(bars, fmt="%.0f", fontsize=8, color="#3d3d3a", padding=2)
        if has_compare:
            cbars = ax.bar(x + w/2, cvals, w, color=CORAL,
                           label=compare_renderer, alpha=0.85)
            ax.bar_label(cbars, fmt="%.0f", fontsize=8, color="#3d3d3a", padding=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if has_compare:
            ax.legend()

    grouped_bars(axes[0], avg_fps,  cavg_fps if has_compare else None,
                 "FPS", "Avg FPS")
    grouped_bars(axes[1], low_fps,  clow_fps if has_compare else None,
                 "FPS", "1% low FPS")
    grouped_bars(axes[2], avg_ms,   cavg_ms  if has_compare else None,
                 "ms", "Avg frame time (ms)")

    fig.tight_layout()
    _save(fig, out_dir, "summary_comparison.png")


def _save(fig, out_dir, filename):
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / filename
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"  Saved: {path}")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot voxel engine benchmark results")
    parser.add_argument("--results",  required=True,
                        help="Path to results directory (e.g. results/)")
    parser.add_argument("--compare",  default=None,
                        help="Path to a second results directory to overlay for comparison")
    parser.add_argument("--out",      default=None,
                        help="Output directory for saved PNGs (default: show interactively)")
    args = parser.parse_args()

    print(f"Loading: {args.results}")
    renderer, summaries, frames = load_results(args.results)
    print(f"  Renderer: {renderer}")
    print(f"  Scenes found: {list(summaries.keys())}")

    compare_renderer   = None
    compare_summaries  = None
    compare_frames     = None

    if args.compare:
        print(f"Loading compare: {args.compare}")
        compare_renderer, compare_summaries, compare_frames = load_results(args.compare)
        print(f"  Renderer: {compare_renderer}")

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