#!/usr/bin/env python3
"""
Figure 11 Generator: Triangle Count Comparison
===============================================
Reads JSON summary files from the naive and optimised renderers
and generates a dissertation-ready bar chart comparing the average
triangle counts across different scenes.

Usage:
    python3 scripts/fig11.py
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg") # Use non-interactive backend for file generation
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Style Configuration (Dissertation Style) ──────────────────────────────────

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
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "axes.labelsize":    11,
    "axes.titlesize":    12,
    "legend.fontsize":   10,
    "legend.frameon":    True,
    "legend.edgecolor":  "#D3D1C7",
    "font.family":       "sans-serif",
})

# ── Data Loading ──────────────────────────────────────────────────────────────

def load_triangle_data(data_dir: Path):
    """
    Scans the directory recursively for *_summary.json files and extracts the average 
    triangle counts for naive and optimised renderers per scene.
    Reads benchmark_config.json to maintain the correct scene order.
    """
    data_naive = {}
    data_opt = {}
    found_scenes = set()
    
    # Scenes that skew the graph too heavily to be useful here
    EXCLUDED_SCENES = {"mega_world_stres"}
    
    # Recursively find all summary files (finds them inside /results and /results_optimised)
    summary_files = list(data_dir.rglob("*_summary.json"))
    
    for filepath in summary_files:
        try:
            with open(filepath, "r") as f:
                summary = json.load(f)
                renderer = summary.get("renderer")
                scene = summary.get("scene_id")
                tri_count = summary.get("avg_triangle_count", 0)
                
                # Skip excluded scenes
                if scene in EXCLUDED_SCENES:
                    continue
                
                if renderer and scene:
                    found_scenes.add(scene)
                    if renderer == "naive":
                        data_naive[scene] = tri_count
                    elif renderer in ["optimised", "optimized"]:
                        data_opt[scene] = tri_count
        except Exception as e:
            print(f"Warning: Failed to read {filepath}: {e}")

    # Try to find config for proper ordering
    config_path = next(data_dir.rglob("benchmark_config.json"), None)
    ordered_scenes = []
    
    if config_path:
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                # Keep order from config, but only include scenes we actually have data for
                ordered_scenes = [s.get("id") for s in config.get("scenes", []) if s.get("id") in found_scenes]
        except Exception as e:
            print(f"Warning: Failed to read {config_path}: {e}")
            
    # Add any extra scenes that were found in JSONs but weren't in the config
    for s in found_scenes:
        if s not in ordered_scenes:
            ordered_scenes.append(s)

    # Ensure default values exist so plotting doesn't break
    for s in ordered_scenes:
        if s not in data_naive: data_naive[s] = 0
        if s not in data_opt: data_opt[s] = 0
        
    return ordered_scenes, data_naive, data_opt

# ── Plotting Logic ────────────────────────────────────────────────────────────

def format_thousands(x, pos):
    """Formats y-axis labels with commas (e.g., 1,000,000)."""
    return f"{int(x):,}"

def format_bar_label(val):
    """Formats labels above bars nicely (e.g., 1.8M or 426K)."""
    if val == 0:
        return "0"
    elif val >= 1_000_000:
        return f"{val/1_000_000:.2f}M"
    elif val >= 1_000:
        return f"{val/1_000:.0f}K"
    return str(val)

def create_bar_chart(scenes, naive_data, opt_data, out_dir: Path):
    """Generates the grouped bar chart and saves it."""
    
    # Format scene names for the x-axis (e.g., "dynamic_remesh" -> "Dynamic Remesh")
    scene_labels = [s.replace("_", " ").title() for s in scenes]
    
    naive_counts = [naive_data[s] for s in scenes]
    opt_counts = [opt_data[s] for s in scenes]
    
    # Calculate dynamic figure width to prevent cramped labels (min width 8.0)
    fig_width = max(8.0, len(scenes) * 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    
    # Plotting variables
    x = np.arange(len(scenes))
    width = 0.35
    
    # Create the grouped bars
    bars_naive = ax.bar(x - width/2, naive_counts, width, label='Naive Renderer', color=BLUE, zorder=3)
    bars_opt = ax.bar(x + width/2, opt_counts, width, label='Optimised Renderer', color=CORAL, zorder=3)
    
    # Customise the axes
    ax.set_ylabel("Average Triangle Count (per frame)")
    ax.set_title("Fig 11 — Triangle Count Comparison by Scene", pad=15, fontweight="bold", color="#333333")
    ax.set_xticks(x)
    
    # Angle text labels slightly to save horizontal space
    ax.set_xticklabels(scene_labels, rotation=15, ha="right")
    
    # Format Y-axis with thousands commas
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_thousands))
    
    # Put gridlines behind bars
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='-', alpha=0.7)
    ax.xaxis.grid(False) # Turn off vertical grid lines for cleaner look on bar charts
    
    # Add data labels on top of the bars
    ax.bar_label(bars_naive, labels=[format_bar_label(v) for v in naive_counts], padding=3, fontsize=9, color=BLUE, fontweight='bold')
    ax.bar_label(bars_opt, labels=[format_bar_label(v) for v in opt_counts], padding=3, fontsize=9, color=CORAL, fontweight='bold')
    
    # Add a percentage reduction annotation safely
    for i in range(len(scenes)):
        if naive_counts[i] > 0 and opt_counts[i] > 0:
            reduction = (1 - (opt_counts[i] / naive_counts[i])) * 100
            # Place text right between the two bars
            ax.text(x[i], max(naive_counts[i], opt_counts[i]) * 0.5, 
                    f"-{reduction:.1f}%", 
                    ha='center', va='center', 
                    color='white', fontweight='bold',
                    bbox=dict(facecolor='#333333', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.3'))

    # Clean up legend and layout
    ax.legend(loc='upper right', framealpha=1.0)
    
    plt.tight_layout()
    
    # Save outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path_png = out_dir / "fig11_triangle_count.png"
    out_path_pdf = out_dir / "fig11_triangle_count.pdf"
    
    plt.savefig(out_path_png, dpi=300)
    plt.savefig(out_path_pdf)
    print(f"✅ Saved Fig 11 to {out_path_png} and {out_path_pdf}")
    plt.close(fig)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Automatically detect the project root folder (the folder containing 'scripts')
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent if script_path.parent.name == "scripts" else script_path.parent

    parser = argparse.ArgumentParser(description="Generate Fig 11 - Triangle Count Comparison")
    # Default to scanning the entire project root
    parser.add_argument("--dir", default=str(project_root), help="Directory to scan for summary JSON files")
    parser.add_argument("--out", default=str(project_root / "figures"), help="Directory to save the output figures")
    args = parser.parse_args()

    data_dir = Path(args.dir)
    out_dir = Path(args.out)

    print(f"Scanning {data_dir} for Fig 11 data...")
    scenes, naive_data, opt_data = load_triangle_data(data_dir)
    
    if not scenes:
        print("❌ Error: No valid benchmark data found. Could not find any *_summary.json files.")
        return

    print(f"Found data for scenes: {', '.join(scenes)}")
    print("Generating chart...")
    create_bar_chart(scenes, naive_data, opt_data, out_dir)

if __name__ == "__main__":
    main()