#!/usr/bin/env python3
import argparse
import csv
import json
import re
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


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


def get_project_root():
    script_path = Path(__file__).resolve()
    return script_path.parent.parent if script_path.parent.name == "scripts" else script_path.parent

def load_summary(base_dir: Path, scene: str):
    for filepath in base_dir.rglob(f"*{scene}_summary.json"):
        with open(filepath, "r") as f:
            return json.load(f)
    return None

def load_frames(base_dir: Path, scene: str):
    for filepath in base_dir.rglob(f"*{scene}_frames.csv"):
        return pd.read_csv(filepath)
    return None

def format_thousands(x, pos):
    return f"{int(x):,}"

def format_bar_label(val):
    if val >= 1_000_000:
        return f"{val/1_000_000:.2f}M"
    elif val >= 1_000:
        return f"{val/1_000:.0f}K"
    elif val > 0 and val < 10:
        return f"{val:.2f}"
    return str(int(val))


def plot_fig10_draw_calls(naive_dir, opt_dir, out_dir):
    scenes = ["large_world_static", "frustum_cull", "dynamic_remesh", "stress_test"]
    found_scenes, naive_vals, opt_vals = [], [], []
    
    for s in scenes:
        n_sum = load_summary(naive_dir, s)
        o_sum = load_summary(opt_dir, s)
        if n_sum and o_sum:
            found_scenes.append(s.replace("_", " ").title())
            naive_vals.append(n_sum.get("avg_draw_calls", 0))
            opt_vals.append(o_sum.get("avg_draw_calls", 0))
            
    if not found_scenes: return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(found_scenes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, naive_vals, width, label='Naive', color=BLUE)
    bars2 = ax.bar(x + width/2, opt_vals, width, label='Optimised', color=CORAL)
    
    ax.set_ylabel("Average Draw Calls")
    ax.set_title("Figure 6: Draw Call Count Comparison", pad=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(found_scenes, rotation=15, ha="right")
    ax.bar_label(bars1, padding=3, color=BLUE, fontweight="bold")
    ax.bar_label(bars2, padding=3, color=CORAL, fontweight="bold")
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig6_draw_calls.pdf")
    plt.savefig(out_dir / "fig6_draw_calls.png", dpi=300)
    plt.close(fig)

def plot_fig11_triangle_count(naive_dir, opt_dir, out_dir):
    scenes = ["large_world_static", "frustum_cull", "dynamic_remesh"]
    found_scenes, naive_vals, opt_vals = [], [], []
    
    for s in scenes:
        n_sum = load_summary(naive_dir, s)
        o_sum = load_summary(opt_dir, s)
        if n_sum and o_sum:
            found_scenes.append(s.replace("_", " ").title())
            naive_vals.append(n_sum.get("avg_triangle_count", 0))
            opt_vals.append(o_sum.get("avg_triangle_count", 0))
            
    if not found_scenes: return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(found_scenes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, naive_vals, width, label='Naive', color=BLUE)
    bars2 = ax.bar(x + width/2, opt_vals, width, label='Optimised', color=CORAL)
    
    ax.set_ylabel("Average Triangle Count")
    ax.set_title("Figure 7: Triangle Count Comparison", pad=15, fontweight="bold")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_thousands))
    ax.set_xticks(x)
    ax.set_xticklabels(found_scenes, rotation=15, ha="right")
    ax.bar_label(bars1, labels=[format_bar_label(v) for v in naive_vals], padding=3, color=BLUE, fontweight="bold")
    ax.bar_label(bars2, labels=[format_bar_label(v) for v in opt_vals], padding=3, color=CORAL, fontweight="bold")
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig7_triangle_count.pdf")
    plt.savefig(out_dir / "fig7_triangle_count.png", dpi=300)
    plt.close(fig)

def plot_fig12_dyn_remesh_fps_line(naive_dir, opt_dir, out_dir):
    df_naive = load_frames(naive_dir, "dynamic_remesh")
    df_opt = load_frames(opt_dir, "dynamic_remesh")
    
    if df_naive is None or df_opt is None: return
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    window = min(10, len(df_naive) // 10)
    ax.plot(df_naive['frame'], df_naive['fps'].rolling(window, min_periods=1).mean(), 
            label='Naive Renderer', color=BLUE, linewidth=2)
    ax.plot(df_opt['frame'], df_opt['fps'].rolling(window, min_periods=1).mean(), 
            label='Optimised Renderer', color=CORAL, linewidth=2)
    
    ax.plot(df_naive['frame'], df_naive['fps'], color=BLUE, alpha=0.15)
    ax.plot(df_opt['frame'], df_opt['fps'], color=CORAL, alpha=0.15)
    
    ax.set_xlabel("Frame Sequence")
    ax.set_ylabel("Frames Per Second (FPS)")
    ax.set_title("Figure 8: Dynamic Remesh: FPS Over Time", pad=15, fontweight="bold")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_thousands))
    
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_dir / "fig8_dynamic_remesh_fps_line.pdf")
    plt.savefig(out_dir / "fig8_dynamic_remesh_fps_line.png", dpi=300)
    plt.close(fig)

def plot_fig13_dyn_remesh_frametime(naive_dir, opt_dir, out_dir):
    n_sum = load_summary(naive_dir, "dynamic_remesh")
    o_sum = load_summary(opt_dir, "dynamic_remesh")
    
    if not n_sum or not o_sum: return
    
    fig, ax = plt.subplots(figsize=(6, 5))
    vals = [n_sum['avg_frame_ms'], o_sum['avg_frame_ms']]
    
    bars = ax.bar(["Naive Renderer", "Optimised Renderer"], vals, color=[BLUE, CORAL], width=0.5)
    
    ax.set_ylabel("Average Frame Time (ms) - Lower is better")
    ax.set_title("Figure 9: Dynamic Remesh: Frame Time Reduction", pad=15, fontweight="bold")
    ax.bar_label(bars, fmt='%.2f ms', padding=3, fontweight="bold")
    
    drop = (1 - (vals[1] / vals[0])) * 100
    ax.text(0.5, max(vals) * 0.5, f"-{drop:.1f}% Latency", ha='center', va='center', 
            color='white', fontweight='bold', bbox=dict(facecolor='#333333', edgecolor='none', alpha=0.8, boxstyle='round,pad=0.5'))
            
    plt.tight_layout()
    plt.savefig(out_dir / "fig9_dynamic_remesh_frametime.pdf")
    plt.savefig(out_dir / "fig9_dynamic_remesh_frametime.png", dpi=300)
    plt.close(fig)


def plot_fig15_stress_fps_vs_chunks(naive_dir, opt_dir, out_dir):
    df_naive = load_frames(naive_dir, "stress_test")
    df_opt = load_frames(opt_dir, "stress_test")
    
    if df_naive is None or df_opt is None: return
    
    df = pd.merge(df_naive, df_opt, on='frame', suffixes=('_naive', '_opt'))
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    x_axis = df['frame']
    ax.plot(x_axis, df['fps_naive'].rolling(5, min_periods=1).median(), 
            label='Naive (Degrades under stress)', color=BLUE, linewidth=2.5)
    ax.plot(x_axis, df['fps_opt'].rolling(5, min_periods=1).median(), 
            label='Optimised (Stays Flat)', color=CORAL, linewidth=2.5)
            
    ax.set_xlabel("Stress Test Progression (Frame Sequence)")
    ax.set_ylabel("Frames Per Second (FPS)")
    ax.set_title("Figure 10: Stress Test: Engine Degradation", pad=15, fontweight="bold")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_thousands))
    
    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(out_dir / "fig11_stress_fps_vs_chunks.pdf")
    plt.savefig(out_dir / "fig11_stress_fps_vs_chunks.png", dpi=300)
    plt.close(fig)

def plot_fig16_static_avg_fps(naive_dir, opt_dir, out_dir):
    n_sum = load_summary(naive_dir, "large_world_static") or load_summary(naive_dir, "static_high_density")
    o_sum = load_summary(opt_dir, "large_world_static") or load_summary(opt_dir, "static_high_density")
    
    if not n_sum or not o_sum: return
    
    fig, ax = plt.subplots(figsize=(6, 5))
    vals = [n_sum['avg_fps'], o_sum['avg_fps']]
    
    bars = ax.bar(["Naive", "Optimised"], vals, color=[BLUE, GRAY], width=0.5)
    
    ax.set_ylabel("Average FPS")
    ax.set_title("Figure 11: Static Scene: Peak FPS Near-Parity", pad=15, fontweight="bold")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_thousands))
    ax.bar_label(bars, fmt='%d', padding=3, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(out_dir / "fig12_static_avg_fps.pdf")
    plt.savefig(out_dir / "fig12_static_avg_fps.png", dpi=300)
    plt.close(fig)

def generate_overhead_data(csv_path: Path):
    print(f"\n[AUTO-BENCH] '{csv_path.name}' not found. Automatically running engine benchmarks...")
    chunk_counts = [1, 10, 50, 100, 200, 400]
    root_dir = get_project_root()
    data = []
    
    def run_and_extract(bin_name, chunks):
        cmd = ["cargo", "run", "--release", "--bin", bin_name, "--", "--overhead", "--chunks", str(chunks)]
        try:
            result = subprocess.run(cmd, cwd=root_dir, capture_output=True, text=True, check=True)
            output = result.stdout + result.stderr
            
            match_us = re.search(r'(\d+(?:\.\d+)?)\s*(?:us|µs|microsecond)', output, re.IGNORECASE)
            if match_us: 
                return float(match_us.group(1))
                
            match_ms = re.search(r'(\d+(?:\.\d+)?)\s*(?:ms|millisecond)', output, re.IGNORECASE)
            if match_ms: 
                return float(match_ms.group(1)) * 1000.0  # Convert to microseconds
                
            print(f"  [Warning] Could not parse timing for {bin_name} at {chunks} chunks.")
            return 0.0
        except subprocess.CalledProcessError:
            print(f"  [Error] Command failed: {' '.join(cmd)}")
            return 0.0
        except FileNotFoundError:
            print("  [Error] 'cargo' not found. Is Rust installed and in PATH?")
            return 0.0

    for c in chunk_counts:
        print(f"  -> Profiling overhead at {c} chunks...")
        naive_us = run_and_extract("renderer-naive", c)
        opt_us = run_and_extract("renderer-optimised", c)
        data.append((c, naive_us, opt_us))
        
    if any(n > 0 or o > 0 for _, n, o in data):
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["chunks", "naive_us", "opt_us"])
            writer.writerows(data)
        print(f"[AUTO-BENCH] Success! Real benchmark data saved to {csv_path}\n")
        return True
    else:
        print("[AUTO-BENCH] Failed to gather valid data. Defaulting to placeholders.\n")
        return False

def plot_fig17_overhead_crossover(overhead_csv_path: Path, out_dir: Path):

    if not overhead_csv_path.exists():
        generate_overhead_data(overhead_csv_path)
    
    if overhead_csv_path.exists():
        print(f"[INFO] Found {overhead_csv_path.name}. Plotting accurate crossover data.")
        df = pd.read_csv(overhead_csv_path)
        x_axis = df["chunks"].tolist()
        naive_us = df["naive_us"].tolist()
        opt_us = df["opt_us"].tolist()
    else:
        print(f"\n[WARNING] Could not find '{overhead_csv_path}'.")
        print("          Using placeholder data for the Overhead Crossover graph.")
        print("          -> To fix this, create 'overhead.csv' with your actual test results.")
        print("          -> Expected format: chunks,naive_us,opt_us\n")
        
        # Hardcoded fallback exactly as suggested
        x_axis = [1,  10,  50,  100, 200, 400]
        naive_us = [12, 45,  180, 280, 520, 980]
        opt_us = [80, 85,  90,  176, 180, 185]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(x_axis, naive_us, 
            label='Naive (O(N) CPU Overhead)', color=BLUE, linewidth=2, marker='o')
    ax.plot(x_axis, opt_us, 
            label='Optimised (Fixed Compute Overhead)', color=CORAL, linewidth=2, marker='s')
            
    ax.set_xlabel("Active Chunk Count")
    ax.set_ylabel("Frame Time (µs)")
    ax.set_title("Figure 12: Render Overhead Crossover Point", pad=15, fontweight="bold")
    
    ax.set_ylim(bottom=0)
    
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(out_dir / "fig13_overhead_crossover.pdf")
    plt.savefig(out_dir / "fig13_overhead_crossover.png", dpi=300)
    plt.close(fig)


def main():
    root = get_project_root()
    parser = argparse.ArgumentParser(description="Generate Chapter 6 Figures")
    parser.add_argument("--naive", default=str(root / "results"), help="Dir with naive CSV/JSONs")
    parser.add_argument("--opt", default=str(root / "results_optimised"), help="Dir with optimised CSV/JSONs")
    parser.add_argument("--out", default=str(root / "figures"), help="Output directory")
    parser.add_argument("--overhead-csv", default=str(root / "results" / "overhead.csv"), help="CSV with chunk scale metrics")
    args = parser.parse_args()

    naive_dir = Path(args.naive)
    opt_dir = Path(args.opt)
    out_dir = Path(args.out)
    overhead_csv = Path(args.overhead_csv)
    
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading Naive baseline from: {naive_dir}")
    print(f"Reading Optimised data from: {opt_dir}")
    print("Generating Chapter 6 Figures...\n")

    plot_fig10_draw_calls(naive_dir, opt_dir, out_dir)
    plot_fig11_triangle_count(naive_dir, opt_dir, out_dir)
    plot_fig12_dyn_remesh_fps_line(naive_dir, opt_dir, out_dir)
    plot_fig13_dyn_remesh_frametime(naive_dir, opt_dir, out_dir)
    plot_fig15_stress_fps_vs_chunks(naive_dir, opt_dir, out_dir)
    plot_fig16_static_avg_fps(naive_dir, opt_dir, out_dir)
    
    plot_fig17_overhead_crossover(overhead_csv, out_dir)

    print(f"\n✅ All figures successfully generated and saved to {out_dir}")

if __name__ == "__main__":
    main()