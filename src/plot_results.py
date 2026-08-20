"""Generates publication-quality time-series graphs of crowd dynamics and moving averages."""
import os
import sys
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless / script execution
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_CSV = str(ROOT_DIR / "data" / "outputs" / "log.csv")
OUTPUT_PLOT = str(ROOT_DIR / "data" / "outputs" / "count_moving_average.png")


def generate_plot(csv_path=LOG_CSV, output_png=OUTPUT_PLOT):
    if not os.path.exists(csv_path):
        print(f"[Plot] Error: Log file not found: {csv_path}")
        return False

    with open(csv_path, mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("[Plot] Warning: CSV is empty.")
        return False

    time_sec = [float(r.get("time_sec", i * 0.25)) for i, r in enumerate(rows)]
    count_raw = [float(r.get("count", 0)) for r in rows]
    count_sma = [float(r.get("count_sma", r.get("count", 0))) for r in rows]
    risk_sma  = [float(r.get("risk_score_sma", r.get("risk_score_5s", 0.0))) for r in rows]
    p_sma     = [float(r.get("max_pressure_sma", r.get("max_pressure", 0.0))) for r in rows]
    dens_sma  = [float(r.get("peak_density_sma", r.get("peak_density", 0.0))) for r in rows]
    alerts    = [r.get("alert_name", "NORMAL") for r in rows]

    # Style configuration
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True, dpi=150)
    fig.patch.set_facecolor("#0F172A")

    for ax in (ax1, ax2, ax3):
        ax.set_facecolor("#1E293B")
        ax.tick_params(colors="#94A3B8", labelsize=10)
        ax.grid(True, linestyle="--", alpha=0.25, color="#64748B")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    # ── Subplot 1: Crowd Count & Moving Average ───────────────────────
    ax1.plot(time_sec, count_raw, color="#38BDF8", alpha=0.35, linewidth=1.0, linestyle=":", label="Instantaneous Count")
    ax1.plot(time_sec, count_sma, color="#0284C7", linewidth=2.4, label="Count (Moving Average)")
    ax1.fill_between(time_sec, count_raw, count_sma, color="#38BDF8", alpha=0.10)

    # Annotate peak and median
    max_idx = int(np.argmax(count_sma))
    ax1.scatter([time_sec[max_idx]], [count_sma[max_idx]], color="#F43F5E", s=40, zorder=5)
    ax1.annotate(f"Peak SMA: {count_sma[max_idx]:.1f}",
                 xy=(time_sec[max_idx], count_sma[max_idx]),
                 xytext=(time_sec[max_idx] + 1.0, count_sma[max_idx] + 3.0),
                 color="#FDA4AF", fontsize=9, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#F43F5E", lw=1.2))

    ax1.set_ylabel("Crowd Count (Heads)", color="#E2E8F0", fontsize=11, fontweight="bold")
    ax1.set_title("Crowd Dynamics — Moving Average Analysis & Risk Forecasting",
                  color="#F8FAFC", fontsize=14, fontweight="bold", pad=12)
    ax1.legend(loc="upper right", facecolor="#1E293B", edgecolor="#334155", labelcolor="#E2E8F0")

    # ── Subplot 2: Stampede Risk Score (Moving Average) ───────────────
    # Risk zones
    ax2.axhspan(0.00, 0.30, color="#10B981", alpha=0.12, label="NORMAL (<30%)")
    ax2.axhspan(0.30, 0.55, color="#FBBF24", alpha=0.12, label="WATCH (30-55%)")
    ax2.axhspan(0.55, 0.80, color="#FB923C", alpha=0.12, label="WARNING (55-80%)")
    ax2.axhspan(0.80, 1.00, color="#EF4444", alpha=0.15, label="CRITICAL (>80%)")

    ax2.plot(time_sec, risk_sma, color="#F59E0B", linewidth=2.6, label="Risk Score (Moving Average)")
    ax2.set_ylabel("Stampede Risk (SMA)", color="#E2E8F0", fontsize=11, fontweight="bold")
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(loc="upper right", facecolor="#1E293B", edgecolor="#334155", labelcolor="#E2E8F0", ncol=2)

    # ── Subplot 3: Pressure & Density Moving Averages ─────────────────
    line1 = ax3.plot(time_sec, [p / 1000.0 for p in p_sma], color="#A855F7", linewidth=2.0, label="Pressure (SMA, k)")
    ax3.set_ylabel("Crowd Pressure (k)", color="#C084FC", fontsize=11, fontweight="bold")

    ax3_twin = ax3.twinx()
    ax3_twin.tick_params(colors="#94A3B8", labelsize=10)
    for spine in ax3_twin.spines.values():
        spine.set_color("#334155")
    line2 = ax3_twin.plot(time_sec, dens_sma, color="#F472B6", linewidth=2.0, linestyle="--", label="Peak Density (SMA)")
    ax3_twin.set_ylabel("Peak Density (SMA)", color="#F472B6", fontsize=11, fontweight="bold")

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc="upper right", facecolor="#1E293B", edgecolor="#334155", labelcolor="#E2E8F0")

    ax3.set_xlabel("Elapsed Time (seconds)", color="#E2E8F0", fontsize=11, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    plt.savefig(output_png, facecolor=fig.get_facecolor(), dpi=150)
    plt.close()
    print(f"[Plot] Graph successfully generated and saved -> {output_png}")
    return True


# Alias for seamless compatibility
plot_dynamics = generate_plot


if __name__ == "__main__":
    generate_plot()

