import matplotlib.pyplot as plt
import numpy as np

# ==================== CLEAN STYLE ====================
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica"],
        "font.size": 12,
        "axes.linewidth": 1.5,
        "figure.dpi": 300,
        "savefig.dpi": 600,
    }
)

# ==================== DATA ====================
sequences = ["3d1", "3d2", "3d3", "3d4", "3d5", "3d6", "3d7", "3d8"]
lengths = [20, 24, 25, 36, 46, 48, 50, 58]
best_known = [-11, -13, -9, -18, -35, -31, -34, -44]

# CP-SAT energies
cpsat_energy = [-11, -13, -9, -18, -32, -29, -30, -40]

# DQN energies
dqn_energy = [-11, -13, -9, -18, -33, -30, -32, -40]

# ==================== CREATE BAR CHART ====================
fig, ax = plt.subplots(figsize=(11, 6), facecolor="white")

x = np.arange(len(sequences))
width = 0.25

# Colors
best_color = "#4CAF50"  # Green
cp_color = "#2E86AB"  # Blue
dqn_color = "#A23B72"  # Purple
missing_color = "#E8E8E8"  # Light gray

# Prepare data
cp_bars = cpsat_energy
dqn_bars = dqn_energy

# Plot bars
ax.bar(
    x - width,
    best_known,
    width,
    label="Best Known",
    color=best_color,
    edgecolor="white",
    linewidth=2,
    alpha=0.85,
)

ax.bar(
    x,
    cp_bars,
    width,
    label="CP-SAT",
    color=cp_color,
    edgecolor="white",
    linewidth=2,
    alpha=0.85,
)

ax.bar(
    x + width,
    dqn_bars,
    width,
    label="LSTM-A",
    color=dqn_color,
    edgecolor="white",
    linewidth=2,
    alpha=0.85,
)

# Gray placeholder for missing CP-SAT
ax.bar(8, -1, width, color=missing_color, edgecolor="#CCCCCC", linewidth=1.5, alpha=0.6)

# ==================== STYLING ====================
ax.set_xlabel("Sequence (Length)", fontweight="bold", fontsize=13)
ax.set_ylabel("Energy (H-H contacts)", fontweight="bold", fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels([f"{s}\n({l})" for s, l in zip(sequences, lengths)], fontsize=11)

ax.legend(loc="lower left", frameon=True, fontsize=11, ncol=3)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.25, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()

# ==================== SAVE ====================
plt.savefig(
    "fig_energy_comparison.png", dpi=600, bbox_inches="tight", facecolor="white"
)
print("✓ Saved: fig_energy_comparison.png")
