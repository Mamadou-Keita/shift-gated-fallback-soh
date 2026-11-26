#!/usr/bin/env python3
"""Graphical abstract (Elsevier: >= 1328 x 531 px, read at 13 x 5 cm). Writes PNG (300 dpi) and PDF."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE, ORANGE, INK, INK2, GRID, TINT = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#dddcd8", "#f3f2ef"
out = sys.argv[1] if len(sys.argv) > 1 else "graphical_abstract"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})

fig = plt.figure(figsize=(13.28, 5.31))
W, H = 13.28, 5.31
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, W); bg.set_ylim(0, H); bg.axis("off")


def box(x, y, w, h, title, sub=None, fill=TINT, edge=INK2, tc=INK, lw=1.0, ts=11):
    bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fill, ec=edge, lw=lw))
    if sub:
        bg.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=ts, fontweight="bold", color=tc)
        bg.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=8.5, color=tc)
    else:
        bg.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=ts, fontweight="bold", color=tc)


def arrow(x1, y1, x2, y2, color=INK2, lw=1.4, style="-|>", ls="-"):
    bg.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12, color=color, lw=lw,
                                 linestyle=ls, shrinkA=0, shrinkB=0))


# ---------------- panel 1: the input window
bg.text(0.35, 4.95, "1  Input: one partial CC charging window", fontsize=11.5, fontweight="bold", color=INK)
ax = fig.add_axes([0.04, 0.22, 0.23, 0.62])
q = np.linspace(0, 1, 400)
vA = 3.35 + 0.55 * q ** 0.55 + 0.25 * q ** 3
vB = vA + 0.05 * np.sin(2 * np.pi * (q - 0.25) / 0.5) * np.exp(-((q - 0.5) / 0.22) ** 2)
ax.plot(q, vA, color=INK, lw=2, label="chemistry A")
ax.plot(q, vB, color=INK2, lw=1.5, ls="--", label="chemistry B")
ax.axhspan(3.70, 4.00, color=BLUE, alpha=0.10, lw=0)
ax.text(0.02, 3.985, "fixed voltage window", fontsize=8.5, color=BLUE, va="top", ha="left")
q0, q1 = np.interp(3.70, vA, q), np.interp(4.00, vA, q)
ax.annotate("", xy=(q0, 4.07), xytext=(q1, 4.07), arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2))
ax.plot([q0, q0], [3.70, 4.07], color=INK, lw=0.6, ls=":"); ax.plot([q1, q1], [4.00, 4.07], color=INK, lw=0.6, ls=":")
ax.text((q0 + q1) / 2, 4.10, "amplitude \u0394Q/Q_nom: transfers", ha="center", va="bottom", fontsize=8.3)
ax.annotate("shape: chemistry-specific", xy=(0.47, np.interp(0.47, q, vB)), xytext=(0.04, 3.33),
            fontsize=8.3, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.set_xlabel("Charge Q / Q_nom"); ax.set_ylabel("Voltage (V)")
ax.set_ylim(3.30, 4.30); ax.set_xlim(0, 1)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(labelsize=8); ax.legend(frameon=False, fontsize=8, loc="lower right")

# ---------------- panel 2: the envelope
bg.text(4.25, 4.95, "2  Shift-gated fallback (any SOH architecture)", fontsize=11.5, fontweight="bold", color=INK)
arrow(3.75, 2.85, 4.25, 3.75); arrow(3.75, 2.85, 4.25, 2.45); arrow(3.75, 2.85, 4.25, 1.15)
box(4.25, 3.35, 2.75, 0.85, "Expert model", "accurate, fragile under shift", fill="#fbe7df", edge=ORANGE)
box(4.25, 2.05, 2.75, 0.85, "Fallback", "ridge on ΔQ/Q_nom + T, transferable", fill="#ececea", edge=INK2)
box(4.25, 0.70, 2.75, 0.95, "Shift score on shape", "Mahalanobis distance,\ncalibrated on held-out cells", fill="#fff",
    edge=INK2)
box(7.4, 0.80, 1.65, 0.75, "weight w", "1 in domain → 0 if shifted", fill="#fff", edge=INK2, ts=10)
arrow(7.0, 1.18, 7.4, 1.18)
box(7.45, 2.35, 1.55, 1.55, "", fill=BLUE, edge=BLUE)
bg.text(8.225, 3.45, "Estimate", ha="center", fontsize=11, fontweight="bold", color="white")
bg.text(8.225, 2.95, "ŷ = ŷ_b + w(ŷ_X − ŷ_b)", ha="center", fontsize=9.2, color="white")
bg.text(8.225, 2.6, "one per cell, causal", ha="center", fontsize=8, color="white")
arrow(7.0, 3.78, 7.45, 3.45); arrow(7.0, 2.48, 7.45, 2.75); arrow(8.225, 1.55, 8.225, 2.35)

# ---------------- panel 3: the evidence
bg.text(9.45, 4.95, "3  Result: close to the better of the two", fontsize=11.5, fontweight="bold", color=INK)
ax3 = fig.add_axes([0.745, 0.24, 0.235, 0.58])
labels = ["Expert\nalone", "Fallback\nalone", "Shift-gated\nfallback"]
vals = [1.16, 0.79, 0.10]
bars = ax3.bar(range(3), vals, color=[ORANGE, INK2, BLUE], width=0.62)
for i, v in enumerate(vals):
    ax3.text(i, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK)
ax3.set_xticks(range(3), labels, fontsize=8.6)
ax3.set_ylabel("Mean regret (SOH points)", fontsize=8.6)
ax3.set_ylim(0, 1.45)
ax3.grid(axis="y", color=GRID, lw=0.6); ax3.set_axisbelow(True)
for s in ("top", "right"):
    ax3.spines[s].set_visible(False)
ax3.tick_params(axis="y", labelsize=8)
bg.text(11.1, 0.42, "regret = MAE − MAE of the better of {expert, fallback}\n"
        "3 public datasets · 12 architectures · 11 shift protocols · 3 seeds",
        ha="center", va="center", fontsize=8, color=INK2)
bg.text(11.1, 4.62, "Unseen chemistry: 1.0–10.1 → 0.7–1.0 points; no cost in-domain",
        ha="center", va="center", fontsize=8.6, color=INK)

fig.savefig(out + ".png", dpi=300)
fig.savefig(out + ".pdf")
print(out + ".png")
