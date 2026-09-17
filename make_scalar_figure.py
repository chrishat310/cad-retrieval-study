"""Generate figures/scalar_features_explainer.png — what each scalar group measures.

Four panels, one per group of the 12-scalar block:
A face-type fractions (two contrasting parts)  B sphericity spectrum (rendered)
C bbox aspect-ratio map (all 1008 parts)       D modelling complexity (log face count)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import features

BLUE, ORANGE, GRAY = "#3572b0", "#e07b39", "#6b7280"
TYPES = ["plane", "cyl", "cone", "sphere", "torus", "bezier", "bspline", "other"]

z = np.load("cache/features_v0.npz", allow_pickle=True)
stems, SC, DISP = z["stems"], z["scalars"], z["display"]
frac, spher = SC[:, :8], SC[:, 8]
ba, ca, logf = SC[:, 9], SC[:, 10], SC[:, 11]


def render(ax, stem, color=BLUE):
    m = features.shape_to_trimesh(features.load_shape(f"fusion360subset/step/{stem}.stp"))
    v, t = np.array(m.vertices), np.array(m.faces)
    ax.add_collection3d(Poly3DCollection(v[t], facecolor=color, edgecolor="none", alpha=0.95))
    lo, hi = v.min(0), v.max(0)
    ctr, rad = (lo + hi) / 2, (hi - lo).max() / 2
    for dim, cc in zip("xyz", ctr):
        getattr(ax, f"set_{dim}lim")(cc - rad, cc + rad)
    ax.set_axis_off()


fig = plt.figure(figsize=(15, 9.5))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.18)

# --- A: face-type fractions, two contrasting parts ---
ok = frac.sum(1) > 0
turned = int(np.argmax((frac[:, 1] + frac[:, 4]) * ok))   # cylinder+torus-heavy
boxy = int(np.argmax(frac[:, 0] * (SC[:, 11] > 1) * ok))  # plane-heavy, >=10 faces
axA = fig.add_subplot(gs[0, 0])
x = np.arange(8)
axA.bar(x - 0.2, frac[turned], 0.38, color=ORANGE, label=f"turned-style ({stems[turned].split('_')[0]})")
axA.bar(x + 0.2, frac[boxy], 0.38, color=BLUE, label=f"boxy-style ({stems[boxy].split('_')[0]})")
axA.set_xticks(x, TYPES, fontsize=8.5)
axA.legend(fontsize=8.5)
axA.set_title("A — face-type fractions: WHAT the part is made of\n(CAD-native identity; invisible to D2)", fontsize=10)
axA.spines[["top", "right"]].set_visible(False)

# --- B: sphericity spectrum, rendered ---
gsB = gs[0, 1].subgridspec(2, 5, height_ratios=[4, 1], hspace=0)
order = np.argsort(spher)
picks = [order[int(q * (len(order) - 1))] for q in (0.02, 0.25, 0.5, 0.75, 0.99)]
for k, i in enumerate(picks):
    axc = fig.add_subplot(gsB[0, k], projection="3d")
    render(axc, str(stems[i]))
    axc.set_title(f"ψ = {spher[i]:.2f}", fontsize=9)
axB = fig.add_subplot(gsB[1, :])
axB.axis("off")
axB.set_title("B — sphericity ψ = π$^{1/3}$(6V)$^{2/3}$/A: flat → chunky, GLOBAL mass proportions\n"
              "(a cube has ψ≈0.81 with zero sphere faces — not the same thing as the sphere face-type)",
              fontsize=10, y=0.0)

# --- C: bbox aspect-ratio map ---
axC = fig.add_subplot(gs[1, 0])
axC.scatter(ba, ca, s=6, alpha=0.35, color=BLUE, edgecolors="none")
examples = {"100028": ("rod", None), "51350": ("thin plate", None)}
cube = int(np.argmax(np.minimum(ba, ca)))
ann = [(int(np.flatnonzero([s.startswith("100028") for s in stems])[0]), "rod"),
       (int(np.flatnonzero([s.startswith("51350") for s in stems])[0]), "thin plate"),
       (cube, "cube-ish")]
for i, label in ann:
    axC.scatter(ba[i], ca[i], s=60, color=ORANGE, zorder=3)
    axC.annotate(f"{label}\n({ba[i]:.2f}, {ca[i]:.2f})", (ba[i], ca[i]),
                 textcoords="offset points", xytext=(10, 6), fontsize=9, color="#1f2937")
axC.set_xlabel("b / a  (middle / longest)")
axC.set_ylabel("c / a  (shortest / longest)")
axC.set_title("C — bbox aspect ratios: sort dims a ≥ b ≥ c, take (b/a, c/a)\n"
              "each part = one point; (1,1) cube · (1,0) plate · (0,0) rod", fontsize=10)
axC.spines[["top", "right"]].set_visible(False)

# --- D: complexity ---
axD = fig.add_subplot(gs[1, 1])
axD.hist(logf, bins=30, color=BLUE, edgecolor="white")
for count in (3, 9, 31, 130):
    axD.axvline(np.log10(count), color=ORANGE, ls="--", lw=1.2)
    axD.text(np.log10(count), axD.get_ylim()[1] * 0.92, f"{count}\nfaces",
             ha="center", fontsize=8, color="#1f2937")
axD.set_xlabel("log$_{10}$(face count)")
axD.set_ylabel("parts")
axD.set_title("D — modelling complexity: log$_{10}$(face count)\n"
              "log: 9→90 matters like 90→900 (ratios, not absolutes)", fontsize=10)
axD.spines[["top", "right"]].set_visible(False)

fig.suptitle("The 12-scalar block, visually — four groups, each seeing something D2 cannot", fontsize=13)
plt.savefig("figures/scalar_features_explainer.png", dpi=115, bbox_inches="tight")
print("saved figures/scalar_features_explainer.png")
print("turned:", stems[turned], frac[turned].round(2), "| boxy:", stems[boxy], frac[boxy].round(2))
print("sphericity picks:", [(str(stems[i]), round(float(spher[i]), 2)) for i in picks])
