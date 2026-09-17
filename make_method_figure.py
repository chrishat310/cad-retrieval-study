"""Generate figures/method_explainer.png — the D2 'dart game' in 6 panels.

Panels: sample points -> random pair distances -> D2 histogram -> overlay
(similar vs different) -> L2 as the gap -> sorted distances / top-k.
Anchor numbers printed on the figure: ring-ring vs ring-rod L2.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

import features

BLUE, ORANGE, GRAY = "#3572b0", "#e07b39", "#9ca3af"

RING, RING2, ROD = "53216_2857e8ac_7", "147630_7f858240_3", "100028_0c3a8f1c_1"


def main():
    z = np.load("cache/features_v0.npz", allow_pickle=True)
    stems, D2 = z["stems"], z["d2"]
    idx = {s: i for i, s in enumerate(stems)}

    m = features.shape_to_trimesh(features.load_shape(f"fusion360subset/step/{RING}.stp"))
    pts, _ = trimesh.sample.sample_surface(m, features.D2_SAMPLES, seed=features.SEED)
    pts = np.asarray(pts)

    fig = plt.figure(figsize=(15, 9))

    ax = fig.add_subplot(2, 3, 1, projection="3d")
    v, t = np.array(m.vertices), np.array(m.faces)
    ax.add_collection3d(Poly3DCollection(v[t], facecolor=ORANGE, alpha=0.25, edgecolor="none"))
    ax.scatter(*pts[::8].T, s=2, c=BLUE)
    lo, hi = v.min(0), v.max(0)
    c_, r_ = (lo + hi) / 2, (hi - lo).max() / 2
    for d, cc in zip("xyz", c_):
        getattr(ax, f"set_{d}lim")(cc - r_, cc + r_)
    ax.set_axis_off()
    ax.set_title("STEP 1 - sample 2048 points on the surface\n(area-weighted: big faces get more points)", fontsize=10)

    ax = fig.add_subplot(2, 3, 2, projection="3d")
    ax.scatter(*pts[::8].T, s=2, c=BLUE, alpha=0.4)
    rng = np.random.default_rng(3)
    for _ in range(25):
        i, j = rng.integers(0, len(pts), 2)
        ax.plot(*np.c_[pts[i], pts[j]], c=ORANGE, lw=1)
    for d, cc in zip("xyz", c_):
        getattr(ax, f"set_{d}lim")(cc - r_, cc + r_)
    ax.set_axis_off()
    ax.set_title("STEP 2 - measure 100,000 random\npoint-to-point distances (25 drawn)", fontsize=10)

    ax = fig.add_subplot(2, 3, 3)
    bins = np.linspace(0, 1, features.D2_BINS + 1)[:-1]
    ax.bar(bins, D2[idx[RING]], width=1 / features.D2_BINS, color=BLUE, align="edge")
    ax.set_title("STEP 3 - histogram of those distances = D2\n'the shape's fingerprint', 64 numbers", fontsize=10)
    ax.set_xlabel("distance / 99th-percentile distance")
    ax.set_ylabel("density")
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(2, 3, 4)
    ax.plot(bins, D2[idx[RING]], c=ORANGE, lw=2, label="query ring")
    ax.plot(bins, D2[idx[RING2]], c=BLUE, lw=2, label="another ring")
    ax.plot(bins, D2[idx[ROD]], c=GRAY, lw=2, ls="--", label="a rod")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("STEP 4 - similar shapes have similar fingerprints\nrings overlap; the rod doesn't", fontsize=10)

    ax = fig.add_subplot(2, 3, 5)
    d_rr = np.linalg.norm(D2[idx[RING]] - D2[idx[RING2]])
    d_rd = np.linalg.norm(D2[idx[RING]] - D2[idx[ROD]])
    ax.plot(bins, D2[idx[RING]], c=ORANGE, lw=2)
    ax.plot(bins, D2[idx[ROD]], c=GRAY, lw=2, ls="--")
    ax.fill_between(bins, D2[idx[RING]], D2[idx[ROD]], color=GRAY, alpha=0.3)
    ax.set_title(f"STEP 5 - L2 = one number for the gap\nring vs ring: d={d_rr:.2f} - ring vs rod: d={d_rd:.2f}", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(2, 3, 6)
    d = np.linalg.norm(D2 - D2[idx[RING]], axis=1)
    dd = d[np.argsort(d)][1:]
    ax.plot(range(1, len(dd) + 1), dd, c=BLUE, lw=1.5)
    ax.scatter(range(1, 6), dd[:5], c=ORANGE, zorder=3, s=30, label="top-5 returned")
    ax.axhline(1.5, c=GRAY, ls=":", lw=1.5)
    ax.text(400, 1.55, "possible confidence threshold:\nabove this line, return 'no good match'",
            fontsize=8.5, color="#374151")
    ax.set_xlabel("all 1007 database parts, sorted by distance")
    ax.set_ylabel("L2 distance")
    ax.legend(fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("STEP 6 - sort all distances, return the k smallest\n= top-k retrieval", fontsize=10)

    fig.suptitle("How one query works, end to end: points -> distances -> fingerprint -> compare -> rank", fontsize=13)
    plt.tight_layout()
    plt.savefig("figures/method_explainer.png", dpi=115, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figures/method_explainer.png | ring-ring {d_rr:.2f} vs ring-rod {d_rd:.2f}")


if __name__ == "__main__":
    main()
