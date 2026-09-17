"""Generate figures/pipeline_dataflow.png — the whole system in one diagram.

Offline lane: stp -> OCC Shape -> three feature branches -> npz cache.
Query lane: same extraction -> L2 + argsort -> top-k. Data shapes on every arrow.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BLUE, ORANGE, GRAY, GREEN = "#3572b0", "#e07b39", "#6b7280", "#4a7c59"


def main():
    fig, ax = plt.subplots(figsize=(15, 8.5))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, title, body, color=BLUE):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                     facecolor="white", edgecolor=color, linewidth=2))
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", fontsize=10, fontweight="bold", color=color)
        ax.text(x + w / 2, y + (h - 0.55) / 2, body, ha="center", va="center", fontsize=8.6,
                family="monospace", color="#1f2937")

    def arrow(x1, y1, x2, y2, label="", color=GRAY):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                     linewidth=1.6, color=color))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.16, label, ha="center", fontsize=8,
                    color=color, style="italic")

    ax.text(0.2, 9.6, "OFFLINE - runs once, ~1 min for all 1008 parts   (features.py main)",
            fontsize=11, fontweight="bold", color=BLUE)
    box(0.2, 7.6, 2.4, 1.5, "part.stp", "text file, ~19 KB\nexact b-rep\n(ISO STEP format)", GRAY)
    box(3.6, 7.6, 2.6, 1.5, "load_shape", "OCC / OpenCASCADE\nkernel parses it\n-> Shape object", BLUE)
    box(7.2, 7.6, 3.0, 1.5, "OCC Shape (b-rep)", "faces w/ exact surfaces\n+ topology graph\ne.g. 9 faces, median part", BLUE)
    arrow(2.6, 8.35, 3.6, 8.35)
    arrow(6.2, 8.35, 7.2, 8.35)

    box(0.4, 5.2, 3.2, 1.6, "face_type_fractions", "IN : Shape\nOUT: 8 fractions + n_faces\n[.67 plane, .33 cyl, ...]", GREEN)
    box(4.0, 5.2, 3.2, 1.6, "global_props", "IN : Shape\nOUT: volume, area,\nsorted bbox dims (exact)", GREEN)
    box(7.6, 5.2, 3.4, 1.6, "shape_to_trimesh", "IN : Shape, deflection 0.1\nOUT: triangle mesh\n(V x 3 verts, T x 3 tris)", GREEN)
    arrow(8.0, 7.6, 2.2, 6.8)
    arrow(8.6, 7.6, 5.6, 6.8)
    arrow(9.2, 7.6, 9.3, 6.8)

    box(11.6, 5.2, 3.1, 1.6, "d2_histogram", "IN : 2048 sampled pts\nOUT: 64-bin fingerprint\n(Osada 2002)", GREEN)
    arrow(11.0, 6.0, 11.6, 6.0, "sample 2048 pts\n(area-weighted)")

    box(4.2, 2.9, 6.6, 1.5, "extract_part  ->  one row per part",
        "d2(64) = ranks results   |   scalars(12) = v1 block   |   display(5) = shown, never ranked", ORANGE)
    arrow(2.0, 5.2, 5.5, 4.4)
    arrow(5.6, 5.2, 6.5, 4.4)
    arrow(13.1, 5.2, 9.5, 4.4)

    box(4.9, 0.7, 5.2, 1.5, "cache/features_v0.npz  (~0.5 MB)",
        "stems(1008) - d2(1008x64)\nscalars(1008x12) - display(1008x5)\n+ meta: deflection/bins/seed", ORANGE)
    arrow(7.5, 2.9, 7.5, 2.2, "x 1008 parts")

    ax.text(11.3, 9.6, "QUERY TIME - per query, < 1 ms", fontsize=11, fontweight="bold", color=ORANGE)
    box(11.6, 7.9, 3.1, 1.2, "query part", "same extract_part\n-> its 64-dim d2", ORANGE)
    box(11.6, 2.9, 3.1, 1.6, "L2 + argsort", "d(q, all 1008)\nsort ascending\nreturn k smallest", ORANGE)
    arrow(13.15, 7.9, 13.15, 4.5, "compare against cache")
    arrow(10.1, 1.45, 11.6, 3.2, "load once")
    box(11.6, 0.5, 3.1, 1.4, "top-k results", "ranked stems + distances\n+ size shown beside each\n(threshold idea lives here)", ORANGE)
    arrow(13.15, 2.9, 13.15, 1.9)

    fig.suptitle("CAD-to-CAD retrieval v0 - the whole system, with what flows along every arrow", fontsize=13)
    plt.savefig("figures/pipeline_dataflow.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved figures/pipeline_dataflow.png")


if __name__ == "__main__":
    main()
