"""Gold set: pre-registered hand-labelled evaluation (the load-bearing number).

PROTOCOL (registered 2026-08-07, BEFORE any label exists):
- 30 queries, stratified: 10 per face-count tercile of the corpus, rng seed 0.
- Per query: the UNION of v0's top-5 and v1's top-5 (5-10 unique candidates),
  shuffled (seed = query index), lettered A.., system identity hidden.
- Question per candidate: "Would I open this file as a starting point for the
  query part?"  y / n.  Unsure counts as n (conservative), noted in a comment.
- Labeler: Christophe, fresh (not end-of-day), one sitting if possible.
- Scoring: precision@5 per system = mean fraction of its top-5 labelled y.
  Also reported per tercile. Labels are never edited after scores are seen.

Usage:  python gold_set.py build   -> gold/sheets/*.png + gold/labels.tsv + mapping
        python gold_set.py score   -> P@5(v0), P@5(v1) from the filled labels.tsv
"""

import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import features
import evaluate

HERE = pathlib.Path(__file__).parent
GOLD = HERE / "gold"
K = 5
SEED = 0
N_PER_TERCILE = 10


def load():
    z = np.load(HERE / "cache/features_v0.npz", allow_pickle=True)
    stems, D2, SC, DISP = z["stems"], z["d2"], z["scalars"], z["display"]
    V1 = evaluate.build_v1(D2, SC)
    return stems, D2, SC, DISP, V1


def topk(D, q, k=K):
    d = np.linalg.norm(D - D[q], axis=1)
    d[q] = np.inf
    return list(np.argsort(d)[:k])


def sample_queries(stems):
    fc = np.array([len(open(HERE / f"fusion360subset/seg/{s}.seg").read().split()) for s in stems])
    t1, t2 = np.percentile(fc, [33.3, 66.7])
    rng = np.random.default_rng(SEED)
    picks = []
    for lo, hi in [(-1, t1), (t1, t2), (t2, 1e9)]:
        pool = np.flatnonzero((fc > lo) & (fc <= hi))
        picks += list(rng.choice(pool, N_PER_TERCILE, replace=False))
    return picks, fc


def render(ax, stem, color):
    m = features.shape_to_trimesh(features.load_shape(str(HERE / f"fusion360subset/step/{stem}.stp")))
    v, t = np.array(m.vertices), np.array(m.faces)
    ax.add_collection3d(Poly3DCollection(v[t], facecolor=color, edgecolor="none", alpha=0.95))
    lo, hi = v.min(0), v.max(0)
    ctr, rad = (lo + hi) / 2, (hi - lo).max() / 2
    for dim, cc in zip("xyz", ctr):
        getattr(ax, f"set_{dim}lim")(cc - rad, cc + rad)
    ax.set_axis_off()


def build():
    (GOLD / "sheets").mkdir(parents=True, exist_ok=True)
    stems, D2, SC, DISP, V1 = load()
    picks, fc = sample_queries(stems)
    mapping, tsv_rows = {}, []
    for qi, q in enumerate(picks):
        q = int(q)
        hits_v0, hits_v1 = topk(D2.copy(), q), topk(V1.copy(), q)
        pool = sorted(set(hits_v0) | set(hits_v1))
        rng = np.random.default_rng(qi)
        order = list(rng.permutation(len(pool)))
        letters = [chr(65 + i) for i in range(len(pool))]
        shuffled = [pool[o] for o in order]
        mapping[str(stems[q])] = {
            "query_index": q,
            "candidates": {letters[i]: {"stem": str(stems[shuffled[i]]),
                                        "in_v0": shuffled[i] in hits_v0,
                                        "in_v1": shuffled[i] in hits_v1}
                           for i in range(len(shuffled))},
            "v0_top5": [str(stems[i]) for i in hits_v0],
            "v1_top5": [str(stems[i]) for i in hits_v1],
        }
        n = len(shuffled)
        cols = 6
        rows = int(np.ceil((n + 1) / cols))
        fig = plt.figure(figsize=(2.6 * cols, 2.9 * rows))
        ax = fig.add_subplot(rows, cols, 1, projection="3d")
        render(ax, str(stems[q]), "#e07b39")
        ax.set_title(f"QUERY ({fc[q]} faces)\n{str(stems[q]).split('_')[0]} - {DISP[q][4]:.1f} cm", fontsize=9)
        for i, ci in enumerate(shuffled):
            ax = fig.add_subplot(rows, cols, i + 2, projection="3d")
            render(ax, str(stems[ci]), "#7fb3d5")
            ax.set_title(f"[{letters[i]}]\n{DISP[ci][4]:.1f} cm", fontsize=9)
        fig.suptitle(f"Sheet {qi + 1:02d}/30 - 'Would I open this file as a starting point?'  y/n per letter",
                     fontsize=11)
        plt.tight_layout()
        plt.savefig(GOLD / "sheets" / f"sheet_{qi + 1:02d}_{stems[q]}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        for i in range(n):
            tsv_rows.append(f"{qi + 1:02d}\t{stems[q]}\t{letters[i]}\t\t")
        print(f"sheet {qi + 1:02d}: {n} candidates", flush=True)
    json.dump(mapping, open(GOLD / "mapping.json", "w"), indent=1)
    with open(GOLD / "labels.tsv", "w") as f:
        f.write("# Gold-set labels - protocol registered 2026-08-07 (see gold_set.py docstring)\n")
        f.write("# Fill the 'label' column with y or n. Unsure = n (add a note). Do not reorder rows.\n")
        f.write("sheet\tquery\tletter\tlabel\tnote\n")
        f.write("\n".join(tsv_rows) + "\n")
    print(f"\nbuilt: {len(picks)} sheets, {len(tsv_rows)} judgements to make")
    print(f"label file: {GOLD/'labels.tsv'}")


def score():
    mapping = json.load(open(GOLD / "mapping.json"))
    labels = {}
    for line in open(GOLD / "labels.tsv"):
        if line.startswith("#") or line.startswith("sheet") or not line.strip():
            continue
        sheet, qstem, letter, label = (line.rstrip("\n").split("\t") + [""])[:4]
        labels[(qstem, letter)] = label.strip().lower()
    missing = [k for k, v in labels.items() if v not in ("y", "n")]
    if missing:
        print(f"NOT SCORED: {len(missing)} unlabelled rows (first: {missing[:3]})")
        return
    p5 = {"v0": [], "v1": []}
    for qstem, m in mapping.items():
        rel = {c["stem"]: labels[(qstem, letter)] == "y" for letter, c in m["candidates"].items()}
        p5["v0"].append(np.mean([rel[s] for s in m["v0_top5"]]))
        p5["v1"].append(np.mean([rel[s] for s in m["v1_top5"]]))
    n = len(p5["v0"])
    for sys_ in ("v0", "v1"):
        arr = np.array(p5[sys_])
        print(f"precision@5 {sys_}: {arr.mean():.3f}  (per-tercile: "
              f"{arr[:10].mean():.2f} / {arr[10:20].mean():.2f} / {arr[20:].mean():.2f})")
    diff = np.array(p5["v0"]) - np.array(p5["v1"])
    print(f"paired mean difference (v0 - v1): {diff.mean():+.3f} over {n} queries")


if __name__ == "__main__":
    {"build": build, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "build"]()
