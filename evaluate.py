"""Evaluation for CAD-to-CAD retrieval (Block 3).

What it measures (and what each result can/cannot show):
1. self-retrieval  - every part retrieves itself at d=0 (sanity, not quality)
2. perturbation    - re-extract fingerprints under changed nuisance parameters
                     (sampling seed, tessellation deflection); original must rank #1
                     -> stability, not correctness
3. seg referee     - label agreement of top-5 vs a random-5 baseline; labels were
                     never features, so beating random is independent evidence
4. threshold       - does NN distance predict referee agreement? (Chris's idea)
5. v0 vs v1        - scalar block adopted only if the referee improves

Honest limits: the referee is a WEAK proxy (construction similarity, not
functional reuse); perturbation tests stability; none of this replaces human
relevance judgements.
"""

import json
import pathlib

import numpy as np
import trimesh

import features

CACHE_DIR = pathlib.Path(__file__).parent / "cache"
K = 5
RANDOM_BASELINE_SEED = 42


# ---------- building blocks ----------

def load_base():
    z = np.load(CACHE_DIR / "features_v0.npz", allow_pickle=True)
    return z["stems"], z["d2"], z["scalars"], z["display"]


def seg_histograms(stems):
    """(N, 8) per-part operation fractions - the referee's scorecard."""
    H = []
    for s in stems:
        labels = [int(x) for x in open(f"fusion360subset/seg/{s}.seg").read().split()]
        h = np.bincount(labels, minlength=8).astype(float)
        H.append(h / h.sum())
    return np.array(H)


def extract_d2_variant(path, deflection, seed):
    """Same D2 pipeline as features.py but with explicit nuisance parameters."""
    shape = features.load_shape(path)
    mesh = features.shape_to_trimesh(shape, deflection=deflection)
    pts, _ = trimesh.sample.sample_surface(mesh, features.D2_SAMPLES, seed=seed)
    return features.d2_histogram(np.asarray(pts), seed=seed)


def build_variant_cache(tag, deflection, seed):
    """Re-fingerprint the whole corpus under a perturbation; cached on disk."""
    out = CACHE_DIR / f"d2_{tag}.npy"
    if out.exists():
        return np.load(out)
    stems, _, _, _ = load_base()
    D2 = np.array([extract_d2_variant(f"fusion360subset/step/{s}.stp", deflection, seed)
                   for s in stems])
    np.save(out, D2)
    return D2


def topk_indices(D, q_vec, exclude=None, k=K):
    d = np.linalg.norm(D - q_vec, axis=1)
    order = np.argsort(d)
    hits = [i for i in order if i != exclude][:k]
    return hits, d


# ---------- metrics ----------

def self_retrieval(D2):
    """Fraction of parts whose nearest corpus vector (incl. self) is themselves."""
    ok = 0
    for i in range(len(D2)):
        d = np.linalg.norm(D2 - D2[i], axis=1)
        ok += int(np.argmin(d) == i)
    return ok / len(D2)


def perturbation_ranks(D2_base, D2_variant):
    """For each part: rank of its true self when querying with the perturbed print."""
    ranks = []
    for i in range(len(D2_base)):
        d = np.linalg.norm(D2_base - D2_variant[i], axis=1)
        ranks.append(int(np.argsort(d).tolist().index(i)) + 1)
    return np.array(ranks)


def seg_referee(D, SEG, k=K, random_baseline=False):
    """Referee distance: mean L2 between query and top-k seg histograms (LOWER = better).

    L2 per Chris's co-design choice (one metric everywhere in the system).
    random_baseline=True scores k random parts instead - the zero-intelligence control.
    """
    n = len(D)
    rng = np.random.default_rng(RANDOM_BASELINE_SEED)
    out = np.empty(n)
    for i in range(n):
        if random_baseline:
            hits = [j for j in rng.choice(n, k + 1, replace=False) if j != i][:k]
        else:
            hits, _ = topk_indices(D, D[i], exclude=i, k=k)
        out[i] = float(np.mean(np.linalg.norm(SEG[hits] - SEG[i], axis=1)))
    return out


def nn_distances(D):
    """Distance to the nearest non-self neighbour, per part (the confidence signal)."""
    out = np.empty(len(D))
    for i in range(len(D)):
        d = np.linalg.norm(D - D[i], axis=1)
        d[i] = np.inf
        out[i] = d.min()
    return out


def build_v1(D2, SC):
    """v1 vector: z-scored scalars + D2, equal-block-energy weighting.

    Each block z-scored per dimension, then divided by sqrt(block dim count) so
    64 D2 bins and 12 scalars contribute comparable total energy to L2.
    """
    def z(X):
        std = X.std(0)
        std[std == 0] = 1.0
        return (X - X.mean(0)) / std
    return np.hstack([z(D2) / np.sqrt(D2.shape[1]), z(SC) / np.sqrt(SC.shape[1])])


# ---------- main ----------

def main():
    stems, D2, SC, _ = load_base()
    SEG = seg_histograms(stems)

    print("1. self-retrieval:", f"{self_retrieval(D2):.1%}")

    for tag, defl, seed in [("seed1", features.DEFLECTION, 1), ("defl005", 0.05, features.SEED)]:
        Dv = build_variant_cache(tag, defl, seed)
        r = perturbation_ranks(D2, Dv)
        print(f"2. perturbation {tag}: rank-1 {np.mean(r == 1):.1%} | top-5 {np.mean(r <= 5):.1%} | worst rank {r.max()}")

    ref_v0 = seg_referee(D2, SEG)
    ref_rnd = seg_referee(D2, SEG, random_baseline=True)
    print(f"3. seg referee (L2, lower=better): v0 top-5 {ref_v0.mean():.3f} vs random-5 control {ref_rnd.mean():.3f}")

    V1 = build_v1(D2, SC)
    ref_v1 = seg_referee(V1, SEG)
    print(f"5. v1 (D2+scalars): referee distance {ref_v1.mean():.3f}")

    nn = nn_distances(D2)
    from scipy.stats import spearmanr
    rho, pval = spearmanr(nn, ref_v0)
    print(f"4. threshold: Spearman(NN distance, referee distance) rho={rho:.3f} (p={pval:.1e})")

    np.savez(CACHE_DIR / "eval_results.npz",
             ref_v0=ref_v0, ref_rnd=ref_rnd, ref_v1=ref_v1, nn=nn)
    print("saved cache/eval_results.npz")


if __name__ == "__main__":
    main()
