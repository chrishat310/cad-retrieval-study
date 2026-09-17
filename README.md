# CAD-to-CAD retrieval: how far do cheap geometric descriptors get you?

A shape-retrieval study on 1008 b-rep CAD parts. Query a part, get the most similar
historic parts back. No learned model anywhere: a 64-bin D2 shape distribution,
exact nearest-neighbour search in memory, and a confidence gauge on every result.

The retrieval is the small half. Most of the notebook goes on **how you would know
whether it works**, on a corpus that ships no relevance labels at all.

**[Read the notebook](cad_retrieval.ipynb)** — it is the report, top to bottom.

## What came out of it

| | |
| --- | --- |
| Corpus | 1008 single-body b-rep parts (STEP), Fusion 360 Gallery segmentation subset |
| Method | D2 shape distribution, 64 bins, 2048 area-weighted surface samples, tessellation deflection 0.1 |
| Storage | ~0.5 MB numpy archive in memory, brute-force exact k-NN. No vector database, and the notebook argues why |
| Human evaluation | precision@5 = **0.387** over 277 pre-registered, blinded judgements — roughly two useful starting points in every five |
| Confidence signal | Spearman(NN distance, referee distance) rho = +0.331, p = 3.4e-27 |

Three findings worth more than the number:

**The evaluation method was the thing that needed debugging.** With no relevance
labels, I built a referee from the dataset's per-face operation labels. It looked
principled and it was wrong: a retriever using nothing but face count beat the real
system on it, while returning visual garbage. The referee got demoted from judge to
falsifier, and the quality claim moved onto label-free evidence plus hand labelling.

**The referee and the human disagree, and the human wins.** Adding a 12-scalar
feature block (v1) beats the baseline on the referee for 78.3% of queries and is
strictly more stable. On the 277 human judgements it is *worse* — 0.313 against
0.387, Wilcoxon p = 0.281, so the difference is not significant either. The baseline
shipped. An automated metric that disagrees with the only ground truth you have is
not a tiebreaker.

**Reuse is asymmetric.** This came out of hand-labelling and is the finding I did not
go looking for. A retrieved part *simpler* than the query is not a starting point at
all — the work saved can go negative. Symmetric similarity cannot express that, which
is a structural limit of the whole approach rather than a tuning problem.

## Running it

```bash
conda env create -f environment.yml
conda activate cad-retrieval
# then obtain the dataset -- see DATASET.md
jupyter lab cad_retrieval.ipynb
```

`pythonocc-core` is conda-forge only, which is why there is an `environment.yml`
and no `requirements.txt`.

The dataset is **not** in this repository. `DATASET.md` explains how to obtain the
1008 parts from Autodesk and where to put them; `filelist.txt` names them.

| File | What it is |
| --- | --- |
| `cad_retrieval.ipynb` | The study. Everything is here. |
| `features.py` | b-rep loading, D2 fingerprint, the 12-scalar block |
| `evaluate.py` | Tests 1-4, the seg referee, stability under re-sampling |
| `gold_set.py` | Gold-set protocol: build, then score |
| `make_labeler.py` | Browser labelling tool, one candidate at a time, blinded |
| `make_*_figure.py` | Regenerate the explainer figures |
| `gold/labels.tsv` | The 277 judgements |

## Limitations

- D2 is a global summary. It cannot separate a through-hole from a blind hole, four
  holes from four bosses, or a part from its mirror image.
- The v1 aspect ratios use an axis-aligned bounding box, so their rotation-invariance
  is a property of this corpus rather than of the descriptor.
- The corpus is simple single-body parts. A real archive of thin-walled, multi-body
  enclosures would stress a fingerprint that mostly sees the outer envelope.
- **The gold set has one rater, and that rater is also the author.** Blinding and
  pre-registering the protocol reduce that bias. They do not remove it.

## Where it would go next

Cheapest first: D2's siblings from the same paper (A3, D3, D4) reuse the sampling and
histogram machinery unchanged. Then a frozen b-rep-native encoder — UV-Net or BRepNet
— through the same four-test harness and gold set. Topology the b-rep already holds
and this study never reads: hole counts, face-adjacency statistics. A directional
complexity guard, so a retrieval simpler than the query flags itself.

## Origin and licence

This work began as a take-home exercise. The problem framing, the relevance
definition, the evaluation design and every conclusion are my own, and no company's
material is reproduced here.

Code is MIT (see `LICENSE`). Material derived from the Fusion 360 Gallery Dataset —
the rendered part images in `figures/` and in the notebook outputs, `filelist.txt`,
and `gold/` — is governed by **Autodesk's dataset licence, non-commercial research
use only**. See `DATASET.md` before reusing any of it.

Christophe Hatterer
