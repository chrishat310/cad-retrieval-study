# Dataset — obtaining it, and what the licence permits

## This repository contains a modified portion of the Fusion 360 Gallery Dataset

Some material here is **derived from the Fusion 360 Gallery Dataset**, published
by Autodesk AI Lab: <https://github.com/AutodeskAILab/Fusion360GalleryDataset>.

Specifically, the following are a portion or modification of that dataset and are
**not** original work of this repository's author:

| Item | What it is |
| --- | --- |
| `figures/*.png`, `figures/*.svg` | Rendered images of dataset parts, and plots computed from them |
| Rendered images inside `cad_retrieval.ipynb` outputs | Same, embedded in the notebook |
| `filelist.txt` | The 1008 dataset stem identifiers used as the corpus |
| `gold/labels.tsv`, `gold/mapping.json`, `gold/recheck.tsv` | Human relevance judgements about dataset parts, keyed by stem |

**No dataset geometry is redistributed here.** No `.stp`, `.seg`, or
`segment_names.json` file is included.

## Licence

The derived material above is governed by the **Fusion 360 Gallery Dataset
licence**, not by this repository's MIT licence. The terms that matter:

- **Non-commercial research use only.** This applies to the dataset and to
  derivative works of it, including rendered images.
- The complete dataset may not be redistributed. Portions and modified versions
  may be shared only with a clear indication that they are not the original
  dataset — which is the purpose of this file.
- These notices must be preserved and passed on to anyone who receives this
  material from you.

Read the licence itself before using any of it:
<https://github.com/AutodeskAILab/Fusion360GalleryDataset/blob/master/LICENSE.md>

## Obtaining the corpus

1. Follow the download instructions in the Fusion 360 Gallery Dataset repository
   and obtain the **Segmentation Dataset** (the release containing STEP files
   with per-face segmentation).
2. Place the files so that this layout exists at the repository root:

   ```
   fusion360subset/
     step/<stem>.stp          1008 files
     seg/<stem>.seg           1008 files
     segment_names.json
   ```

3. `filelist.txt` lists the 1008 stems this study used. The corpus is a subset of
   the full segmentation dataset; restricting to these stems reproduces it.

`fusion360subset/` is gitignored and must never be committed.

## Verify your install before running anything

```bash
python -c "import features; s = features.load_shape('fusion360subset/step/' + open('filelist.txt').readline().strip() + '.stp'); print('loaded OK:', s)"
```

If that prints a shape, the OCC stack and the data layout are both correct.
