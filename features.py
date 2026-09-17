"""Feature extraction for CAD-to-CAD retrieval on b-rep STEP parts.

Decisions this file implements (each stated inline below, and in the notebook):
- similarity features are scale/rotation/translation-invariant
- absolute size (bbox, volume) extracted for DISPLAY only, never ranking
- v0 = D2 shape distribution (64 bins, 2048 area-weighted samples, p99-normalised)
- v1 scalar block = face-type fractions, sphericity, sorted aspect ratios, log face count
- tessellation deflection fixed at 0.1 (recorded method parameter)

DATAFLOW (one part):
    part.stp --load_shape--> OCC Shape (exact b-rep)
        |--face_type_fractions--> 8 fractions + face count
        |--global_props--------> volume, area, sorted bbox dims (exact, from b-rep)
        |--shape_to_trimesh----> triangle mesh --sample 2048 pts--> d2_histogram --> 64 bins
    extract_part bundles these into:  d2(64) | scalars(12) | display(5)
    main() sweeps all 1008 parts -> cache/features_v0.npz
"""

import json
import math
import pathlib

import numpy as np
import trimesh                      # mesh library; used ONLY for area-weighted point sampling

# OCC = pythonocc = Python bindings to the OpenCASCADE b-rep kernel
from OCC.Core.Bnd import Bnd_Box                        # axis-aligned bounding box container
from OCC.Core.BRep import BRep_Tool                     # accessor: face -> its triangulation
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface    # accessor: face -> its surface type
from OCC.Core.BRepBndLib import brepbndlib              # fills a Bnd_Box from a shape
from OCC.Core.BRepGProp import brepgprop                # exact volume / surface integrals
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh  # the tessellator (b-rep -> triangles)
from OCC.Core.GProp import GProp_GProps                 # result container for brepgprop
from OCC.Core.STEPControl import STEPControl_Reader     # STEP file parser
from OCC.Core.TopAbs import TopAbs_FACE                 # topology type tag: "give me FACES"
from OCC.Core.TopExp import TopExp_Explorer             # topology iterator
from OCC.Core.TopLoc import TopLoc_Location             # a face's placement transform

# ---- frozen method parameters: change any -> every cached fingerprint changes ----
DEFLECTION = 0.1      # tessellation: max deviation (cm) between true surface and triangles
D2_BINS = 64          # fingerprint resolution
D2_SAMPLES = 2048     # points sampled on each part's surface
D2_PAIRS = 100_000    # random point-pairs measured per part
SEED = 0              # reproducibility: same part -> same fingerprint, always

# GeomAbs surface-type index -> our 8 histogram slots
# 0 Plane, 1 Cylinder, 2 Cone, 3 Sphere, 4 Torus, 5 Bezier, 6 BSpline, 7 other
N_FACE_TYPES = 8


def load_shape(path):
    # IN:  path to a .stp file
    # OUT: one OCC Shape object (the exact b-rep, all solids merged into one)
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))     # parse the STEP text file
    if status != 1:                         # 1 == IFSelect_RetDone == success
        raise IOError(f"STEP read failed ({status}): {path}")   # fail loudly, never silently
    reader.TransferRoots()                  # convert parsed entities -> OCCT native shape
    return reader.OneShape()                # hand back a single shape object


def iter_faces(shape):
    # IN:  OCC Shape
    # OUT: yields its faces one by one, ALWAYS in the same order
    #      (this stable order is what makes .seg line i <-> face i trustworthy)
    exp = TopExp_Explorer(shape, TopAbs_FACE)   # cursor over the topology graph, faces only
    while exp.More():
        yield exp.Current()
        exp.Next()


def face_type_fractions(shape):
    # IN:  OCC Shape
    # OUT: (8-dim fractions vector, face count)
    #      e.g. our rod: [0.667 plane, 0.333 cylinder, 0, ...], n=3
    counts = np.zeros(N_FACE_TYPES)
    for face in iter_faces(shape):
        t = BRepAdaptor_Surface(face).GetType()  # ask the face: what surface are you? -> int
        counts[t if t < 7 else 7] += 1           # fold exotic types into slot 7 ("other")
    n = counts.sum()
    return counts / n if n else counts, int(n)   # divide by n: 9-face and 90-face parts comparable


def shape_to_trimesh(shape, deflection=DEFLECTION):
    # IN:  OCC Shape (exact b-rep)
    # OUT: trimesh.Trimesh (one stitched triangle soup: V x 3 vertices, T x 3 triangle indices)
    #      -- the b-rep -> mesh bridge; deflection is the quality knob
    BRepMesh_IncrementalMesh(shape, deflection)  # tessellate in place (adds triangulation data)
    verts, tris, off = [], [], 0
    for face in iter_faces(shape):
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)     # this face's own little triangle patch
        if tri is None:                              # degenerate face with no triangulation
            continue
        trsf = loc.Transformation()                  # face's placement transform --
        for i in range(1, tri.NbNodes() + 1):        # forgetting it scatters faces at origin
            p = tri.Node(i).Transformed(trsf)        # (OCC counts from 1, not 0)
            verts.append((p.X(), p.Y(), p.Z()))
        for i in range(1, tri.NbTriangles() + 1):
            a, b, c = tri.Triangle(i).Get()          # triangle = 3 indices into THIS face's nodes
            tris.append((off + a - 1, off + b - 1, off + c - 1))  # shift into the global list
        off = len(verts)                             # bookkeeping: where the next face's nodes start
    return trimesh.Trimesh(vertices=np.array(verts), faces=np.array(tris), process=False)
    # process=False: "don't try to repair my mesh" -- keep the geometry exactly as tessellated


def d2_histogram(pts, bins=D2_BINS, n_pairs=D2_PAIRS, seed=SEED):
    # IN:  (n, 3) array of surface points
    # OUT: 64 numbers -- the shape's D2 fingerprint (Osada et al. 2002)
    #      scale/rotation/translation-invariant by construction
    rng = np.random.default_rng(seed)        # fresh reproducible randomness per call
    i = rng.integers(0, len(pts), n_pairs)   # 100k random pair indices...
    j = rng.integers(0, len(pts), n_pairs)
    m = i != j                               # ...drop accidental self-pairs (distance 0)
    d = np.linalg.norm(pts[i[m]] - pts[j[m]], axis=1)   # Euclidean distance of every pair
    scale = np.quantile(d, 0.99)             # THE scale-invariance line: normalise by p99
    if scale <= 0:                           # (p99 not max: one outlier can't stretch the axis)
        raise ValueError("degenerate point cloud: zero extent")
    d = np.clip(d / scale, 0, 1)             # now distances are relative to the part's own size
    h, _ = np.histogram(d, bins=bins, range=(0, 1), density=True)
    return h                                 # density=True: histogram area = 1, parts comparable


def global_props(shape):
    # IN:  OCC Shape
    # OUT: (volume, surface area, bbox dims sorted a >= b >= c)
    #      EXACT values integrated from the b-rep surfaces, not the mesh
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    volume = abs(props.Mass())               # abs(): open/flipped shells can report negative
    sprops = GProp_GProps()
    brepgprop.SurfaceProperties(shape, sprops)
    area = sprops.Mass()
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    dims = np.sort([xmax - xmin, ymax - ymin, zmax - zmin])[::-1]  # sorted: orientation can't
    return volume, area, dims                                      # reorder the axes


def extract_part(path):
    # IN:  path to one .stp
    # OUT: d2(64) -- fingerprint, ranks results
    #      scalars(12) -- 8 face fractions + sphericity + 2 aspect ratios + log faces (v1 block)
    #      display(5) -- a, b, c, volume, diagonal: SHOWN to the user, never ranked on
    #      n_faces -- for cross-checks
    shape = load_shape(path)
    fractions, n_faces = face_type_fractions(shape)
    volume, area, dims = global_props(shape)
    mesh = shape_to_trimesh(shape)
    pts, _ = trimesh.sample.sample_surface(mesh, D2_SAMPLES, seed=SEED)  # area-weighted:
    d2 = d2_histogram(np.asarray(pts))                                   # big faces get more points
    # sphericity in (0, 1]: 1 = perfect sphere, ~0 = leaf/shell; dimensionless -> scale-invariant
    sphericity = (math.pi ** (1 / 3)) * ((6 * volume) ** (2 / 3)) / area if volume > 0 and area > 0 else 0.0
    scalars = np.concatenate([
        fractions,                                   # 8: face-type mix ("what is it made of")
        [sphericity],                                # 1: chunky vs flat/thin
        [dims[1] / dims[0] if dims[0] else 0.0,      # 2: proportions (long? flat? cubic?)
         dims[2] / dims[0] if dims[0] else 0.0],
        [math.log10(n_faces)],                       # 1: modelling complexity (log: 9 vs 90, not 9 vs 900)
    ])
    display = np.array([dims[0], dims[1], dims[2], volume, np.linalg.norm(dims)])
    return d2, scalars, display, n_faces


def main():
    # IN:  fusion360subset/step/*.stp (1008 files)
    # OUT: cache/features_v0.npz containing
    #      stems(1008) | d2(1008 x 64) | scalars(1008 x 12) | display(1008 x 5) | meta(json)
    root = pathlib.Path(__file__).parent / "fusion360subset"
    out_dir = pathlib.Path(__file__).parent / "cache"
    out_dir.mkdir(exist_ok=True)
    paths = sorted((root / "step").glob("*.stp"))
    stems, D2, SC, DISP, failures = [], [], [], [], []
    for k, p in enumerate(paths):
        try:
            d2, sc, disp, _ = extract_part(p)
            stems.append(p.stem)
            D2.append(d2)
            SC.append(sc)
            DISP.append(disp)
        except Exception as e:  # noqa: BLE001 - one bad file must not kill 1007 good ones
            failures.append((p.stem, repr(e)))
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{len(paths)} done, {len(failures)} failures", flush=True)
    np.savez_compressed(
        out_dir / "features_v0.npz",
        stems=np.array(stems), d2=np.array(D2),
        scalars=np.array(SC), display=np.array(DISP),
        meta=json.dumps({"deflection": DEFLECTION, "d2_bins": D2_BINS,   # the method fingerprint:
                         "d2_samples": D2_SAMPLES, "d2_pairs": D2_PAIRS, # cache says HOW it was made
                         "seed": SEED, "units": "cm"}),
    )
    print(f"DONE: {len(stems)} parts extracted, {len(failures)} failures")
    for stem, err in failures:
        print("  FAIL", stem, err)


if __name__ == "__main__":
    main()
