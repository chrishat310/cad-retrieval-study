"""Build gold/web/labeler.html — interactive one-candidate-at-a-time labeling.

Design choice (deliberate): candidates are shown ONE AT A TIME next to the query,
not as a grid — each is judged absolutely against the query, never relative to
its pool-mates. Keyboard: y / n / u (unsure -> n with note), backspace = back.
Progress persists in localStorage; "Export labels.tsv" downloads the file to
overwrite gold/labels.tsv, then: python gold_set.py score
"""

import base64
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import features

HERE = pathlib.Path(__file__).parent
GOLD = HERE / "gold"
WEB = GOLD / "web"
PARTS = WEB / "parts"


def render_png(stem, color, out):
    """Lambert-shaded render: per-triangle normals x two lights, so curvature reads."""
    fig = plt.figure(figsize=(3.6, 3.6), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    m = features.shape_to_trimesh(features.load_shape(str(HERE / f"fusion360subset/step/{stem}.stp")))
    v, t = np.array(m.vertices), np.array(m.faces)
    tri = v[t]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    key_light = np.array([0.5, -0.35, 0.79])
    fill_light = np.array([-0.6, 0.7, 0.39])
    key_light /= np.linalg.norm(key_light)
    fill_light /= np.linalg.norm(fill_light)
    # abs(): OCC triangle winding is not consistent; unsigned Lambert avoids black backfaces
    shade = 0.35 + 0.5 * np.abs(n @ key_light) + 0.15 * np.abs(n @ fill_light)
    base = np.array(matplotlib.colors.to_rgb(color))
    facecolors = np.clip(base[None, :] * shade[:, None], 0, 1)
    ax.add_collection3d(Poly3DCollection(tri, facecolors=facecolors, edgecolor="none"))
    lo, hi = v.min(0), v.max(0)
    ctr, rad = (lo + hi) / 2, (hi - lo).max() / 2
    for dim, cc in zip("xyz", ctr):
        getattr(ax, f"set_{dim}lim")(cc - rad, cc + rad)
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    plt.savefig(out, transparent=False, facecolor="white")
    plt.close(fig)


def main():
    PARTS.mkdir(parents=True, exist_ok=True)
    mapping = json.load(open(GOLD / "mapping.json"))
    z = np.load(HERE / "cache/features_v0.npz", allow_pickle=True)
    disp = {str(s): d for s, d in zip(z["stems"], z["display"])}

    items, rendered = [], set()
    for sheet_no, (qstem, m) in enumerate(mapping.items(), 1):
        for stem in [qstem] + [c["stem"] for c in m["candidates"].values()]:
            if stem not in rendered:
                render_png(stem, "#e07b39" if stem == qstem else "#7fb3d5", PARTS / f"{stem}.png")
                rendered.add(stem)
        for letter, c in sorted(m["candidates"].items()):
            items.append({"sheet": f"{sheet_no:02d}", "query": qstem, "letter": letter,
                          "cand": c["stem"], "qsize": round(float(disp[qstem][4]), 1),
                          "csize": round(float(disp[c["stem"]][4]), 1)})
        print(f"sheet {sheet_no:02d} rendered", flush=True)

    html = """<!doctype html><html><head><meta charset="utf-8"><title>Gold-set labeler</title>
<style>
 body{font-family:system-ui,sans-serif;background:#f5f6f8;margin:0;padding:24px;color:#1f2937}
 .wrap{max-width:900px;margin:0 auto}
 .imgs{display:flex;gap:16px;justify-content:center}
 .card{background:#fff;border-radius:10px;padding:12px;box-shadow:0 1px 4px #0002;text-align:center}
 .card img{width:360px;height:360px;object-fit:contain}
 .q{font-weight:600;font-size:15px;margin:14px 0;text-align:center}
 .btns{display:flex;gap:14px;justify-content:center;margin-top:10px}
 button{font-size:17px;padding:10px 30px;border-radius:8px;border:none;cursor:pointer}
 .yes{background:#16a34a;color:#fff}.no{background:#dc2626;color:#fff}
 .unsure{background:#e5e7eb}.back{background:#e5e7eb}
 .bar{height:8px;background:#e5e7eb;border-radius:4px;margin:16px 0}
 .fill{height:8px;background:#3572b0;border-radius:4px;width:0}
 .meta{color:#6b7280;font-size:13px;text-align:center}
 #export{background:#3572b0;color:#fff;display:none;margin:18px auto}
 #note{width:60%;padding:6px;margin-top:8px}
</style></head><body><div class="wrap">
<div class="bar"><div class="fill" id="fill"></div></div>
<div class="meta" id="meta"></div>
<div class="q">&ldquo;Would I open this file as a starting point for the query part?&rdquo;
 &nbsp;(y / n / u keys &middot; unsure = no + note)</div>
<div class="imgs">
 <div class="card"><div>QUERY</div><img id="qimg"><div class="meta" id="qcap"></div></div>
 <div class="card"><div>CANDIDATE <span id="letter"></span></div><img id="cimg"><div class="meta" id="ccap"></div></div>
</div>
<div class="btns">
 <button class="back" onclick="back()">&larr; back</button>
 <button class="yes" onclick="mark('y')">YES (y)</button>
 <button class="no" onclick="mark('n')">NO (n)</button>
 <button class="unsure" onclick="unsure()">unsure (u)</button>
</div>
<div style="text-align:center"><input id="note" placeholder="optional note (saved with next y/n)"></div>
<button id="export" onclick="exportTsv()">Export labels.tsv</button>
</div><script>
const ITEMS = __ITEMS__;
let labels = JSON.parse(localStorage.getItem('gold_labels') || '{}');
let i = ITEMS.findIndex((it) => !(key(it) in labels));
if (i < 0) i = ITEMS.length;
function key(it){ return it.query + '|' + it.letter; }
function show(){
  document.getElementById('fill').style.width = (100 * Object.keys(labels).length / ITEMS.length) + '%';
  document.getElementById('meta').textContent = Object.keys(labels).length + ' / ' + ITEMS.length + ' labelled';
  if (i >= ITEMS.length){ document.getElementById('export').style.display = 'block'; return; }
  const it = ITEMS[i];
  document.getElementById('qimg').src = 'parts/' + it.query + '.png';
  document.getElementById('cimg').src = 'parts/' + it.cand + '.png';
  document.getElementById('qcap').textContent = it.query.split('_')[0] + ' - ' + it.qsize + ' cm';
  document.getElementById('ccap').textContent = it.csize + ' cm';
  document.getElementById('letter').textContent = it.letter + '  (sheet ' + it.sheet + ')';
}
function mark(v, noteExtra){
  const it = ITEMS[i];
  const note = (document.getElementById('note').value || '') + (noteExtra || '');
  labels[key(it)] = {label: v, note: note};
  document.getElementById('note').value = '';
  localStorage.setItem('gold_labels', JSON.stringify(labels));
  i++; show();
}
function unsure(){ mark('n', '[unsure]'); }
function back(){ if (i > 0){ i--; delete labels[key(ITEMS[i])];
  localStorage.setItem('gold_labels', JSON.stringify(labels)); show(); } }
document.addEventListener('keydown', (e) => {
  if (e.target.id === 'note') return;
  if (e.key === 'y') mark('y'); else if (e.key === 'n') mark('n');
  else if (e.key === 'u') unsure(); else if (e.key === 'Backspace') back();
});
function exportTsv(){
  let out = '# Gold-set labels - protocol registered 2026-08-07 (see gold_set.py docstring)\\n' +
            '# Labelled via web labeler; unsure -> n with [unsure] note.\\n' +
            'sheet\\tquery\\tletter\\tlabel\\tnote\\n';
  for (const it of ITEMS){
    const l = labels[key(it)] || {label: '', note: ''};
    out += it.sheet + '\\t' + it.query + '\\t' + it.letter + '\\t' + l.label + '\\t' + l.note + '\\n';
  }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([out], {type: 'text/tab-separated-values'}));
  a.download = 'labels.tsv';
  a.click();
}
show();
</script></body></html>"""
    html = html.replace("__ITEMS__", json.dumps(items))
    (WEB / "labeler.html").write_text(html)
    print(f"\nlabeler: {WEB/'labeler.html'}  ({len(items)} judgements, {len(rendered)} parts rendered)")


if __name__ == "__main__":
    main()
