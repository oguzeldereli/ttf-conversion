"""summary of a kfac.py result: P1 (rank correlation of the predictors with measured single-linear KL), per linear type
(H-error, K-FAC damage, benign ratio, effective ranks, P6a captured damage), checkpoints' per-layer deviation (P4a)
  python -m ttfconv.tools.kfac_report r1.json
"""
import json, sys
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr, pearsonr

d = json.load(open(sys.argv[1])); R = d["rows"]
m = [r for r in R if "kl_meas" in r]
km = np.array([r["kl_meas"] for r in m])
print(f"{len(R)} linears, {len(m)} measured swaps")
if len(m) < 3:
    m = []
for key, f in ([] if not m else (("kfac", lambda r: r["kfac"]), ("e_H", lambda r: r["e_H"]), ("e_H*trG", lambda r: r["e_H"] * r["trG"]),
               ("rel_w", lambda r: r["rel_w"]))):
    x = np.array([f(r) for r in m])
    print(f"  P1 {key:8s} spearman {spearmanr(km, x).correlation:+.3f}  pearson(log) {pearsonr(np.log(km), np.log(np.abs(x) + 1e-30))[0]:+.3f}")
if m: print(f"  measured / K-FAC: median {np.median(km / np.array([r['kfac'] for r in m])):.1f}, "
      f"range {np.min(km / np.array([r['kfac'] for r in m])):.1f}-{np.max(km / np.array([r['kfac'] for r in m])):.1f}")
by = defaultdict(list)
for r in R:
    by[r["leaf"]].append(r)
cks = sorted({k for r in R for k in r if k.startswith("ck_")})
hdr = "type          n   e_H   kfac*1e3 benign  prG   prH  leftH  leftGH" + "".join(f"  {c[3:]}:benign kfac*1e3" for c in cks)
print(hdr)
for k, v in sorted(by.items()):
    f = lambda key: np.mean([r[key] for r in v])
    line = (f"{k:12s} {len(v):3d} {f('e_H'):.3f} {1e3 * f('kfac'):8.3f} {f('benign'):.3f} {f('prG'):.3f} {f('prH'):.3f} "
            f"{f('p6a_kfac_left_H'):.3f} {f('p6a_kfac_left_GH'):.3f}")
    for c in cks:
        line += f"  {np.mean([r[c]['benign'] for r in v]):.3f} {1e3 * np.mean([r[c]['kfac'] for r in v]):8.3f}"
    print(line)
if m: print("measured KL by type (mean):", {k: round(float(np.mean([r['kl_meas'] for r in v if 'kl_meas' in r])), 4)
                                     for k, v in by.items() if any('kl_meas' in r for r in v)})
depth = defaultdict(list)
for r in R:
    depth[r["layer"] // 6].append(r)
print("by depth sixth: kfac*1e3", {k * 6: round(1e3 * float(np.mean([r['kfac'] for r in v])), 3) for k, v in sorted(depth.items())})
for c in cks:
    print(f"{c}: benign mean {np.mean([r[c]['benign'] for r in R]):.3f} vs GPTQ-alone {np.mean([r['benign'] for r in R]):.3f}; "
          f"kfac sum {sum(r[c]['kfac'] for r in R):.3f} vs GPTQ-alone {sum(r['kfac'] for r in R):.3f}")
