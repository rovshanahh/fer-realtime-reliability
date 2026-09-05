#!/usr/bin/env python3
"""Boundary-aware moving-block bootstrap, dort yontem icin, iki katilimci.
Makalede tarif edilen yontemin aynisi: 30 karelik bitisik bloklar, ardisik
farklar sadece blok icinde hesaplanir, bloklar arasi yapay gecisler haric."""
import numpy as np, pandas as pd, glob, json, os
from collections import defaultdict

B, NBOOT, SEED = 30, 2000, 123
METHODS = ["none", "ema", "voting", "hybrid"]
INVALID = -1


def load(csv):
    d = pd.read_csv(csv)
    return d.label.values, d[[f"p{i}" for i in range(8)]].values, d.face_detected.values


def stats_from_blocks(lab, prob, det, idx):
    """idx: secilen blok baslangiclari. Metrikleri sadece blok icinden hesaplar."""
    flips = pairs = 0
    pj_sum = pj_n = 0.0
    segs = []
    ndet = ntot = 0
    for s0 in idx:
        L = lab[s0:s0+B]; P = prob[s0:s0+B]; D = det[s0:s0+B]
        ntot += len(D); ndet += D.sum()
        v = L[L != INVALID]
        if len(v) >= 2:
            flips += int((v[1:] != v[:-1]).sum()); pairs += len(v)-1
            c = 1
            for i in range(1, len(v)):
                if v[i] == v[i-1]: c += 1
                else: segs.append(c); c = 1
            segs.append(c)
        Pv = P[L != INVALID]
        if len(Pv) >= 2:
            pj_sum += np.abs(np.diff(Pv, axis=0)).sum(1).sum(); pj_n += len(Pv)-1
    return (flips/pairs if pairs else np.nan,
            float(np.mean(segs)) if segs else np.nan,
            pj_sum/pj_n if pj_n else np.nan,
            1-ndet/ntot if ntot else np.nan)


def boot(lab, prob, det, rng):
    n = len(lab); starts = np.arange(0, n-B+1)
    k = max(1, n//B)
    out = []
    for _ in range(NBOOT):
        out.append(stats_from_blocks(lab, prob, det, rng.choice(starts, k, replace=True)))
    return np.array(out, float)


def ci(a):
    a = a[~np.isnan(a)]
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)), float(np.mean(a)))


rows, paired = [], []
for part, d in [("P1", "results_P1"), ("P2", "results_P2")]:
    scen = sorted({os.path.basename(f).split("_")[1] for f in glob.glob(f"{d}/logs/*.csv")})
    scen = [s for s in scen if s in ("S6", "S7", "S8", "S9")]
    for s in scen:
        cache = {}
        for m in METHODS:
            f = f"{d}/logs/{part}_{s}_{m}.csv"
            if not os.path.exists(f): continue
            lab, prob, det = load(f); cache[m] = (lab, prob, det)
            rng = np.random.default_rng(SEED)
            bs = boot(lab, prob, det, rng)
            pt = stats_from_blocks(lab, prob, det, np.arange(0, len(lab)-B+1, B))
            for j, nm in enumerate(["LFR", "MSL", "PJ_L1", "FDDR"]):
                lo, hi, mu = ci(bs[:, j])
                rows.append(dict(participant=part, scenario=s, method=m, metric=nm,
                                 estimate=pt[j], boot_mean=mu, lo=lo, hi=hi))
        # ESLESTIRILMIS FARK: ayni bloklar, hybrid eksi digerleri
        if "hybrid" in cache:
            n = len(cache["hybrid"][0]); starts = np.arange(0, n-B+1); k = max(1, n//B)
            rng = np.random.default_rng(SEED)
            sel = [rng.choice(starts, k, replace=True) for _ in range(NBOOT)]
            for other in ["none", "ema", "voting"]:
                if other not in cache: continue
                diffs = []
                for idx in sel:
                    a = stats_from_blocks(*cache["hybrid"], idx)[0]
                    b = stats_from_blocks(*cache[other], idx)[0]
                    diffs.append(a-b)
                dd = np.array(diffs, float); dd = dd[~np.isnan(dd)]
                lo, hi = np.percentile(dd, [2.5, 97.5])
                paired.append(dict(participant=part, scenario=s, contrast=f"hybrid-{other}",
                                   mean_diff=float(dd.mean()), lo=float(lo), hi=float(hi),
                                   excludes_zero=bool(hi < 0 or lo > 0)))
        print(f"  {part} {s} tamam", flush=True)

json.dump({"per_method": rows, "paired_LFR": paired}, open("bootstrap_all.json", "w"), indent=2)

print("\n" + "="*84)
print("ESLESTIRILMIS FARK, LFR (hybrid eksi digeri). Negatif = hybrid daha iyi.")
print("="*84)
print(f'{"P":4}{"S":5}{"karsilastirma":18}{"ort. fark":>12}{"%95 CI":>26}{"sifiri disliyor":>18}')
for r in paired:
    print(f'{r["participant"]:4}{r["scenario"]:5}{r["contrast"]:18}{r["mean_diff"]:>12.4f}'
          f'{("[%.4f, %.4f]" % (r["lo"], r["hi"])):>26}'
          f'{("EVET" if r["excludes_zero"] else "hayir"):>18}')
n_ok = sum(r["excludes_zero"] for r in paired)
print(f'\n{n_ok}/{len(paired)} karsilastirmada %95 guven araligi sifiri dislyor.')
