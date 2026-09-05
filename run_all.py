#!/usr/bin/env python3
"""
run_all.py — tek komutla butun senaryolari isler.

Kullanim:
    python run_all.py --videos videos_gular --participant P2

Ne yapar:
  1. videos klasorundeki her videoyu bulur
  2. her biri icin 4 smoothing yontemini calistirir (none, ema, voting, hybrid)
  3. LFR, MSL, PJ-L1, FDDR, RD hesaplar
  4. scenarios.json'da gecis karesi tanimliysa FRAME SEVIYESI DOGRULUK da hesaplar
  5. ozet tabloyu ekrana basar ve LaTeX tablosunu dosyaya yazar

Onemli fark: bu betik olasilik akisini AYRI logluyor. Voting ve Hybrid tek bir
etiket uretiyor (one-hot), o yuzden orijinal kodda PJ-L1 = 2 x LFR cikiyordu.
Burada PJ-L1 her zaman gercek olasilik akisindan hesaplaniyor.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from collections import Counter, deque

import cv2, numpy as np, torch
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FERConfig
from model.resnet50_fer import ResNet50FER
from detection.haar import HaarCascadeDetector

CLASSES = ["neutral","happy","sad","surprise","fear","disgust","anger","contempt"]
NAME2ID = {n: i for i, n in enumerate(CLASSES)}
METHODS = ["none", "ema", "voting", "hybrid"]
INVALID = -1


# ----------------------------------------------------------------- model pass
def extract_probs(video, model, detector, cfg, device):
    """Videoyu bir kez gecer, her kare icin ham softmax vektorunu dondurur.
    Model bir kez calisir, dort yontem ayni ciktilardan turetilir. Boylece
    yontemler arasi karsilastirma tamamen adil olur."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Video acilamadi: {video}")
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize(cfg.imagenet_mean, cfg.imagenet_std)])
    probs, detected = [], []
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        box = detector.detect(frame)
        if box is None:
            probs.append(None); detected.append(0); continue
        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            probs.append(None); detected.append(0); continue
        crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, (cfg.input_size, cfg.input_size), interpolation=cv2.INTER_LINEAR)
        with torch.no_grad():
            p = torch.softmax(model(tfm(crop).unsqueeze(0).to(device)), 1)[0].cpu().numpy()
        probs.append(p.astype(np.float64)); detected.append(1)
    cap.release()
    fps_src = cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS) or 30.0
    return probs, detected, len(probs) / max(1e-9, time.time() - t0), fps_src


# ------------------------------------------------------------------ smoothing
def apply_method(probs, method, alpha, window):
    """Dondurur: (etiketler, olasilik_akisi, guven).
    olasilik_akisi her zaman gercek dagilim, one-hot degil."""
    labels, stream, conf = [], [], []
    ema = None
    hist = deque(maxlen=window)
    for p in probs:
        if p is None:
            labels.append(INVALID); stream.append(None); conf.append(0.0); continue
        if method == "none":
            cur = p
            lab = int(cur.argmax()); c = float(cur.max())
        elif method == "ema":
            ema = p.copy() if ema is None else alpha * p + (1 - alpha) * ema
            cur = ema / ema.sum()
            lab = int(cur.argmax()); c = float(cur.max())
        elif method == "voting":
            cur = p                      # oylama olasiligi degistirmez
            hist.append(int(p.argmax()))
            lab = Counter(hist).most_common(1)[0][0]
            c = float(p.max())
        elif method == "hybrid":
            ema = p.copy() if ema is None else alpha * p + (1 - alpha) * ema
            cur = ema / ema.sum()        # hybrid'in olasilik akisi = EMA akisi
            hist.append(int(cur.argmax()))
            lab = Counter(hist).most_common(1)[0][0]
            c = float(cur.max())
        labels.append(lab); stream.append(cur); conf.append(c)
    return labels, stream, conf


# -------------------------------------------------------------------- metrics
def temporal_metrics(labels, stream, detected):
    valid = [l for l in labels if l != INVALID]
    lfr = msl = None
    if len(valid) >= 2:
        lfr = sum(1 for i in range(1, len(valid)) if valid[i] != valid[i-1]) / (len(valid) - 1)
    if valid:
        segs, cur, n = [], valid[0], 1
        for l in valid[1:]:
            if l == cur: n += 1
            else: segs.append(n); cur, n = l, 1
        segs.append(n); msl = float(np.mean(segs))
    sv = [s for s in stream if s is not None]
    pj = float(np.mean([np.abs(sv[i] - sv[i-1]).sum() for i in range(1, len(sv))])) if len(sv) >= 2 else None
    fddr = 1.0 - sum(detected) / len(detected) if detected else None
    return lfr, msl, pj, fddr


def reaction_delay(labels, change_frame, target, k=5):
    if change_frame is None or target is None: return None
    clean = [(i, l) for i, l in enumerate(labels) if l != INVALID]
    start = next((j for j, (i, _) in enumerate(clean) if i >= change_frame), None)
    if start is None: return None
    for j in range(start, len(clean) - k + 1):
        if all(clean[j+o][1] == target for o in range(k)):
            return clean[j][0] - change_frame
    return None


def frame_accuracy(labels, change_frame, src, tgt, margin):
    """Kaba segment seviyesi referansa gore frame dogrulugu.
    Gecis anindan +/- margin kare disarida birakilir."""
    if change_frame is None or src is None or tgt is None: return None, None, None, 0
    lo, hi = change_frame - margin, change_frame + margin
    ok = tot = 0; pre_ok = pre_tot = post_ok = post_tot = 0
    for i, l in enumerate(labels):
        if l == INVALID or lo <= i <= hi: continue
        gt = src if i < lo else tgt
        tot += 1; ok += (l == gt)
        if i < lo: pre_tot += 1; pre_ok += (l == gt)
        else: post_tot += 1; post_ok += (l == gt)
    if tot == 0: return None, None, None, 0
    return (ok/tot,
            pre_ok/pre_tot if pre_tot else None,
            post_ok/post_tot if post_tot else None,
            tot)


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="Videolarin bulundugu klasor")
    ap.add_argument("--weights", default="checkpoints/resnet50_layer3_finetuned_best.pt")
    ap.add_argument("--scenarios", default="scenarios.json", help="Senaryo tanim dosyasi")
    ap.add_argument("--participant", default="P1")
    ap.add_argument("--out", default="results_new")
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--margin", type=int, default=15, help="Gecis etrafinda haric tutulan kare")
    args = ap.parse_args()

    cfg = FERConfig()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Cihaz: {device}\n")

    meta = {}
    sp = Path(args.scenarios)
    if sp.exists():
        meta = json.loads(sp.read_text())
        print(f"{len(meta)} senaryo tanimi okundu: {sp}\n")
    else:
        print(f"UYARI: {sp} yok. Frame dogrulugu ve RD hesaplanmayacak.\n")

    model = ResNet50FER(num_classes=cfg.num_classes, pretrained_imagenet=False)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.to(device).eval()
    detector = HaarCascadeDetector()

    out = Path(args.out); (out/"logs").mkdir(parents=True, exist_ok=True); (out/"metrics").mkdir(exist_ok=True)

    vids = sorted([p for p in Path(args.videos).iterdir()
                   if p.suffix.lower() in (".mov",".mp4",".avi",".m4v") and not p.name.startswith(".")])
    if not vids:
        sys.exit(f"HATA: {args.videos} icinde video bulunamadi")
    print(f"{len(vids)} video bulundu\n" + "="*100)

    rows = []
    for v in vids:
        scen = v.stem.split("_")[0].upper()
        info = meta.get(scen, {})
        cf  = info.get("change_frame")
        src = NAME2ID.get(info.get("source_label",""))
        tgt = NAME2ID.get(info.get("target_label",""))

        print(f"\n[{scen}] {v.name}  ... model calisiyor", flush=True)
        probs, det, fps, srcfps = extract_probs(v, model, detector, cfg, device)
        print(f"  {len(probs)} kare, kaynak {srcfps:.1f} FPS, isleme {fps:.1f} FPS")

        for m in METHODS:
            labels, stream, conf = apply_method(probs, m, args.alpha, args.window)
            lfr, msl, pj, fddr = temporal_metrics(labels, stream, det)
            rd = reaction_delay(labels, cf, tgt)
            acc, pre, post, n = frame_accuracy(labels, cf, src, tgt, args.margin)

            r = dict(participant=args.participant, scenario=scen, method=m,
                     frames=len(probs), LFR=lfr, MSL=msl, PJ_L1=pj, FDDR=fddr,
                     RD_frames=rd, frame_acc=acc, pre_acc=pre, post_acc=post,
                     n_scored=n, mean_conf=float(np.mean([c for c in conf if c>0]) or 0),
                     avg_FPS=fps, change_frame=cf,
                     source=info.get("source_label"), target=info.get("target_label"))
            rows.append(r)
            (out/"metrics"/f"{args.participant}_{scen}_{m}.json").write_text(json.dumps(r, indent=2))

            with (out/"logs"/f"{args.participant}_{scen}_{m}.csv").open("w") as f:
                f.write("frame,face_detected,label,confidence," + ",".join(f"p{i}" for i in range(8)) + "\n")
                for i,(l,s,c,d) in enumerate(zip(labels,stream,conf,det)):
                    pv = ",".join(f"{x:.8f}" for x in (s if s is not None else np.zeros(8)))
                    f.write(f"{i},{d},{l},{c:.6f},{pv}\n")

            fa = f"{acc:.4f}" if acc is not None else "  -   "
            print(f"    {m:7} LFR={lfr:.4f}  MSL={msl:7.2f}  PJ={pj:.4f}  FDDR={fddr:.4f}"
                  f"  RD={str(rd):>5}  frame-acc={fa}")

    # ---------------------------------------------------------------- ozet
    print("\n" + "="*100)
    print("OZET  (senaryolar ortalamasi)")
    print("="*100)
    print(f"{'Method':<9}{'LFR':>9}{'MSL':>9}{'PJ-L1':>9}{'FDDR':>9}{'frame-acc':>11}")
    summary = {}
    for m in METHODS:
        s = [r for r in rows if r["method"] == m]
        g = lambda k: [r[k] for r in s if r[k] is not None]
        f = lambda k: float(np.mean(g(k))) if g(k) else None
        summary[m] = dict(LFR=f("LFR"), MSL=f("MSL"), PJ_L1=f("PJ_L1"),
                          FDDR=f("FDDR"), frame_acc=f("frame_acc"))
        a = summary[m]["frame_acc"]
        print(f"{m:<9}{summary[m]['LFR']:>9.4f}{summary[m]['MSL']:>9.2f}"
              f"{summary[m]['PJ_L1']:>9.4f}{summary[m]['FDDR']:>9.4f}"
              f"{(f'{a:.4f}' if a is not None else '-'):>11}")

    base = summary["none"]
    print("\nHam cikarima gore iyilesme:")
    for m in ["ema","voting","hybrid"]:
        s = summary[m]
        print(f"  {m:7} LFR -%{100*(1-s['LFR']/base['LFR']):.2f}   "
              f"MSL x{s['MSL']/base['MSL']:.2f}   PJ-L1 -%{100*(1-s['PJ_L1']/base['PJ_L1']):.2f}")

    json.dump(dict(rows=rows, summary=summary), (out/"summary.json").open("w"), indent=2)

    # LaTeX
    with (out/"table.tex").open("w") as f:
        f.write("% run_all.py tarafindan uretildi\n\\begin{tabular}{llrrrrr}\n\\toprule\n")
        f.write("\\textbf{S} & \\textbf{Method} & \\textbf{LFR} & \\textbf{MSL} & "
                "\\textbf{PJ-L1} & \\textbf{FDDR} & \\textbf{Frame Acc.}\\\\\n\\midrule\n")
        for r in rows:
            a = f"{r['frame_acc']:.4f}" if r["frame_acc"] is not None else "---"
            pj = f"{r['PJ_L1']:.4f}" if r["method"] in ("none","ema","hybrid") else "---"
            f.write(f"{r['scenario']} & {r['method'].upper()} & {r['LFR']:.4f} & "
                    f"{r['MSL']:.2f} & {pj} & {r['FDDR']:.4f} & {a}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    print(f"\nYazildi: {out}/summary.json, {out}/table.tex, {out}/metrics/, {out}/logs/")


if __name__ == "__main__":
    main()
