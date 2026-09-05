# FER Real-Time Reliability

Code, per-frame logs and computed metrics for the manuscript *Reliability-Oriented
Evaluation of a Real-Time Facial Emotion Recognition System*.

The pipeline fine-tunes a ResNet-50 on an eight-class AffectNet subset at three
adaptation depths, runs it on webcam and recorded video, and evaluates the result
along recognition, detector, temporal-stability, calibration, selective-prediction,
interpretability and cross-dataset axes.

Classes, in output order: `neutral happy sad surprise fear disgust anger contempt`.

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

CPU-only is sufficient. RetinaFace detection additionally requires
`insightface` and `onnxruntime`, both listed in `requirements.txt`.

---

## Data

**AffectNet subset**, used for training and offline evaluation. Public
redistribution at <https://www.kaggle.com/datasets/mstjebashazida/affectnet>.
The original database is licensed by its authors and may be requested at
<http://mohammadmahoor.com/affectnet/>. Splits are fixed by
`data/affectnet_subset/train.csv` and `val.csv`, 15,579 training and 12,596
validation images, and are reproducible from the distribution's directory
structure with `datasets/make_splits_from_labels_csv.py`.

**FER2013**, used for cross-dataset evaluation only, never for training.
<https://www.kaggle.com/datasets/msambare/fer2013>. Build the index with
`datasets/make_fer2013_csv.py` after setting `FER2013_ROOT` inside it.

**Video recordings** are not released. They contain identifiable facial imagery
of the authors. Every number derived from them is reproducible from the
per-frame logs in `results_P1/` and `results_P2/`, which contain the full
softmax output for every frame.

---

## Reproducing the manuscript

### Offline training and evaluation

```bash
python -m experiments.run_offline_affectnet --data_dir /path/to/affectnet_subset --epochs 3
```

Writes classification reports and confusion matrices to `results/reports/`.
Three adaptation depths are compared: frozen backbone, Layer4+FC, and
Layer3+Layer4+FC. The manuscript reports Layer3+Layer4+FC throughout,
checkpoint `resnet50_layer3_finetuned_best.pt`.

### Cross-dataset transfer to FER2013

```bash
python evaluate_fer2013.py \
    --data-root /path/to/fer2013 \
    --csv fer2013_test.csv \
    --ckpt checkpoints/resnet50_layer3_finetuned_best.pt
```

Reports two decision rules from one forward pass. With all eight logits
retained, accuracy is 24.42% and Macro-F1 0.2308, and contempt accounts for
17.90% of predictions, every one of which is necessarily wrong because FER2013
has no such class. With the contempt logit masked before the arg-max, accuracy
is 32.11% and Macro-F1 0.2813. Macro-F1 is averaged over the seven classes that
have support in FER2013.

> **Note on an earlier version of this script.** The version published before
> this commit loaded `best_layer4_fc_affectnet_resnet50.pth` rather than the
> Layer3 checkpoint the manuscript reports, and averaged Macro-F1 over all
> eight AffectNet classes, so the unsupported contempt class contributed a hard
> zero to the mean. Together those two defects produced 26.54% accuracy and
> 0.2168 Macro-F1. Both are corrected here.

### Real-time scenarios, original single-participant pipeline

```bash
python -m experiments.run_realtime_scenarios --detector haar      --smoothing none
python -m experiments.run_realtime_scenarios --detector retinaface --smoothing hybrid
```

Writes `results/logs/*.csv` and `results/metrics/*.json`. This entry point
processes the video once per smoothing method and backs the detector ablation
and the parameter-sensitivity study.

### Two-participant re-analysis, unified pipeline

```bash
python run_all.py \
    --videos /path/to/videos \
    --weights checkpoints/resnet50_layer3_finetuned_best.pt \
    --participant P2 \
    --out results_P2
```

`run_all.py` is the implementation described in the two-participant section of
the manuscript. It differs from `run_realtime_scenarios` in three ways.

1. The classifier runs **once per sequence** and all four temporal modes are
   derived from the same frame-level softmax outputs, so a difference between
   two methods cannot come from a difference between two recordings.
2. Probability Jitter is computed from the probability stream each mode
   actually maintains, the raw softmax for None and Voting and the EMA
   distribution for EMA and Hybrid. `smoothing/voting.py` returns a one-hot
   vector by design, so reading PJ-L1 off its output makes PJ-L1 exactly twice
   the Label Flip Rate rather than a probability measurement.
3. Frame-level accuracy is computed against reference labels derived from the
   scripted transition point in `scenarios.json`, excluding a symmetric band of
   `--margin` frames (default 15) around the change frame.

`scenarios.json` holds, per scenario, the transition frame and the source and
target expression. `change_frame` is expressed in frames, so 30 FPS with a
change at the ten-second mark means 300. `run_all.py` prints each video's true
FPS on load.

### Paired significance testing

```bash
python bootstrap_all.py
```

Boundary-aware moving-block bootstrap over `results_P1/` and `results_P2/`.
Contiguous 30-frame blocks, 2000 resamples, consecutive differences taken only
within a block so that artificial transitions at block joins are excluded.
Paired contrasts reuse the same resampled block indices for both methods.
Writes `bootstrap_all.json`.

---

## What is in this repository

| Path | Contents |
|---|---|
| `config.py` | class list, input size, normalisation, default smoothing parameters |
| `model/` | ResNet-50 wrapper and adaptation-depth control |
| `detection/` | Haar cascade and RetinaFace detector wrappers behind one interface |
| `smoothing/` | None, EMA, majority voting, hybrid |
| `datasets/`, `data/` | dataset classes and the fixed CSV splits |
| `train/`, `experiments/` | training and scenario entry points |
| `tools/` | calibration, temperature scaling, selective prediction, bootstrap, transition-frame helpers |
| `interpretability/` | Grad-CAM |
| `evaluate_fer2013.py` | cross-dataset evaluation, corrected |
| `run_all.py` | unified two-participant processing |
| `bootstrap_all.py` | paired moving-block bootstrap |
| `scenarios.json` | transition frames and expression labels |
| `results/` | offline reports, calibration, Grad-CAM, original real-time logs and metrics |
| `results_P1/`, `results_P2/` | per-frame logs and metrics for the two-participant re-analysis |
| `idap_*/` | parameter-sensitivity, window, confidence and transition sub-studies |

### Log format

Each row of `results_P*/logs/<participant>_<scenario>_<method>.csv` is one frame.

| Column | Meaning |
|---|---|
| `frame` | zero-based frame index |
| `face_detected` | 1 if the detector returned a face, else 0 |
| `label` | emitted class index after smoothing, `-1` when no face was detected |
| `confidence` | probability of the emitted class |
| `p0` … `p7` | full probability vector for that frame, in class order |

Because the probability vector is stored for every frame, every metric in the
manuscript can be recomputed from these files alone without rerunning the model.

---

## Metrics

| Metric | Definition |
|---|---|
| LFR | Label Flip Rate, share of consecutive valid frame pairs whose emitted labels differ |
| MSL | Mean Segment Length, mean run length of a constant emitted label, in frames |
| PJ-L1 | Probability Jitter, mean L1 distance between consecutive probability vectors |
| FDDR | Face Detection Drop Rate, share of frames with no detected face |
| RD | Reaction Delay, frames between the reference transition and the first emitted target label |
| Frame accuracy | agreement with segment-level reference labels outside the exclusion band |

---

## Licence and citation

Please cite the manuscript when using this code or the released logs. The
AffectNet and FER2013 datasets remain under the licences of their respective
authors and are not redistributed here.
