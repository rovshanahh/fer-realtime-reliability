'''from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np

@dataclass
class TemporalMetrics:
    lfr: float
    msl: float
    pj_l1: float
    fddr: float
    rd_frames: Optional[float] = None
    avg_fps: Optional[float] = None

def label_flip_rate(labels: List[int]) -> float:
    if len(labels) < 2:
        return 0.0
    flips = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i-1])
    return flips / (len(labels) - 1)

def mean_segment_length(labels: List[int]) -> float:
    if not labels:
        return 0.0
    seg_lens = []
    curr = labels[0]
    run = 1
    for y in labels[1:]:
        if y == curr:
            run += 1
        else:
            seg_lens.append(run)
            curr = y
            run = 1
    seg_lens.append(run)
    return float(np.mean(seg_lens)) if seg_lens else 0.0

def prob_jitter_l1(probs: List[np.ndarray]) -> float:
    if len(probs) < 2:
        return 0.0
    diffs = [np.abs(probs[i] - probs[i-1]).sum() for i in range(1, len(probs))]
    return float(np.mean(diffs))

def face_detection_drop_rate(face_detected: List[int]) -> float:
    if not face_detected:
        return 1.0
    return float(1.0 - (np.sum(face_detected) / len(face_detected)))

def reaction_delay_frames(labels: List[int], change_frame: int, stable_k: int = 5) -> Optional[int]:
    if change_frame < 0 or change_frame >= len(labels):
        return None

    # Remove invalid frames (-1)
    clean = [(i, y) for i, y in enumerate(labels) if y != -1]

    if len(clean) < stable_k:
        return None

    # Map original change_frame to cleaned index
    clean_indices = [i for i, _ in clean]
    if change_frame not in clean_indices:
        return None

    start = clean_indices.index(change_frame)

    for t in range(start, len(clean) - stable_k + 1):
        window = [clean[t + k][1] for k in range(stable_k)]
        if all(y == window[0] for y in window):
            return t - start

    return None

def compute_metrics(
    labels: List[int],
    probs: List[np.ndarray],
    face_detected: List[int],
    fps_series: Optional[List[float]] = None,
    change_frame: Optional[int] = None,
    stable_k: int = 5
) -> TemporalMetrics:
    lfr = label_flip_rate(labels)
    msl = mean_segment_length(labels)
    pj = prob_jitter_l1(probs)
    fddr = face_detection_drop_rate(face_detected)
    rd = reaction_delay_frames(labels, change_frame, stable_k) if change_frame is not None else None
    avg_fps = float(np.mean(fps_series)) if fps_series else None
    return TemporalMetrics(lfr=lfr, msl=msl, pj_l1=pj, fddr=fddr, rd_frames=rd, avg_fps=avg_fps)

def as_dict(tm: TemporalMetrics) -> Dict[str, Any]:
    return {
        "LFR": tm.lfr,
        "MSL": tm.msl,
        "PJ_L1": tm.pj_l1,
        "FDDR": tm.fddr,
        "RD_frames": tm.rd_frames,
        "avg_FPS": tm.avg_fps,
    }'''


from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Sequence, Tuple
import numpy as np


INVALID_LABEL = -1  # convention: -1 means "no prediction / invalid frame"


@dataclass(frozen=True)
class TemporalMetrics:
    lfr: Optional[float]          # Label Flip Rate
    msl: Optional[float]          # Mean Segment Length
    pj_l1: Optional[float]        # Probability Jitter (L1)
    fddr: Optional[float]         # Face Detection Drop Rate
    rd_frames: Optional[int] = None   # Reaction Delay in original frame units
    avg_fps: Optional[float] = None


def _filter_invalid_labels(labels: Sequence[int], invalid_label: int = INVALID_LABEL) -> List[int]:
    return [int(y) for y in labels if int(y) != invalid_label]


def _filter_valid_probs(
    probs: Sequence[Optional[np.ndarray]],
) -> List[np.ndarray]:
    # Keep only non-empty numpy arrays
    out: List[np.ndarray] = []
    for p in probs:
        if p is None:
            continue
        a = np.asarray(p)
        if a.size == 0:
            continue
        out.append(a)
    return out


def label_flip_rate(labels: Sequence[int], invalid_label: int = INVALID_LABEL) -> Optional[float]:
    clean = _filter_invalid_labels(labels, invalid_label=invalid_label)
    if len(clean) < 2:
        return None
    flips = sum(1 for i in range(1, len(clean)) if clean[i] != clean[i - 1])
    return flips / (len(clean) - 1)


def mean_segment_length(labels: Sequence[int], invalid_label: int = INVALID_LABEL) -> Optional[float]:
    clean = _filter_invalid_labels(labels, invalid_label=invalid_label)
    if not clean:
        return None

    seg_lens: List[int] = []
    curr = clean[0]
    run = 1

    for y in clean[1:]:
        if y == curr:
            run += 1
        else:
            seg_lens.append(run)
            curr = y
            run = 1

    seg_lens.append(run)
    return float(np.mean(seg_lens)) if seg_lens else None


def prob_jitter_l1(
    probs: Sequence[Optional[np.ndarray]],
    *,
    normalize: bool = False,
) -> Optional[float]:
    """
    Mean L1 difference between consecutive probability vectors.

    normalize=False:
      Uses raw output vectors as-is.

    normalize=True:
      Renormalizes each vector to sum to 1 (useful if inputs are logits or unnormalized scores).
    """
    clean = _filter_valid_probs(probs)
    if len(clean) < 2:
        return None

    # Ensure consistent shapes across time
    shape0 = clean[0].shape
    for i, p in enumerate(clean):
        if p.shape != shape0:
            raise ValueError(f"Inconsistent prob shapes: probs[0].shape={shape0}, probs[{i}].shape={p.shape}")

    diffs: List[float] = []
    prev = clean[0].astype(np.float64, copy=False)

    if normalize:
        s = prev.sum()
        if s != 0:
            prev = prev / s

    for p in clean[1:]:
        curr = p.astype(np.float64, copy=False)
        if normalize:
            s = curr.sum()
            if s != 0:
                curr = curr / s
        diffs.append(float(np.abs(curr - prev).sum()))
        prev = curr

    return float(np.mean(diffs)) if diffs else None


def face_detection_drop_rate(face_detected: Sequence[int]) -> Optional[float]:
    """
    face_detected is expected to be a 0/1 sequence (0=no face, 1=face).
    Returns None if the sequence is empty (unknown / not logged).
    """
    if not face_detected:
        return None
    arr = np.asarray(face_detected, dtype=np.float64)
    return float(1.0 - (arr.sum() / arr.size))


def reaction_delay_frames(
    labels: Sequence[int],
    change_frame: int,
    stable_k: int = 5,
    *,
    invalid_label: int = INVALID_LABEL,
    target_label: Optional[int] = None,
    require_change_from_pre: bool = False,
) -> Optional[int]:
    """
    Reaction delay in ORIGINAL frame indices (not "cleaned-step" indices).

    Definitions supported:
    - If target_label is provided:
        delay is the first time after change_frame that we see stable_k consecutive valid labels == target_label.
    - If target_label is None:
        delay is the first time after change_frame that we see stable_k consecutive valid labels that are all equal.
        If require_change_from_pre=True, the stable label must differ from the label BEFORE change_frame
        (first valid label before change_frame).
    Invalid frames (labels == invalid_label) are ignored when forming the stable window.
    """
    labels_list = [int(y) for y in labels]
    n = len(labels_list)
    if n == 0 or change_frame < 0 or change_frame >= n or stable_k <= 0:
        return None

    # Build (orig_index, label) list for valid frames only
    clean: List[Tuple[int, int]] = [(i, y) for i, y in enumerate(labels_list) if y != invalid_label]
    if len(clean) < stable_k:
        return None

    # Start at the first valid frame at or after change_frame
    start_pos = next((idx for idx, (orig_i, _) in enumerate(clean) if orig_i >= change_frame), None)
    if start_pos is None:
        return None

    pre_label: Optional[int] = None
    if require_change_from_pre:
        # Find last valid label before change_frame
        for orig_i, y in reversed(clean[:start_pos]):
            if orig_i < change_frame:
                pre_label = y
                break

    for t in range(start_pos, len(clean) - stable_k + 1):
        window_labels = [clean[t + k][1] for k in range(stable_k)]
        window_orig_start = clean[t][0]

        if target_label is not None:
            if all(y == target_label for y in window_labels):
                return window_orig_start - change_frame
        else:
            if all(y == window_labels[0] for y in window_labels):
                if require_change_from_pre and pre_label is not None and window_labels[0] == pre_label:
                    continue
                return window_orig_start - change_frame

    return None


def compute_metrics(
    labels: Sequence[int],
    probs: Sequence[Optional[np.ndarray]],
    face_detected: Sequence[int],
    fps_series: Optional[Sequence[float]] = None,
    change_frame: Optional[int] = None,
    stable_k: int = 5,
    *,
    invalid_label: int = INVALID_LABEL,
    pj_normalize: bool = False,
    rd_target_label: Optional[int] = None,
    rd_require_change_from_pre: bool = False,
) -> TemporalMetrics:
    lfr = label_flip_rate(labels, invalid_label=invalid_label)
    msl = mean_segment_length(labels, invalid_label=invalid_label)
    pj = prob_jitter_l1(probs, normalize=pj_normalize)
    fddr = face_detection_drop_rate(face_detected)

    rd: Optional[int] = None
    if change_frame is not None:
        rd = reaction_delay_frames(
            labels,
            change_frame,
            stable_k=stable_k,
            invalid_label=invalid_label,
            target_label=rd_target_label,
            require_change_from_pre=rd_require_change_from_pre,
        )

    avg_fps = float(np.mean(fps_series)) if fps_series and len(fps_series) > 0 else None

    return TemporalMetrics(
        lfr=lfr,
        msl=msl,
        pj_l1=pj,
        fddr=fddr,
        rd_frames=rd,
        avg_fps=avg_fps,
    )


def as_dict(tm: TemporalMetrics) -> Dict[str, Any]:
    return {
        "LFR": tm.lfr,
        "MSL": tm.msl,
        "PJ_L1": tm.pj_l1,
        "FDDR": tm.fddr,
        "RD_frames": tm.rd_frames,
        "avg_FPS": tm.avg_fps,
    }