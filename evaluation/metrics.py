from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


INVALID_LABEL = -1


@dataclass(frozen=True)
class TemporalMetrics:
    lfr: Optional[float]
    msl: Optional[float]
    pj_l1: Optional[float]
    fddr: Optional[float]
    rd_frames: Optional[int] = None
    avg_fps: Optional[float] = None


def _filter_invalid_labels(
    labels: Sequence[int],
    invalid_label: int = INVALID_LABEL,
) -> List[int]:
    return [
        int(label)
        for label in labels
        if int(label) != invalid_label
    ]


def _filter_valid_probs(
    probs: Sequence[Optional[np.ndarray]],
) -> List[np.ndarray]:
    valid_probs: List[np.ndarray] = []

    for probability in probs:
        if probability is None:
            continue

        array = np.asarray(probability)

        if array.size == 0:
            continue

        valid_probs.append(array)

    return valid_probs


def label_flip_rate(
    labels: Sequence[int],
    invalid_label: int = INVALID_LABEL,
) -> Optional[float]:
    clean_labels = _filter_invalid_labels(
        labels,
        invalid_label=invalid_label,
    )

    if len(clean_labels) < 2:
        return None

    flips = sum(
        1
        for index in range(1, len(clean_labels))
        if clean_labels[index] != clean_labels[index - 1]
    )

    return flips / (len(clean_labels) - 1)


def mean_segment_length(
    labels: Sequence[int],
    invalid_label: int = INVALID_LABEL,
) -> Optional[float]:
    clean_labels = _filter_invalid_labels(
        labels,
        invalid_label=invalid_label,
    )

    if not clean_labels:
        return None

    segment_lengths: List[int] = []
    current_label = clean_labels[0]
    current_length = 1

    for label in clean_labels[1:]:
        if label == current_label:
            current_length += 1
        else:
            segment_lengths.append(current_length)
            current_label = label
            current_length = 1

    segment_lengths.append(current_length)

    return float(np.mean(segment_lengths))


def prob_jitter_l1(
    probs: Sequence[Optional[np.ndarray]],
    *,
    normalize: bool = False,
) -> Optional[float]:
    clean_probs = _filter_valid_probs(probs)

    if len(clean_probs) < 2:
        return None

    reference_shape = clean_probs[0].shape

    for index, probability in enumerate(clean_probs):
        if probability.shape != reference_shape:
            raise ValueError(
                "Inconsistent probability shapes: "
                f"probs[0].shape={reference_shape}, "
                f"probs[{index}].shape={probability.shape}"
            )

    differences: List[float] = []

    previous = clean_probs[0].astype(
        np.float64,
        copy=False,
    )

    if normalize:
        total = previous.sum()

        if total != 0:
            previous = previous / total

    for probability in clean_probs[1:]:
        current = probability.astype(
            np.float64,
            copy=False,
        )

        if normalize:
            total = current.sum()

            if total != 0:
                current = current / total

        differences.append(
            float(np.abs(current - previous).sum())
        )

        previous = current

    return float(np.mean(differences))


def face_detection_drop_rate(
    face_detected: Sequence[int],
) -> Optional[float]:
    if not face_detected:
        return None

    detections = np.asarray(
        face_detected,
        dtype=np.float64,
    )

    return float(
        1.0 - detections.sum() / detections.size
    )


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
    Return the response delay in original video frames.

    When target_label is supplied, the delay is the first frame
    at which stable_k consecutive valid predictions equal that
    target label.

    Otherwise, the first stable sequence after change_frame is
    used. If require_change_from_pre is True, that stable label
    must differ from the last valid label before change_frame.
    """

    labels_list = [int(label) for label in labels]
    number_of_frames = len(labels_list)

    if (
        number_of_frames == 0
        or change_frame < 0
        or change_frame >= number_of_frames
        or stable_k <= 0
    ):
        return None

    clean: List[Tuple[int, int]] = [
        (frame_index, label)
        for frame_index, label in enumerate(labels_list)
        if label != invalid_label
    ]

    if len(clean) < stable_k:
        return None

    start_position = next(
        (
            position
            for position, (frame_index, _) in enumerate(clean)
            if frame_index >= change_frame
        ),
        None,
    )

    if start_position is None:
        return None

    previous_label: Optional[int] = None

    if require_change_from_pre:
        for frame_index, label in reversed(
            clean[:start_position]
        ):
            if frame_index < change_frame:
                previous_label = label
                break

    for position in range(
        start_position,
        len(clean) - stable_k + 1,
    ):
        window = [
            clean[position + offset][1]
            for offset in range(stable_k)
        ]

        window_start_frame = clean[position][0]

        if target_label is not None:
            if all(
                label == target_label
                for label in window
            ):
                return window_start_frame - change_frame

        else:
            stable_label = window[0]

            if all(
                label == stable_label
                for label in window
            ):
                if (
                    require_change_from_pre
                    and previous_label is not None
                    and stable_label == previous_label
                ):
                    continue

                return window_start_frame - change_frame

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
    lfr = label_flip_rate(
        labels,
        invalid_label=invalid_label,
    )

    msl = mean_segment_length(
        labels,
        invalid_label=invalid_label,
    )

    pj_l1 = prob_jitter_l1(
        probs,
        normalize=pj_normalize,
    )

    fddr = face_detection_drop_rate(
        face_detected
    )

    rd_frames: Optional[int] = None

    if change_frame is not None:
        rd_frames = reaction_delay_frames(
            labels,
            change_frame,
            stable_k=stable_k,
            invalid_label=invalid_label,
            target_label=rd_target_label,
            require_change_from_pre=rd_require_change_from_pre,
        )

    avg_fps = (
        float(np.mean(fps_series))
        if fps_series and len(fps_series) > 0
        else None
    )

    return TemporalMetrics(
        lfr=lfr,
        msl=msl,
        pj_l1=pj_l1,
        fddr=fddr,
        rd_frames=rd_frames,
        avg_fps=avg_fps,
    )


def as_dict(
    metrics: TemporalMetrics,
) -> Dict[str, Any]:
    return {
        "LFR": metrics.lfr,
        "MSL": metrics.msl,
        "PJ_L1": metrics.pj_l1,
        "FDDR": metrics.fddr,
        "RD_frames": metrics.rd_frames,
        "avg_FPS": metrics.avg_fps,
    }