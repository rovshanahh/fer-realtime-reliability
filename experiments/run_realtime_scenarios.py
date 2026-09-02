import argparse
import json
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from torchvision import transforms

from config import FERConfig
from model.resnet50_fer import ResNet50FER
from detection.haar import HaarCascadeDetector
from evaluation.metrics import compute_metrics, as_dict

from smoothing.ema import EMASmoother
from smoothing.voting import VotingSmoother
from smoothing.hybrid import HybridSmoother

from interpretability.gradcam import GradCAM, overlay_cam_on_bgr


INVALID_LABEL = -1


def load_detector(name: str):
    if name == "haar":
        return HaarCascadeDetector()

    if name == "retinaface":
        from detection.retinaface_det import RetinaFaceDetector

        return RetinaFaceDetector()

    raise ValueError(f"Unknown detector: {name}")


def load_smoother(name: str, cfg: FERConfig):
    if name == "none":
        return None

    if name == "ema":
        return EMASmoother(alpha=cfg.ema_alpha)

    if name == "voting":
        return VotingSmoother(window=cfg.voting_window)

    if name == "hybrid":
        return HybridSmoother(
            alpha=cfg.ema_alpha,
            window=cfg.voting_window,
        )

    raise ValueError(f"Unknown smoothing: {name}")


def preprocess_face(
    bgr: np.ndarray,
    bbox,
    cfg: FERConfig,
):
    x1, y1, x2, y2 = bbox
    height, width = bgr.shape[:2]

    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(width, int(x2))
    y2 = min(height, int(y2))

    if x2 <= x1 or y2 <= y1:
        return None, None

    crop_bgr = bgr[y1:y2, x1:x2]

    if crop_bgr.size == 0:
        return None, None

    rgb = cv2.cvtColor(
        crop_bgr,
        cv2.COLOR_BGR2RGB,
    )

    rgb = cv2.resize(
        rgb,
        (cfg.input_size, cfg.input_size),
        interpolation=cv2.INTER_LINEAR,
    )

    return crop_bgr, rgb


def write_json(
    path: Path,
    obj,
) -> None:
    path.write_text(
        json.dumps(obj, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Real-time FER experiment runner using a webcam "
            "or recorded video."
        )
    )

    parser.add_argument(
        "--detector",
        choices=["haar", "retinaface"],
        default="haar",
    )

    parser.add_argument(
        "--smoothing",
        choices=["none", "ema", "voting", "hybrid"],
        default="none",
    )

    parser.add_argument(
        "--weights",
        required=True,
        help="Path to FER checkpoint.",
    )

    parser.add_argument(
        "--out_dir",
        default="results",
    )

    parser.add_argument(
        "--scenario",
        default="S1",
    )

    parser.add_argument(
        "--seconds",
        type=int,
        default=20,
        help=(
            "Duration of webcam experiments. "
            "Ignored when --video is used."
        ),
    )

    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help=(
            "Optional path to a recorded video. "
            "If omitted, the webcam is used."
        ),
    )

    parser.add_argument(
        "--voting_window",
        type=int,
        default=None,
        help="Override the voting window defined in config.py.",
    )

    parser.add_argument(
        "--ema_alpha",
        type=float,
        default=None,
        help="Override the EMA alpha defined in config.py.",
    )

    parser.add_argument(
        "--change_frame",
        type=int,
        default=None,
        help="Known frame at which the target expression changes.",
    )

    parser.add_argument(
        "--rd_target_label",
        type=int,
        default=None,
        help=(
            "Target emotion class used to calculate reaction delay. "
            "neutral=0, happy=1, sad=2, surprise=3, fear=4, "
            "disgust=5, anger=6, contempt=7."
        ),
    )

    parser.add_argument(
        "--no_display",
        action="store_true",
        help="Process without displaying the video window.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123,
    )

    args = parser.parse_args()

    cfg = FERConfig()

    if args.voting_window is not None:
        if args.voting_window <= 0:
            raise ValueError(
                "--voting_window must be greater than 0"
            )

        object.__setattr__(
            cfg,
            "voting_window",
            args.voting_window,
        )

        print(
            f"Voting window override: {cfg.voting_window}"
        )

    if args.ema_alpha is not None:
        if not 0.0 < args.ema_alpha <= 1.0:
            raise ValueError(
                "--ema_alpha must be greater than 0 "
                "and less than or equal to 1"
            )

        object.__setattr__(
            cfg,
            "ema_alpha",
            args.ema_alpha,
        )

        print(
            f"EMA alpha override: {cfg.ema_alpha}"
        )

    if args.rd_target_label is not None:
        if not 0 <= args.rd_target_label < cfg.num_classes:
            raise ValueError(
                f"--rd_target_label must be between 0 "
                f"and {cfg.num_classes - 1}"
            )

        print(
            "Reaction-delay target: "
            f"{args.rd_target_label} "
            f"({cfg.class_names[args.rd_target_label]})"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    out_dir = Path(args.out_dir)

    (out_dir / "logs").mkdir(
        parents=True,
        exist_ok=True,
    )

    (out_dir / "metrics").mkdir(
        parents=True,
        exist_ok=True,
    )

    (out_dir / "gradcam").mkdir(
        parents=True,
        exist_ok=True,
    )

    detector = load_detector(args.detector)
    smoother = load_smoother(
        args.smoothing,
        cfg,
    )

    model = ResNet50FER(
        num_classes=cfg.num_classes,
        pretrained_imagenet=True,
    ).to(device)

    weights_path = Path(args.weights)

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {weights_path}"
        )

    state_dict = torch.load(
        str(weights_path),
        map_location=device,
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    print(
        f"Loaded FER weights: {weights_path}"
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg.imagenet_mean,
                std=cfg.imagenet_std,
            ),
        ]
    )

    gradcam = GradCAM(
        model,
        target_layer=model.backbone.layer4[-1].conv3,
    )

    if args.video is not None:
        video_path = Path(
            args.video
        ).expanduser().resolve()

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

        cap = cv2.VideoCapture(
            str(video_path)
        )

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open video: {video_path}"
            )

        source_fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        source_frame_count = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        print(
            f"Processing recorded video: {video_path}"
        )

        print(
            f"Source FPS: {source_fps:.2f}"
        )

        print(
            f"Source frames: {source_frame_count}"
        )

    else:
        cap = cv2.VideoCapture(
            cfg.cam_index,
            cv2.CAP_AVFOUNDATION,
        )

        if not cap.isOpened():
            raise RuntimeError(
                "Cannot open webcam"
            )

        print(
            f"Processing webcam for "
            f"{args.seconds} seconds"
        )

    labels: List[int] = []
    probabilities: List[np.ndarray] = []
    face_detected: List[int] = []

    log_path = (
        out_dir
        / "logs"
        / (
            f"{args.scenario}_"
            f"{args.detector}_"
            f"{args.smoothing}.csv"
        )
    )

    start_time = time.time()
    frame_idx = 0

    with open(
        log_path,
        "w",
        encoding="utf-8",
    ) as log_file:
        probability_columns = ",".join(
            f"p{i}" for i in range(cfg.num_classes)
        )

        log_file.write(
            "frame,face_detected,y_raw,y_smooth,"
            "confidence_raw,confidence_smooth,"
            f"{probability_columns}\n"
        )

        while True:
            if (
                args.video is None
                and time.time() - start_time
                >= args.seconds
            ):
                break

            ok, frame = cap.read()

            if (
                not ok
                or frame is None
                or frame.size == 0
            ):
                break

            bbox = detector.detect(frame)

            detected = (
                1
                if bbox is not None
                else 0
            )

            face_detected.append(detected)

            y_raw = INVALID_LABEL
            y_smooth = INVALID_LABEL

            confidence_raw = 0.0
            confidence_smooth = 0.0

            probability_used = np.zeros(
                cfg.num_classes,
                dtype=np.float32,
            )

            if bbox is not None:
                face_crop_bgr, face_rgb = preprocess_face(
                    frame,
                    bbox,
                    cfg,
                )

                if face_rgb is not None:
                    model_input = (
                        transform(face_rgb)
                        .unsqueeze(0)
                        .to(device)
                    )

                    with torch.no_grad():
                        logits = model(
                            model_input
                        )

                        probability_raw = (
                            torch.softmax(
                                logits,
                                dim=1,
                            )
                            .cpu()
                            .numpy()[0]
                            .astype(np.float32)
                        )

                    y_raw = int(
                        np.argmax(
                            probability_raw
                        )
                    )

                    confidence_raw = float(
                        np.max(
                            probability_raw
                        )
                    )

                    probability_used = probability_raw

                    if smoother is not None:
                        probability_used = np.asarray(
                            smoother.update(probability_raw),
                            dtype=np.float32,
                        )

                        y_smooth = int(
                            np.argmax(probability_used)
                        )

                        if args.smoothing == "hybrid":
                            if smoother.last_ema_probs is None:
                                confidence_smooth = confidence_raw
                            else:
                                confidence_smooth = float(
                                    np.max(smoother.last_ema_probs)
                                )

                        elif args.smoothing == "ema":
                            confidence_smooth = float(
                                np.max(probability_used)
                            )

                        elif args.smoothing == "voting":
                            # Hard voting returns a one-hot vector, so use the
                            # classifier confidence rather than the artificial 1.0.
                            confidence_smooth = confidence_raw

                    else:
                        y_smooth = y_raw
                        confidence_smooth = confidence_raw

                    x1, y1, x2, y2 = [
                        int(value)
                        for value in bbox
                    ]

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

                    label_name = cfg.class_names[
                        y_smooth
                    ]

                    cv2.putText(
                        frame,
                        (
                            f"{label_name} "
                            f"{confidence_smooth:.2f}"
                        ),
                        (
                            x1,
                            max(0, y1 - 10),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

            labels.append(y_smooth)

            probabilities.append(
                probability_used
            )

            probability_values = ",".join(
                f"{float(value):.8f}"
                for value in probability_used
            )

            log_file.write(
                f"{frame_idx},"
                f"{detected},"
                f"{y_raw},"
                f"{y_smooth},"
                f"{confidence_raw:.6f},"
                f"{confidence_smooth:.6f},"
                f"{probability_values}\n"
            )

            if not args.no_display:
                cv2.imshow(
                    "FER",
                    frame,
                )

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if (
                    key == ord("g")
                    and bbox is not None
                    and y_smooth != INVALID_LABEL
                ):
                    face_crop_bgr, face_rgb = preprocess_face(
                        frame,
                        bbox,
                        cfg,
                    )

                    if face_rgb is not None:
                        model_input = (
                            transform(face_rgb)
                            .unsqueeze(0)
                            .to(device)
                        )

                        cam = gradcam.generate(
                            model_input,
                            class_idx=y_smooth,
                        )

                        overlay = overlay_cam_on_bgr(
                            face_crop_bgr,
                            cam,
                        )

                        gradcam_path = (
                            out_dir
                            / "gradcam"
                            / (
                                f"{args.scenario}_"
                                f"{args.detector}_"
                                f"{args.smoothing}_"
                                f"{frame_idx}.png"
                            )
                        )

                        cv2.imwrite(
                            str(gradcam_path),
                            overlay,
                        )

                elif key == ord("q"):
                    break

            frame_idx += 1

    cap.release()

    if not args.no_display:
        cv2.destroyAllWindows()

    elapsed_time = max(
        1e-9,
        time.time() - start_time,
    )

    average_fps = float(
        len(labels) / elapsed_time
    )

    if args.change_frame is not None:
        print(
            "\nLabels around change frame:"
        )

        start = max(
            0,
            args.change_frame - 10,
        )

        end = min(
            len(labels),
            args.change_frame + 20,
        )

        print(
            labels[start:end]
        )

    metrics = compute_metrics(
        labels=labels,
        probs=probabilities,
        face_detected=face_detected,
        fps_series=[average_fps],
        change_frame=args.change_frame,
        stable_k=cfg.stable_k,
        rd_target_label=args.rd_target_label,
    )

    metrics_path = (
        out_dir
        / "metrics"
        / (
            f"{args.scenario}_"
            f"{args.detector}_"
            f"{args.smoothing}.json"
        )
    )

    write_json(
        metrics_path,
        as_dict(metrics),
    )

    print(
        as_dict(metrics)
    )

    print(
        f"Processed frames: {len(labels)}"
    )

    print(
        f"Processing FPS: {average_fps:.2f}"
    )

    print(
        f"Log saved to: {log_path}"
    )

    print(
        f"Metrics saved to: {metrics_path}"
    )


if __name__ == "__main__":
    main()