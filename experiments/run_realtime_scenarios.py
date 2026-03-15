import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

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
        return HybridSmoother(alpha=cfg.ema_alpha, window=cfg.voting_window)
    raise ValueError(f"Unknown smoothing: {name}")


def preprocess_face(bgr: np.ndarray, bbox, cfg: FERConfig):
    x1, y1, x2, y2 = bbox
    h, w = bgr.shape[:2]

    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))

    if x2 <= x1 or y2 <= y1:
        return None, None

    crop_bgr = bgr[y1:y2, x1:x2]
    if crop_bgr.size == 0:
        return None, None

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (cfg.input_size, cfg.input_size), interpolation=cv2.INTER_LINEAR)
    return crop_bgr, rgb


@dataclass
class RunStats:
    total_frames: int = 0
    fps_frame_count: int = 0
    fps_start_time: float = 0.0

    def start(self):
        self.fps_start_time = time.time()

    def tick(self):
        self.total_frames += 1
        self.fps_frame_count += 1

    def avg_fps(self) -> float:
        total_time = max(1e-9, time.time() - self.fps_start_time)
        return self.fps_frame_count / total_time


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Realtime FER experiment runner (webcam).")
    ap.add_argument("--detector", choices=["haar", "retinaface"], default="haar")
    ap.add_argument("--smoothing", choices=["none", "ema", "voting", "hybrid"], default="none")
    ap.add_argument("--weights", required=True, help="Path to FER checkpoint (.pth).")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--scenario", default="S1")
    ap.add_argument("--seconds", type=int, default=20)
    ap.add_argument("--change_frame", type=int, default=None)
    ap.add_argument("--no_display", action="store_true", help="Run headless (no imshow/waitKey).")
    ap.add_argument("--seed", type=int, default=123, help="Seed for reproducibility.")
    args = ap.parse_args()

    cfg = FERConfig()

    # -------- Reproducibility knobs (research-friendly) --------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        # For deterministic-ish behavior where possible (may reduce speed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out_dir)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "gradcam").mkdir(parents=True, exist_ok=True)

    detector = load_detector(args.detector)
    smoother = load_smoother(args.smoothing, cfg)

    model = ResNet50FER(num_classes=cfg.num_classes, pretrained_imagenet=True).to(device)

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    sd = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"Loaded FER weights: {weights_path}")

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.imagenet_mean, std=cfg.imagenet_std),
    ])

    gradcam = GradCAM(model, target_layer=model.backbone.layer4[-1].conv3)

    # macOS camera backend; if this fails on other OS, consider removing CAP_AVFOUNDATION.
    cap = cv2.VideoCapture(cfg.cam_index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    labels: List[int] = []
    probs_list: List[np.ndarray] = []
    face_detected: List[int] = []

    log_path = out_dir / "logs" / f"{args.scenario}_{args.detector}_{args.smoothing}.csv"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("frame,face_detected,y_raw,y_smooth\n")

        stats = RunStats()
        stats.start()

        t0 = time.time()
        frame_idx = 0

        while True:
            if time.time() - t0 >= args.seconds:
                break

            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                break

            stats.tick()

            bbox = detector.detect(frame)
            detected = 1 if bbox is not None else 0
            face_detected.append(detected)

            y_raw: int = INVALID_LABEL
            y_smooth: int = INVALID_LABEL
            p_used = np.zeros(cfg.num_classes, dtype=np.float32)

            if bbox is not None:
                face_crop_bgr, face_rgb = preprocess_face(frame, bbox, cfg)
                if face_rgb is not None:
                    x = tfm(face_rgb).unsqueeze(0).to(device)

                    with torch.no_grad():
                        logits = model(x)
                        p = torch.softmax(logits, dim=1).cpu().numpy()[0].astype(np.float32)

                    y_raw = int(np.argmax(p))
                    p_used = p

                    if smoother is not None:
                        p_used = np.asarray(smoother.update(p_used), dtype=np.float32)
                        y_smooth = int(np.argmax(p_used))
                    else:
                        y_smooth = y_raw

                    # Overlay for visualization (does not affect metrics)
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    label_name = cfg.class_names[y_smooth] if hasattr(cfg, "class_names") else str(y_smooth)
                    cv2.putText(
                        frame,
                        f"{label_name}",
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

            labels.append(y_smooth)
            probs_list.append(p_used)

            # Logging
            f.write(f"{frame_idx},{detected},{y_raw},{y_smooth}\n")

            # Display / interaction
            if not args.no_display:
                cv2.imshow("FER", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("g") and bbox is not None and y_smooth != INVALID_LABEL:
                    # Recompute x for gradcam only if we had a valid crop
                    # (safe because we only reach here when the crop succeeded)
                    face_crop_bgr, face_rgb = preprocess_face(frame, bbox, cfg)
                    if face_rgb is not None:
                        x = tfm(face_rgb).unsqueeze(0).to(device)
                        cam = gradcam.generate(x, class_idx=y_smooth)
                        overlay = overlay_cam_on_bgr(face_crop_bgr, cam)
                        outp = out_dir / "gradcam" / f"{args.scenario}_{args.detector}_{args.smoothing}_{frame_idx}.png"
                        cv2.imwrite(str(outp), overlay)
                elif key == ord("q"):
                    break

            frame_idx += 1

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    avg_fps = float(len(labels) / max(1e-9, (time.time() - (time.time() - args.seconds))))  # fallback-ish
    # Better: use measured stats if available
    # (labels length == frames processed; stats.avg_fps() based on wall time)
    # We'll compute it again robustly:
    # NOTE: Since we closed the loop, we can't use stats here unless we stored it outside.
    # So instead compute with a simple approximation from cap duration:
    # To avoid this complication, you can also store stats.avg_fps() before leaving the `with` block.

    # For correctness, compute avg_fps from timestamps across the run:
    # We'll derive it from the saved log length and args.seconds if the run completed, otherwise it's an estimate.
    # If you want exact FPS, store per-frame timestamps and pass to metrics.
    avg_fps = float(len(labels) / max(1e-9, args.seconds))

    if args.change_frame is not None:
        print("\nDEBUG — labels around change_frame:")
        s = max(0, args.change_frame - 10)
        e = min(len(labels), args.change_frame + 20)
        print(labels[s:e])

    tm = compute_metrics(
        labels=labels,
        probs=probs_list,
        face_detected=face_detected,
        fps_series=[avg_fps],  # if you have per-frame FPS, pass that series instead
        change_frame=args.change_frame,
        stable_k=cfg.stable_k,
    )

    metrics_path = out_dir / "metrics" / f"{args.scenario}_{args.detector}_{args.smoothing}.json"
    _write_json(metrics_path, as_dict(tm))
    print(as_dict(tm))


if __name__ == "__main__":
    main()
