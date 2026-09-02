import argparse
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a video frame by frame."
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Path to the video.",
    )

    args = parser.parse_args()

    video_path = Path(args.video).expanduser().resolve()

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    frame_index = 0

    print(f"Video: {video_path.name}")
    print(f"FPS: {fps:.2f}")
    print(f"Frames: {total_frames}")
    print()
    print("Click the video window first.")
    print("Controls:")
    print("SPACE or n = next frame")
    print("b = previous frame")
    print("f = forward 10 frames")
    print("r = backward 10 frames")
    print("p = print current frame")
    print("q = quit")

    while True:
        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index,
        )

        ok, frame = cap.read()

        if not ok or frame is None:
            break

        timestamp = (
            frame_index / fps
            if fps > 0
            else 0.0
        )

        cv2.putText(
            frame,
            f"Frame: {frame_index} | Time: {timestamp:.2f}s",
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow(
            "Transition Frame Inspector",
            frame,
        )

        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break

        elif key in (ord("n"), ord(" ")):
            frame_index = min(
                total_frames - 1,
                frame_index + 1,
            )

        elif key == ord("b"):
            frame_index = max(
                0,
                frame_index - 1,
            )

        elif key == ord("f"):
            frame_index = min(
                total_frames - 1,
                frame_index + 10,
            )

        elif key == ord("r"):
            frame_index = max(
                0,
                frame_index - 10,
            )

        elif key == ord("p"):
            print(
                f"Selected frame: {frame_index}",
                flush=True,
            )

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()