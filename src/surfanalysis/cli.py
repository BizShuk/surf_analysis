"""Command-line entry point: surf extract / surf render."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from surfanalysis.extraction.analyzer import FrameAnalyzer
from surfanalysis.extraction.mediapipe_engine import MediaPipeEngine
from surfanalysis.extraction.schema import SessionRecord, SourceInfo

EXIT_OK = 0
EXIT_IO = 1
EXIT_ENGINE = 2
EXIT_DECODE = 3
EXIT_SCHEMA = 4


def _open_video(path: Path) -> tuple[cv2.VideoCapture, SourceInfo]:
    if not path.exists():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cv2 could not open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_ms = (total / fps) * 1000.0 if fps else 0.0
    return cap, SourceInfo(path=str(path), width=width, height=height,
                            fps=fps, total_frames=total, duration_ms=duration_ms)


def _iter_frames(cap: cv2.VideoCapture, max_frames: int | None,
                 progress: tqdm | None) -> Iterator[np.ndarray]:
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
        n += 1
        if progress is not None:
            progress.update(1)
        if max_frames is not None and n >= max_frames:
            break


def cmd_extract(args: argparse.Namespace) -> int:
    video = Path(args.video)
    out_path = Path(args.output) if args.output else video.with_suffix(".metrics.json")
    try:
        cap, source = _open_video(video)
    except FileNotFoundError as e:
        print(f"error: video not found: {e}", file=sys.stderr)
        return EXIT_IO
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_DECODE

    try:
        engine = MediaPipeEngine(
            model_complexity=args.model_complexity,
            min_detection_confidence=args.min_confidence,
        )
    except Exception as e:
        print(f"error: engine init failed: {e}", file=sys.stderr)
        cap.release()
        return EXIT_ENGINE

    if not args.quiet:
        print(f"[INFO] {source.width}x{source.height}, {source.fps:.2f} fps, "
              f"{source.total_frames} frames")

    analyzer = FrameAnalyzer(engine=engine, stance=args.stance, source=source)
    progress = None if args.quiet else tqdm(total=source.total_frames, unit="frame")
    try:
        session = analyzer.run(_iter_frames(cap, args.max_frames, progress))
    finally:
        if progress is not None:
            progress.close()
        cap.release()
        engine.close()

    out_path.write_text(session.model_dump_json(indent=2))
    if not args.quiet:
        print(f"[INFO] Detection rate: {session.summary.detection_rate:.1%}")
        print(f"[INFO] Wrote {out_path}")
    return EXIT_OK


def cmd_render(args: argparse.Namespace) -> int:
    from surfanalysis.rendering.overlay import OverlayRenderer
    from surfanalysis.rendering.writer import VideoSink

    video = Path(args.video)
    json_path = Path(args.metrics_json)
    out_path = (
        Path(args.output) if args.output
        else video.with_name(f"{video.stem}_annotated.mp4")
    )

    if not video.exists():
        print(f"error: video not found: {video}", file=sys.stderr)
        return EXIT_IO
    if not json_path.exists():
        print(f"error: metrics json not found: {json_path}", file=sys.stderr)
        return EXIT_IO

    try:
        session = SessionRecord.model_validate_json(json_path.read_text())
    except Exception as e:
        print(f"error: invalid metrics json: {e}", file=sys.stderr)
        return EXIT_SCHEMA

    if session.schema_version != "1.0":
        print(f"error: unsupported schema_version {session.schema_version}",
              file=sys.stderr)
        return EXIT_SCHEMA

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"error: cannot open video {video}", file=sys.stderr)
        return EXIT_DECODE

    sink = VideoSink(out_path, width=session.source.width,
                     height=session.source.height, fps=session.source.fps,
                     codec=args.codec)
    renderer = OverlayRenderer(
        skeleton_color=args.skeleton_color,
        com_color=args.com_color,
        font_scale=args.font_scale,
        show_secondary=args.show_secondary,
    )

    progress = None if args.quiet else tqdm(total=len(session.frames), unit="frame")
    try:
        for record in session.frames:
            ok, frame = cap.read()
            if not ok:
                break
            sink.write(renderer.draw(frame, record))
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()
        cap.release()
        sink.close()

    if not args.quiet:
        print(f"[INFO] Wrote {out_path}")
    return EXIT_OK


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="surf",
                                description="Surfing biomechanical analysis CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="Run pose extraction on a video")
    e.add_argument("video", type=str)
    e.add_argument("-o", "--output", type=str, default=None)
    e.add_argument("--engine", choices=["mediapipe"], default="mediapipe")
    e.add_argument("--stance", choices=["regular", "goofy"], default="regular")
    e.add_argument("--model-complexity", type=int, choices=[0, 1, 2], default=1)
    e.add_argument("--min-confidence", type=float, default=0.5)
    e.add_argument("--max-frames", type=int, default=None)
    e.add_argument("--quiet", action="store_true")
    e.set_defaults(func=cmd_extract)

    r = sub.add_parser("render", help="Render annotated video from metrics JSON")
    r.add_argument("video", type=str)
    r.add_argument("metrics_json", type=str)
    r.add_argument("-o", "--output", type=str, default=None)
    r.add_argument("--show-secondary", action="store_true")
    r.add_argument("--codec", choices=["mp4v", "avc1"], default="mp4v")
    r.add_argument("--font-scale", type=float, default=0.6)
    r.add_argument("--skeleton-color", type=str, default="#00FF00")
    r.add_argument("--com-color", type=str, default="#FFFF00")
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(func=cmd_render)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
