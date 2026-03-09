"""Argparse CLI for pose smoothing and evaluation."""

from __future__ import annotations

import argparse
from typing import Optional

from pose_smooth.config import SmoothConfig
from pose_smooth.io import read_jsonl, write_jsonl
from pose_smooth.metrics import compute_metrics
from pose_smooth import smooth_frames
from pose_smooth.synth import generate_synthetic_frames


def cmd_smooth(args) -> int:
    config = SmoothConfig(
        alpha=args.alpha,
        min_score=args.min_score,
        max_jump_px=args.max_jump_px,
        score_decay=args.score_decay,
    )

    frames = read_jsonl(args.in_file)
    smoothed = smooth_frames(frames, config)

    write_jsonl(args.out, smoothed)

    print(f"Smoothed {len(frames)} frames")
    print(f"Output written to {args.out}")

    return 0


def cmd_metrics(args) -> int:
    raw = read_jsonl(args.in_file)

    smoothed = None
    if args.smoothed:
        smoothed = read_jsonl(args.smoothed)

    metrics = compute_metrics(raw, smoothed, args.min_score)

    print("Metrics")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    if metrics["jitter_after"] is not None:
        print(
            f"jitter_after < jitter_before: "
            f'{metrics["jitter_after"] < metrics["jitter_before"]}'
        )

    return 0


def cmd_demo(args) -> int:
    import os

    os.makedirs(args.out_dir, exist_ok=True)

    input_path = f"{args.out_dir}/input.jsonl"
    output_path = f"{args.out_dir}/output.jsonl"

    data = generate_synthetic_frames(
        frames=args.frames,
        jitter_std=args.jitter,
        dropout_prob=args.dropout_prob,
    )

    write_jsonl(input_path, data)

    config = SmoothConfig(
        alpha=args.alpha,
        min_score=args.min_score,
        max_jump_px=args.max_jump_px,
        score_decay=args.score_decay,
    )

    smoothed = smooth_frames(data, config)

    write_jsonl(output_path, smoothed)

    print("Demo generated")
    print("Input:", input_path)
    print("Output:", output_path)

    metrics = compute_metrics(data, smoothed, args.min_score)

    print("\nMetrics")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    if metrics["jitter_after"] is not None:
        print(
            f"jitter_after < jitter_before: "
            f'{metrics["jitter_after"] < metrics["jitter_before"]}'
        )

    return 0


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(prog="pose-smooth")

    sub = parser.add_subparsers(dest="command", required=True)

    
    smooth = sub.add_parser("smooth")

    smooth.add_argument("--in", dest="in_file", required=True)
    smooth.add_argument("--out", required=True)

    smooth.add_argument("--alpha", type=float, default=0.6)
    smooth.add_argument("--min-score", type=float, default=0.3)
    smooth.add_argument("--max-jump-px", type=float, default=35)
    smooth.add_argument("--score-decay", type=float, default=0.95)

    smooth.set_defaults(func=cmd_smooth)

    
    metrics = sub.add_parser("metrics")

    metrics.add_argument("--in", dest="in_file", required=True)
    metrics.add_argument("--smoothed")
    metrics.add_argument("--min-score", type=float, default=0.3)

    metrics.set_defaults(func=cmd_metrics)

    
    demo = sub.add_parser("demo")

    demo.add_argument("--out-dir", required=True)

    demo.add_argument("--frames", type=int, default=240)
    demo.add_argument("--jitter", type=float, default=8)
    demo.add_argument("--dropout-prob", type=float, default=0.08)

    demo.add_argument("--alpha", type=float, default=0.6)
    demo.add_argument("--min-score", type=float, default=0.3)
    demo.add_argument("--max-jump-px", type=float, default=35)
    demo.add_argument("--score-decay", type=float, default=0.95)

    demo.set_defaults(func=cmd_demo)

    return parser


def main(argv: Optional[list[str]] = None) -> int:

    parser = build_parser()

    args = parser.parse_args(argv)

    result = args.func(args)

    return int(result) if result is not None else 0
