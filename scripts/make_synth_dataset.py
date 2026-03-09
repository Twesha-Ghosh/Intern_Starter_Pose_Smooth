#!/usr/bin/env python3
"""Generate a synthetic pose JSONL dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from pose_smooth.synth import generate_synthetic_frames
from pose_smooth.io import write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic pose dataset")

    parser.add_argument("--out", required=True, help="Output JSONL file path")
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--num-keypoints", type=int, default=17)
    parser.add_argument("--jitter", type=float, default=8.0)
    parser.add_argument("--dropout-prob", type=float, default=0.08)
    parser.add_argument("--teleport-prob", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=7)

    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = generate_synthetic_frames(
        frames=args.frames,
        num_keypoints=args.num_keypoints,
        jitter_std=args.jitter,
        dropout_prob=args.dropout_prob,
        teleport_prob=args.teleport_prob,
        seed=args.seed,
    )

    write_jsonl(out_path, frames)

    print(f"Generated {len(frames)} frames")
    print(f"Saved to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())