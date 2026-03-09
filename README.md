# Intern Assignment Starter (No Solution Included)

This folder is a starter template for the temporal pose smoothing assignment. Core implementation is intentionally omitted.

## Goal
Implement a small installable Python package that smooths per-frame 2D pose keypoints over time, handles missing/low-confidence joints, and provides CLI + tests.

## Required CLI
- `pose-smooth smooth --in input.jsonl --out output.jsonl --alpha 0.6 --min-score 0.3 --max-jump-px 35 --score-decay 0.95`
- `pose-smooth metrics --in input.jsonl [--smoothed output.jsonl]`
- `pose-smooth demo --out-dir demo_out/ --frames 240 --jitter 8 --dropout-prob 0.08`
- `python -m pose_smooth ...` should also work.

## Input schema
Each JSONL line:
```json
{
  "frame_idx": 0,
  "timestamp_s": 0.0,
  "keypoints": [[x, y, score], ...]
}
```

## Where You Should Code
Open this file first:
- `FILES_TO_EDIT.md`

Implement TODOs in package code and script files listed there.

Important:
- Tests are already written for you.
- Do not edit tests; run them to validate your code.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Validate your work
```bash
pytest -q
pose-smooth demo --out-dir demo_out
```

## Success criteria
- `pip install -e .` works
- `pose-smooth demo --out-dir demo_out` runs end-to-end
- `demo_out/input.jsonl` and `demo_out/output.jsonl` exist
- metrics show `jitter_after < jitter_before`
- `pytest -q` passes

## Required submission notes 
Include these in your submission:
1. Three real bugs you hit and how you fixed them.
  - Bug 1:
    - While implementing the CLI demo command, I passed the argument jitter to generate_synthetic_frames(). However, the function expected the parameter name jitter_std. This caused a runtime error. The fix was to map the CLI argument --jitter to the correct function parameter jitter_std before calling the generator.
  - Bug 2:
    - In an early version of the smoother, joints with scores below min_score were still passed into the EMA update. This caused noisy detections to move the smoothed position even when the joint should have been treated as missing. The fix was to add a condition that treats low-confidence joints as missing. This would hold the previous position and ignore the score instead of updating the EMA.
  - Bug 3:
    - Initially the teleport check was performed after the EMA update step. This allowed large coordinate jumps to partially influence the smoothed state before being rejected, which still produced noticeable spikes in the trajectory. The fix was to compute the displacement first and reject large jumps before performing the EMA update.
2. Three failure cases you observed or expect (occlusion, low confidence bursts, jitter spikes, frame drops, etc.).
  - Error 1
    - Occlusion of joints
      - When a joint becomes occluded, detectors often produce very low confidence values or missing coordinates. The smoother holds the previous position and decays the confidence score, but long occlusions may cause the estimated position to become stale until the detector recovers.
    - Large jitter spikes from detector noise
      - Occasionally a detector outputs a noisy coordinate far from the true joint position. While the teleport gating logic rejects large jumps, medium-sized noise spikes below the threshold may still influence the EMA and slightly shift the smoothed trajectory.
    - Bursts of low confidence detections
      - Some detectors output several consecutive frames of low confidence for the same joint. During these bursts the smoother treats the joint as missing and decays the score, which can delay the system’s ability to recover once reliable detections resume.
3. One concrete next improvement (for example: One Euro filter or Kalman filter).
  - A potential improvement would be replacing the exponential moving average with a One Euro filter or Kalman filter. These filters adapt smoothing dynamically based on motion velocity and measurement uncertainty, allowing better jitter reduction while still responding quickly to real movement.
  - Another potential improvement would be adding a temporal median filter before the EMA smoothing stage. Median filters are effective at removing sudden detector spikes or outliers that occur when pose estimators briefly misidentify a joint position. By filtering out these outliers before applying EMA smoothing, the system could achieve more stable trajectories without increasing lag.



Citation:
  - ChatGPT - https://chatgpt.com/
    - Used for fixing write-up with proper grammar and sentence structure
    - Used for organizing and structuring code 
    - Used for scope clarification 
  - Codex 
    - Used for help running pytest -q

