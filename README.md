# Franka FR3 + Revo2 Thrown Object Catch Dataset Generator

MuJoCo task for generating physically validated videos and state data of a Franka FR3 arm with a Revo2-style dexterous hand catching a front-thrown object on a parabolic trajectory.

The current release contains the task code, full robot/hand visual assets, export-schema utilities, a compact sample episode, and browser-safe preview videos.

## What It Exports

Each generated episode writes:

- raw videos from fixed cameras: front, top, left, right, left_oblique, right_oblique
- 2D-box videos, 3D-box videos, and mask videos for the same views
- `dataset.npz` with frame-level state arrays
- `episode_state.json` with object/robot state, physical quantities, trajectories, boxes, masks, camera metadata, mesh pose fields, text descriptions, and coordinate-frame metadata
- `state_replay_validation.json` with replay/physics checks such as replayability, trajectory error, contact error, penetration, and task-constraint violations
- per-episode mesh assets for the incoming object, including visual/collision mesh plus VAE-friendly voxel, point-cloud, and implicit-field matrices

Auxiliary contact pads used for stable contact are collision-only and hidden from rendered videos.

## Repository Layout

- `scripts/run_thrown_ball_cube_episode.py` - single/multi-episode generator
- `scripts/run_strict_thrown_ball_cube_formal_batch.py` - strict retrying batch generator
- `tools/video2sim_schema.py` - shared export schema helpers
- `tools/mesh_vae_encodings.py` - voxel, point-cloud, implicit-field mesh encodings
- `tools/schema_smoke_check.py` - validates required output files/fields
- `assets/full_robot_urdf_mirror/` - Franka/Revo2 visual mesh assets
- `examples/sample_episode/` - compact generated sample with data files and a few videos
- `examples/browser_safe_preview/preview.html` - small preview page with H.264 browser-safe videos

## Install

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install system FFmpeg if it is not already available:

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

For headless Linux rendering, use EGL:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

On a local machine with a working display, `MUJOCO_GL=glfw` can also work.

## Quick Run

```bash
bash scripts/run_smoke_test.sh outputs/smoke_test
```

This quick run verifies that MuJoCo rendering, schema export, videos, masks, boxes, meshes, and VAE encodings are produced. For only physically accepted catches, use the strict batch runner below.

Expected output directory:

```text
outputs/smoke_test/episode_000000/thrown_ball_cube_catch/
```

Open the generated videos under `videos/raw`, `videos/bbox2d`, `videos/bbox3d`, and `videos/mask`.

## Strict Batch Generation

```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export VIDEO2SIM_SOFT_CATCH_CAGE=0 EXTRA_SOFT_FINGER_PADS=1
export POST_GRASP_ROBOT_DAMPING_MUL=2.8 POST_GRASP_ROBOT_FRICTIONLOSS_ADD=0.010
python scripts/run_strict_thrown_ball_cube_formal_batch.py \
  --total 10 \
  --batch-size 5 \
  --tmp-root outputs/strict_batch \
  --archive-root archives/strict_batch \
  --keep-batch-tmp \
  --width 960 --height 540 --fps 60 \
  --seconds 1.55 \
  --max-penetration-m 0.04 \
  --max-attempts 120 \
  --seed 63139
```

The strict batch runner rejects samples that fail physical validation instead of silently saving bad episodes.

## Preview Sample Videos

Open this file in a browser after cloning:

```text
examples/browser_safe_preview/preview.html
```

The preview videos are re-encoded as H.264 baseline with faststart for broad browser compatibility.

## Notes

- Current stable demonstrated samples are sphere catches. Cube catching support is scaffolded but should be tuned further before large-scale formal generation.
- Large production outputs should be written outside the git working tree, then archived separately.
- Generated data is intentionally verbose for video-to-simulation and robot-learning use cases.
