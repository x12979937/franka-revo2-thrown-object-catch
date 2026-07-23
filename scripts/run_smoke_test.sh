#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export VIDEO2SIM_SOFT_CATCH_CAGE=${VIDEO2SIM_SOFT_CATCH_CAGE:-0}
export EXTRA_SOFT_FINGER_PADS=${EXTRA_SOFT_FINGER_PADS:-1}
export POST_GRASP_ROBOT_DAMPING_MUL=${POST_GRASP_ROBOT_DAMPING_MUL:-2.8}
export POST_GRASP_ROBOT_FRICTIONLOSS_ADD=${POST_GRASP_ROBOT_FRICTIONLOSS_ADD:-0.010}
OUT=${1:-outputs/smoke_test}
python scripts/run_thrown_ball_cube_episode.py --out-root "$OUT" --episodes 1 --seed 63139 --seconds 1.55 --width 640 --height 360 --fps 30 --max-penetration-m 0.04
python tools/schema_smoke_check.py --episode "$OUT"/episode_000000/thrown_ball_cube_catch --strict-videos
