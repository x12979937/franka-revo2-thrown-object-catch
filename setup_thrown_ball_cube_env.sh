#!/usr/bin/env bash
export THROWN_BALL_CUBE_ROOT="/autodl-fs/data/mingyu/Mujoco/projects/franka_revo2_thrown_ball_cube_catch"
export THROWN_BALL_CUBE_TMP="/root/autodl-tmp/mingyu/video2sim/franka_revo2_thrown_ball_cube_catch"
export THROWN_BALL_CUBE_ARCHIVE="/autodl-fs/data/mingyu/video2sim/franka_revo2_thrown_ball_cube_catch"
export THROWN_BALL_CUBE_PYTHON="/root/autodl-tmp/conda-envs/robotwin2/bin/python"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="$THROWN_BALL_CUBE_ROOT/tools${PYTHONPATH:+:$PYTHONPATH}"
