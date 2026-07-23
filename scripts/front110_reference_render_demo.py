#!/usr/bin/env python3
"""Render a few clean single-drop front-110 demo clips with the frozen controller."""
import argparse
import importlib.util
import pathlib
import types

PROJECT = pathlib.Path(__file__).resolve().parents[1]
CTRL = PROJECT / "frozen" / "stage4e_front110_v699_front110_random48_passed.py"


def load_controller():
    spec = importlib.util.spec_from_file_location("stage4e_front110_v699", CTRL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_args(mod, angle, seed, out_dir):
    d = dict(seed=seed, angle=angle, yaw=0.0, seconds=1.1, fps=30, no_video=False, no_write=False,
             quiet=True, release_time=0.35, camera="front_arc", out_dir=str(out_dir),
             radial_offset=-0.205, tangent_offset=-0.095, z_offset=0.245, catch_z=0.875,
             wrist_yaw_factor=1.0, wrist_yaw=0.0, wrist_roll_factor=0.0, wrist_roll=0.0,
             wrist_pitch_factor=0.0, wrist_pitch=0.0, ready_tangent=0.120, move_start=0.120,
             move_dur=0.130, insert_tangent=0.040, latch_drop_rate=0.850,
             latch_drop_duration=0.180, latch_min_z=0.580, latch_lift_rate=0.0,
             latch_lift_start=0.160, latch_lift_duration=0.180, latch_max_z=0.920,
             latch_radial_offset=0.0, latch_tangent_offset=-0.010, latch_xy_follow=0.0,
             latch_xy_follow_start=0.0, latch_xy_follow_dur=0.080, middle_latch_scale=0.0,
             middle_latch_start=0.030, middle_latch_dur=0.120, index_latch_scale=0.0,
             index_latch_start=0.030, index_latch_dur=0.120, index_latch_prox=1.05,
             index_latch_dist=1.05, latch_brake_start=0.0, latch_brake_duration=0.160,
             latch_brake_yaw=0.0, latch_brake_roll=0.0, latch_brake_pitch=0.0,
             thumb_lead=0.160, thumb_dur=0.240, finger_lead=0.110, finger_dur=0.180,
             thumb_close_scale=1.0, index_close_scale=1.0, middle_close_scale=0.0,
             front110_auto=True)
    args = types.SimpleNamespace(**d)
    mod.apply_front110_auto_args(args)
    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(PROJECT / "outputs" / "front110_demo_clips"))
    ap.add_argument("--angles", default="38.368,57.041,91.193,115.762,143.082")
    ns = ap.parse_args()
    mod = load_controller()
    out = pathlib.Path(ns.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, raw in enumerate(ns.angles.split(",")):
        angle = float(raw.strip())
        res = mod.run_episode(make_args(mod, angle, 97000 + i, out))
        print(angle, res["success"], res["video"])


if __name__ == "__main__":
    main()
