#!/usr/bin/env python3
"""Run deterministic front-110 continuous random-angle validation for the v699 MuJoCo controller."""
import argparse
import csv
import importlib.util
import json
import pathlib
import random
import time
import types

PROJECT = pathlib.Path(__file__).resolve().parents[1]
CTRL = PROJECT / "frozen" / "stage4e_front110_v699_front110_random48_passed.py"


def load_controller():
    spec = importlib.util.spec_from_file_location("stage4e_front110_v699", CTRL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_args(mod, angle, seed, out_dir, seconds=1.1, video=False, camera="front_arc"):
    d = dict(
        seed=seed, angle=angle, yaw=0.0, seconds=seconds, fps=30,
        no_video=not video, no_write=not video, quiet=True,
        release_time=0.35, camera=camera, out_dir=str(out_dir),
        radial_offset=-0.205, tangent_offset=-0.095, z_offset=0.245, catch_z=0.875,
        wrist_yaw_factor=1.0, wrist_yaw=0.0, wrist_roll_factor=0.0, wrist_roll=0.0,
        wrist_pitch_factor=0.0, wrist_pitch=0.0,
        ready_tangent=0.120, move_start=0.120, move_dur=0.130, insert_tangent=0.040,
        latch_drop_rate=0.850, latch_drop_duration=0.180, latch_min_z=0.580,
        latch_lift_rate=0.0, latch_lift_start=0.160, latch_lift_duration=0.180, latch_max_z=0.920,
        latch_radial_offset=0.0, latch_tangent_offset=-0.010,
        latch_xy_follow=0.0, latch_xy_follow_start=0.0, latch_xy_follow_dur=0.080,
        middle_latch_scale=0.0, middle_latch_start=0.030, middle_latch_dur=0.120,
        index_latch_scale=0.0, index_latch_start=0.030, index_latch_dur=0.120,
        index_latch_prox=1.05, index_latch_dist=1.05,
        latch_brake_start=0.0, latch_brake_duration=0.160,
        latch_brake_yaw=0.0, latch_brake_roll=0.0, latch_brake_pitch=0.0,
        thumb_lead=0.160, thumb_dur=0.240, finger_lead=0.110, finger_dur=0.180,
        thumb_close_scale=1.0, index_close_scale=1.0, middle_close_scale=0.0,
        front110_auto=True,
    )
    args = types.SimpleNamespace(**d)
    mod.apply_front110_auto_args(args)
    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=90909)
    ap.add_argument("--count", type=int, default=48)
    ap.add_argument("--angle-min", type=float, default=35.0)
    ap.add_argument("--angle-max", type=float, default=145.0)
    ap.add_argument("--out-dir", default=str(PROJECT / "outputs" / "front110_random48_repro"))
    ap.add_argument("--seconds", type=float, default=1.1)
    ap.add_argument("--video", action="store_true", help="Render per-angle MP4s; normally keep off for validation.")
    args = ap.parse_args()

    mod = load_controller()
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    angles = [rng.uniform(args.angle_min, args.angle_max) for _ in range(args.count)]
    rows, fails = [], []
    t0 = time.time()
    for i, angle in enumerate(angles):
        run_seed = args.seed + 5000 + i
        res = mod.run_episode(make_args(mod, angle, run_seed, out, seconds=args.seconds, video=args.video))
        row = dict(
            i=i, angle=round(angle, 3), seed=run_seed, success=bool(res["success"]),
            strict=int(res["best_strict_handle_contact_frames"]), final_vz=float(res["final_vz"]),
            min_z=float(res["min_tool_z"]), bad=bool(res["bad_functional_contact"]),
            tangent=float(res["tangent_offset"]), lift=float(res["latch_lift_rate"]),
        )
        rows.append(row)
        if not row["success"]:
            fails.append(row)
        print("{:02d}/{} angle={:.3f} ok={} strict={} vz={} minz={} bad={}".format(
            i + 1, args.count, angle, row["success"], row["strict"], row["final_vz"], row["min_z"], row["bad"]
        ), flush=True)

    (out / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(passed=sum(r["success"] for r in rows), total=len(rows), fails=fails,
                  out=str(out), elapsed=round(time.time() - t0, 1))
    print("RESULT", json.dumps(result, sort_keys=True))
    raise SystemExit(0 if not fails else 2)


if __name__ == "__main__":
    main()
