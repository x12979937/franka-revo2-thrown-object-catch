#!/usr/bin/env python3
import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path


PROJECT = Path("/autodl-fs/data/mingyu/Mujoco/projects/franka_revo2_thrown_ball_cube_catch")
EPISODE_SCRIPT = PROJECT / "scripts" / "run_thrown_ball_cube_episode.py"
PYTHON = Path("/root/autodl-tmp/conda-envs/robotwin2/bin/python")
OUT_BASE = Path("/root/autodl-tmp/mingyu/video2sim/franka_revo2_thrown_ball_cube_catch")


BASE_ENV = {
    "MUJOCO_GL": "egl",
    "QUICK_VALIDATE": "1",
    "VIDEO2SIM_SOFT_CATCH_CAGE": "0",
    "EXTRA_SOFT_FINGER_PADS": "1",
    "PREDICTIVE_IK_CATCH": "1",
    "CONTACT_TRIGGERED_CLOSE": "1",
    "FINGER_DISTAL_PAD_RADIUS": "0.026",
    "FINGER_PROXIMAL_PAD_RADIUS": "0.024",
    "FINGER_EXTRA_PAD_RADIUS": "0.018",
    "FINGER_PAD_FRICTION": "8.5 1.45 0.180",
    "PALM_PAD_FRICTION": "5.0 1.00 0.120",
    "OBJECT_FRICTION": "7.5 1.20 0.150",
    "OBJ_TYPE": "sphere",
    "SPHERE_RADIUS_MIN": "0.034",
    "SPHERE_RADIUS_MAX": "0.039",
    "OBJ_MASS_MIN": "0.0020",
    "OBJ_MASS_MAX": "0.0038",
    "START_X_MIN": "-0.004",
    "START_X_MAX": "0.004",
    "START_Y_MIN": "-0.90",
    "START_Y_MAX": "-0.84",
    "START_Z_MIN": "-0.17",
    "START_Z_MAX": "-0.12",
    "DURATION_MIN": "0.84",
    "DURATION_MAX": "0.90",
    "MIN_VISIBLE_HAND_CLOSURE_DELTA_SUM": "1.15",
    "MIN_HAND_QPOS_RANGE_SUM": "1.35",
    "CONTACT_CLOSE_DELAY": "0",
    "PRELATCH_TIP_DIST": "0.085",
    "PRELATCH_CENTER_DIST": "0.120",
    "PREGRASP": "0.28,0.20,0.14,0.05,0.04,0.05,0.04,0.02,0.02,0.02,0.02",
    "HAND_TARGET": "1.57,1.03,1.03,1.20,1.45,1.20,1.45,1.10,1.35,1.05,1.30",
    "HOLD_TARGET": "1.57,1.03,1.03,1.32,1.55,1.32,1.55,1.22,1.48,1.16,1.42",
    "CLOSE_DUR": "0.20",
    "SQUEEZE_LEAD": "0.05",
    "SQUEEZE_DUR": "0.22",
    "PRECLOSE_LEAD": "0.16",
    "PRECLOSE_DUR": "0.16",
    "ARM_VEL_SCALE": "40",
    "PREDICTIVE_GAIN": "0.92",
    "PREDICTIVE_LOOKAHEAD_S": "0.085",
    "PREDICTIVE_VEL_LEAD": "0.012",
    "CLOSE_LEAD": "0.030",
}


def fmt_vec(vals):
    return ",".join(f"{v:.4f}" for v in vals)


def sample_case(rng, idx):
    target_offset = [
        rng.uniform(-0.012, 0.012),
        rng.uniform(-0.050, -0.010),
        rng.uniform(-0.095, -0.055),
    ]
    env = dict(BASE_ENV)
    env.update({
        "TARGET_OFFSET": fmt_vec(target_offset),
        "PREDICTIVE_GAIN": f"{rng.uniform(0.86, 0.98):.3f}",
        "PREDICTIVE_LOOKAHEAD_S": f"{rng.uniform(0.060, 0.115):.3f}",
        "PREDICTIVE_VEL_LEAD": f"{rng.uniform(0.000, 0.026):.3f}",
        "CLOSE_LEAD": f"{rng.uniform(0.010, 0.060):.3f}",
        "PRECLOSE_LEAD": f"{rng.uniform(0.10, 0.22):.3f}",
        "CLOSE_DUR": f"{rng.uniform(0.15, 0.26):.3f}",
        "FOLLOW_GAIN": f"{rng.uniform(0.20, 0.55):.3f}",
        "FOLLOW_VEL_GAIN": f"{rng.uniform(0.0002, 0.0018):.4f}",
        "CRADLE_GAIN": f"{rng.uniform(0.12, 0.38):.3f}",
        "CRADLE_VEL_GAIN": f"{rng.uniform(0.0002, 0.0018):.4f}",
    })
    return env


def load_metrics(ep_dir):
    val_path = ep_dir / "thrown_ball_cube_catch" / "state_replay_validation.json"
    if not val_path.exists():
        val_path = ep_dir / "state_replay_validation.json"
    data = json.loads(val_path.read_text())
    task = data.get("task_constraint_check", {})
    metrics = task.get("catch_metrics", {})
    return data, metrics


def score(validation, metrics):
    s = 0.0
    s += 1000.0 if metrics.get("strict_catch_success") else 0.0
    s += 1000.0 if metrics.get("true_grasp_success") else 0.0
    s += 250.0 if metrics.get("visible_hand_closure") else 0.0
    s += 180.0 if metrics.get("late_thumb_contact") else 0.0
    s += 20.0 * float(metrics.get("inside_pinch_grasp_final_window_frames", 0) or 0)
    s += 15.0 * float(metrics.get("held_final_window_frames", 0) or 0)
    s += 10.0 * float(metrics.get("late_finger_contact_body_count", 0) or 0)
    s -= 500.0 * max(0.0, float(metrics.get("max_penetration_m", 1.0) or 1.0) - 0.03)
    s -= 120.0 * float(metrics.get("final_relative_speed_m_s", 1.0) or 1.0)
    s += 100.0 if validation.get("validation_pass") else 0.0
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=36)
    ap.add_argument("--seed", type=int, default=172901)
    ap.add_argument("--seconds", type=float, default=1.55)
    ap.add_argument("--fps", type=int, default=72)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=192)
    ap.add_argument("--tag", default=time.strftime("controller_opt_%Y%m%d_%H%M%S"))
    args = ap.parse_args()

    root = OUT_BASE / args.tag
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    results = []

    for i in range(args.cases):
        env = os.environ.copy()
        case_env = sample_case(rng, i)
        env.update(case_env)
        out = root / f"case_{i:04d}"
        cmd = [
            str(PYTHON), str(EPISODE_SCRIPT),
            "--out-root", str(out),
            "--episodes", "1",
            "--seed", str(args.seed + i * 31),
            "--seconds", str(args.seconds),
            "--fps", str(args.fps),
            "--width", str(args.width),
            "--height", str(args.height),
            "--max-penetration-m", "0.04",
        ]
        started = time.time()
        proc = subprocess.run(cmd, env=env, cwd=str(PROJECT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        rec = {"case": i, "out_root": str(out), "returncode": proc.returncode, "seconds": round(time.time() - started, 3), "env": case_env}
        (out / "run.log").write_text(proc.stdout[-20000:])
        try:
            validation, metrics = load_metrics(out / "episode_000000")
            rec.update({
                "validation_pass": bool(validation.get("validation_pass")),
                "score": score(validation, metrics),
                "metrics": metrics,
            })
        except Exception as exc:
            rec.update({"validation_pass": False, "score": -9999, "error": repr(exc)})
        results.append(rec)
        results.sort(key=lambda r: r.get("score", -9999), reverse=True)
        summary = {"root": str(root), "cases_done": i + 1, "cases_total": args.cases, "best": results[:8]}
        (root / "search_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(json.dumps({"case": i, "score": rec.get("score"), "pass": rec.get("validation_pass"), "best": results[0].get("score")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
