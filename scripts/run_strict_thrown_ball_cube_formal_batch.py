#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path


PROJECT = Path("/autodl-fs/data/mingyu/Mujoco/projects/franka_revo2_thrown_ball_cube_catch")
EPISODE_SCRIPT = PROJECT / "scripts" / "run_thrown_ball_cube_episode.py"
PYTHON = Path("/root/autodl-tmp/conda-envs/robotwin2/bin/python")
SEARCH_BASE = Path("/root/autodl-tmp/mingyu/video2sim/franka_revo2_thrown_ball_cube_catch")


BASE_ENV = {
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
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


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_validation(ep_task_dir: Path) -> dict:
    path = ep_task_dir / "state_replay_validation.json"
    return json.loads(path.read_text(encoding="utf-8"))


def catch_ok(validation: dict) -> bool:
    task = validation.get("task_constraint_check", {})
    metrics = task.get("catch_metrics", {})
    return bool(
        validation.get("validation_pass")
        and task.get("true_grasp_success")
        and task.get("strict_catch_success")
        and metrics.get("visible_hand_closure")
        and metrics.get("late_thumb_contact")
    )


def fmt_vec(vals: list[float]) -> str:
    return ",".join(f"{v:.4f}" for v in vals)


def parse_vec(text: str) -> list[float]:
    return [float(x) for x in str(text).split(",")]


def jitter_scalar(rng: random.Random, text: str, scale: float, lo: float, hi: float) -> str:
    val = float(text)
    span = (hi - lo) * scale
    return f"{max(lo, min(hi, val + rng.uniform(-span, span))):.4f}"


def jitter_vec(rng: random.Random, text: str, scale: float, spans: list[float]) -> str:
    vals = parse_vec(text)
    out = [v + rng.uniform(-s * scale, s * scale) for v, s in zip(vals, spans)]
    return fmt_vec(out)


def load_controller_presets() -> dict[str, list[dict]]:
    specs = [
        ("sphere", "controller_opt_lift_*", 203311),
        ("cube", "controller_opt_cube_lift_*", 204001),
    ]
    presets: dict[str, list[dict]] = {"sphere": [], "cube": []}
    for obj_type, pattern, base_seed in specs:
        roots = sorted(
            [p for p in SEARCH_BASE.glob(pattern) if (p / "search_summary.json").exists()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for root in roots[:4]:
            try:
                summary = json.loads((root / "search_summary.json").read_text(encoding="utf-8"))
            except Exception:
                continue
            for rec in summary.get("best", []):
                if not rec.get("validation_pass"):
                    continue
                env = dict(BASE_ENV)
                env.update(rec.get("env", {}))
                env["OBJ_TYPE"] = obj_type
                presets[obj_type].append({
                    "source": str(root),
                    "case": rec.get("case"),
                    "score": rec.get("score"),
                    "seed_hint": int(base_seed + int(rec.get("case", 0)) * 31),
                    "env": env,
                })
    return presets


def sample_env(rng: random.Random, obj_type: str, preset: dict | None = None, jitter_scale: float = 1.0) -> dict:
    if preset is not None:
        env = dict(BASE_ENV)
        env.update(preset["env"])
        env["OBJ_TYPE"] = obj_type
        if jitter_scale > 0:
            env["TARGET_OFFSET"] = jitter_vec(rng, env["TARGET_OFFSET"], jitter_scale, [0.004, 0.010, 0.010])
            env["PREDICTIVE_GAIN"] = jitter_scalar(rng, env["PREDICTIVE_GAIN"], jitter_scale, 0.84, 1.00)
            env["PREDICTIVE_LOOKAHEAD_S"] = jitter_scalar(rng, env["PREDICTIVE_LOOKAHEAD_S"], jitter_scale, 0.055, 0.120)
            env["PREDICTIVE_VEL_LEAD"] = jitter_scalar(rng, env["PREDICTIVE_VEL_LEAD"], jitter_scale, 0.000, 0.030)
            env["CLOSE_LEAD"] = jitter_scalar(rng, env["CLOSE_LEAD"], jitter_scale, 0.006, 0.070)
            env["PRECLOSE_LEAD"] = jitter_scalar(rng, env["PRECLOSE_LEAD"], jitter_scale, 0.090, 0.235)
            env["CLOSE_DUR"] = jitter_scalar(rng, env["CLOSE_DUR"], jitter_scale, 0.135, 0.280)
            if "FOLLOW_GAIN" in env:
                env["FOLLOW_GAIN"] = jitter_scalar(rng, env["FOLLOW_GAIN"], jitter_scale, 0.16, 0.60)
            if "FOLLOW_VEL_GAIN" in env:
                env["FOLLOW_VEL_GAIN"] = jitter_scalar(rng, env["FOLLOW_VEL_GAIN"], jitter_scale, 0.0001, 0.0026)
            if "CRADLE_GAIN" in env:
                env["CRADLE_GAIN"] = jitter_scalar(rng, env["CRADLE_GAIN"], jitter_scale, 0.14, 0.70)
            if "CRADLE_VEL_GAIN" in env:
                env["CRADLE_VEL_GAIN"] = jitter_scalar(rng, env["CRADLE_VEL_GAIN"], jitter_scale, 0.0001, 0.0028)
            if "CUSHION_OFFSET" in env:
                env["CUSHION_OFFSET"] = jitter_vec(rng, env["CUSHION_OFFSET"], jitter_scale, [0.002, 0.006, 0.006])
            if "SETTLE_OFFSET" in env:
                env["SETTLE_OFFSET"] = jitter_vec(rng, env["SETTLE_OFFSET"], jitter_scale, [0.003, 0.006, 0.012])
        return env

    env = dict(BASE_ENV)
    env["OBJ_TYPE"] = obj_type
    if obj_type == "cube":
        env.update({
            "CUBE_SIDE_MIN": "0.052",
            "CUBE_SIDE_MAX": "0.064",
            "OBJ_MASS_MIN": "0.0028",
            "OBJ_MASS_MAX": "0.0052",
        })
    env.update({
        "TARGET_OFFSET": fmt_vec([
            rng.uniform(-0.012, 0.012),
            rng.uniform(-0.058, -0.014),
            rng.uniform(-0.088, -0.050),
        ]),
        "PREDICTIVE_GAIN": f"{rng.uniform(0.86, 0.98):.3f}",
        "PREDICTIVE_LOOKAHEAD_S": f"{rng.uniform(0.060, 0.115):.3f}",
        "PREDICTIVE_VEL_LEAD": f"{rng.uniform(0.000, 0.026):.3f}",
        "CLOSE_LEAD": f"{rng.uniform(0.010, 0.060):.3f}",
        "PRECLOSE_LEAD": f"{rng.uniform(0.10, 0.22):.3f}",
        "CLOSE_DUR": f"{rng.uniform(0.15, 0.26):.3f}",
        "FOLLOW_GAIN": f"{rng.uniform(0.20, 0.55):.3f}",
        "FOLLOW_VEL_GAIN": f"{rng.uniform(0.0002, 0.0022):.4f}",
        "CRADLE_GAIN": f"{rng.uniform(0.18, 0.64):.3f}",
        "CRADLE_VEL_GAIN": f"{rng.uniform(0.0002, 0.0024):.4f}",
        "CUSHION_OFFSET": fmt_vec([
            rng.uniform(-0.004, 0.004),
            rng.uniform(-0.028, -0.006),
            rng.uniform(0.006, 0.030),
        ]),
        "SETTLE_OFFSET": fmt_vec([
            rng.uniform(-0.006, 0.006),
            rng.uniform(-0.030, -0.004),
            rng.uniform(0.020, 0.070),
        ]),
        "CRADLE_CLIP_LOW": fmt_vec([-0.085, -0.115, -0.030]),
        "CRADLE_CLIP_HIGH": fmt_vec([0.085, 0.055, 0.095]),
        "CRADLE_LEAD_LOW": fmt_vec([-0.045, -0.075, -0.025]),
        "CRADLE_LEAD_HIGH": fmt_vec([0.045, 0.040, 0.085]),
    })
    return env


def run_episode(out_root: Path, seed: int, env_vars: dict, args: argparse.Namespace, quick: bool) -> tuple[int, str]:
    env = os.environ.copy()
    env.update(env_vars)
    if quick:
        env["QUICK_VALIDATE"] = "1"
        width, height, fps = args.quick_width, args.quick_height, args.quick_fps
    else:
        env.pop("QUICK_VALIDATE", None)
        width, height, fps = args.width, args.height, args.fps
    cmd = [
        str(PYTHON), str(EPISODE_SCRIPT),
        "--out-root", str(out_root),
        "--episodes", "1",
        "--seed", str(seed),
        "--seconds", str(args.seconds),
        "--fps", str(fps),
        "--width", str(width),
        "--height", str(height),
        "--max-penetration-m", str(args.max_penetration_m),
    ]
    proc = subprocess.run(cmd, env=env, cwd=str(PROJECT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.returncode, proc.stdout[-20000:]


def archive_batch(src_root: Path, archive_root: Path, start: int, end: int) -> Path:
    archive_root.mkdir(parents=True, exist_ok=True)
    archive = archive_root / f"batch_{start:06d}_{end:06d}.tar.zst"
    tmp_tar = archive_root / f"batch_{start:06d}_{end:06d}.tmp.tar"
    if archive.exists():
        return archive
    with tarfile.open(tmp_tar, "w") as tf:
        for idx in range(start, end + 1):
            ep_dir = src_root / f"episode_{idx:06d}"
            if ep_dir.exists():
                tf.add(ep_dir, arcname=ep_dir.name)
    zstd = shutil.which("zstd")
    if zstd:
        subprocess.run([zstd, "-T0", "-12", "-f", str(tmp_tar), "-o", str(archive)], check=True)
        tmp_tar.unlink(missing_ok=True)
    else:
        archive = archive.with_suffix(archive.suffix + ".gz")
        subprocess.run(["gzip", "-9", "-c", str(tmp_tar)], check=True, stdout=archive.open("wb"))
        tmp_tar.unlink(missing_ok=True)
    return archive


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--seed", type=int, default=240910319)
    ap.add_argument("--tmp-root", default="/root/autodl-tmp/mingyu/video2sim/franka_revo2_thrown_ball_cube_catch/formal_strict_tmp")
    ap.add_argument("--archive-root", default="/autodl-fs/data/mingyu/video2sim/franka_revo2_thrown_ball_cube_catch/formal_strict_archives")
    ap.add_argument("--keep-batch-tmp", action="store_true")
    ap.add_argument("--max-attempts", type=int, default=80)
    ap.add_argument("--seconds", type=float, default=1.55)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--quick-width", type=int, default=320)
    ap.add_argument("--quick-height", type=int, default=192)
    ap.add_argument("--quick-fps", type=int, default=72)
    ap.add_argument("--max-penetration-m", type=float, default=0.04)
    ap.add_argument("--start-index", type=int, default=0)
    args = ap.parse_args()

    tmp_root = Path(args.tmp_root)
    archive_root = Path(args.archive_root)
    work_root = tmp_root / "_work"
    progress_path = archive_root / "progress.json"
    tmp_root.mkdir(parents=True, exist_ok=True)
    archive_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    presets = load_controller_presets()
    write_json(archive_root / "controller_presets_loaded.json", {
        "sphere": [{k: v for k, v in p.items() if k != "env"} for p in presets.get("sphere", [])],
        "cube": [{k: v for k, v in p.items() if k != "env"} for p in presets.get("cube", [])],
    })
    completed = 0
    last_archive = None
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        completed = int(progress.get("completed", 0))
        last_archive = progress.get("latest_archive")

    start = max(args.start_index, completed)
    for batch_start in range(start, args.total, args.batch_size):
        batch_end = min(args.total - 1, batch_start + args.batch_size - 1)
        for idx in range(batch_start, batch_end + 1):
            final_ep = tmp_root / f"episode_{idx:06d}"
            if final_ep.exists() and (final_ep / "thrown_ball_cube_catch" / "state_replay_validation.json").exists():
                validation = read_validation(final_ep / "thrown_ball_cube_catch")
                if catch_ok(validation):
                    completed = max(completed, idx + 1)
                    continue
                shutil.rmtree(final_ep, ignore_errors=True)

            accepted = None
            for attempt in range(args.max_attempts):
                obj_type = "sphere" if (idx + attempt) % 2 == 0 else "cube"
                obj_presets = presets.get(obj_type, [])
                preset = obj_presets[(idx + attempt) % len(obj_presets)] if obj_presets else None
                if preset is not None:
                    if attempt < 24:
                        jitter_scale = 0.18 if attempt < 12 else 0.08
                        seed = int(preset["seed_hint"]) + idx * 17 + attempt
                    else:
                        jitter_scale = 0.0
                        seed = int(preset["seed_hint"])
                    env_vars = sample_env(rng, obj_type, preset=preset, jitter_scale=jitter_scale)
                else:
                    env_vars = sample_env(rng, obj_type)
                    seed = args.seed + idx * 1009 + attempt * 31
                quick_root = work_root / f"episode_{idx:06d}_attempt_{attempt:03d}_quick"
                shutil.rmtree(quick_root, ignore_errors=True)
                code, log = run_episode(quick_root, seed, env_vars, args, quick=True)
                (quick_root / "run.log").write_text(log, encoding="utf-8")
                if code != 0:
                    shutil.rmtree(quick_root, ignore_errors=True)
                    continue
                quick_ep = quick_root / "episode_000000" / "thrown_ball_cube_catch"
                try:
                    quick_validation = read_validation(quick_ep)
                except Exception:
                    shutil.rmtree(quick_root, ignore_errors=True)
                    continue
                if not catch_ok(quick_validation):
                    shutil.rmtree(quick_root, ignore_errors=True)
                    continue
                accepted = {
                    "seed": seed,
                    "attempt": attempt,
                    "obj_type": obj_type,
                    "env": env_vars,
                    "preset": {k: v for k, v in preset.items() if k != "env"} if preset else None,
                }
                shutil.rmtree(quick_root, ignore_errors=True)
                break
            if accepted is None:
                raise RuntimeError(f"episode_{idx:06d} failed strict search after {args.max_attempts} attempts")

            full_root = work_root / f"episode_{idx:06d}_full"
            shutil.rmtree(full_root, ignore_errors=True)
            code, log = run_episode(full_root, accepted["seed"], accepted["env"], args, quick=False)
            (full_root / "run.log").write_text(log, encoding="utf-8")
            if code != 0:
                raise RuntimeError(f"episode_{idx:06d} full export failed")
            full_ep_task = full_root / "episode_000000" / "thrown_ball_cube_catch"
            validation = read_validation(full_ep_task)
            if not catch_ok(validation):
                shutil.rmtree(full_root, ignore_errors=True)
                raise RuntimeError(f"episode_{idx:06d} passed quick but failed full strict validation")
            write_json(full_ep_task / "strict_sampling_record.json", accepted)
            if final_ep.exists():
                shutil.rmtree(final_ep)
            (full_root / "episode_000000").rename(final_ep)
            shutil.rmtree(full_root, ignore_errors=True)
            completed = max(completed, idx + 1)
            write_json(progress_path, {
                "task": "franka_revo2_thrown_ball_cube_catch",
                "mode": "strict_true_grasp_batch",
                "completed": completed,
                "total": args.total,
                "current_episode": idx,
                "latest_archive": last_archive,
                "tmp_root": str(tmp_root),
                "archive_root": str(archive_root),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            print(json.dumps({"accepted": idx, "completed": completed, "total": args.total, **accepted}, ensure_ascii=False), flush=True)

        archive = archive_batch(tmp_root, archive_root, batch_start, batch_end)
        last_archive = str(archive)
        if not args.keep_batch_tmp:
            for idx in range(batch_start, batch_end + 1):
                shutil.rmtree(tmp_root / f"episode_{idx:06d}", ignore_errors=True)
        write_json(progress_path, {
            "task": "franka_revo2_thrown_ball_cube_catch",
            "mode": "strict_true_grasp_batch",
            "completed": completed,
            "total": args.total,
            "latest_archive": last_archive,
            "tmp_root": str(tmp_root),
            "archive_root": str(archive_root),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        print(json.dumps({"archived": last_archive, "completed": completed, "total": args.total}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
