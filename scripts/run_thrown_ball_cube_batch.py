#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def archive_batch(src_root: Path, archive_root: Path, start: int, end: int):
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
    subprocess.run(["zstd", "-T0", "-19", "-f", str(tmp_tar), "-o", str(archive)], check=True)
    tmp_tar.unlink(missing_ok=True)
    return archive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=240910319)
    parser.add_argument("--tmp-root", default="/root/autodl-tmp/mingyu/video2sim/franka_revo2_thrown_ball_cube_catch/formal_tmp")
    parser.add_argument("--archive-root", default="/autodl-fs/data/mingyu/video2sim/franka_revo2_thrown_ball_cube_catch/archives")
    parser.add_argument("--keep-batch-tmp", action="store_true")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seconds", type=float, default=1.55)
    parser.add_argument("--max-penetration-m", type=float, default=0.03)
    args = parser.parse_args()

    script = Path(__file__).with_name("run_thrown_ball_cube_episode.py")
    tmp_root = Path(args.tmp_root)
    archive_root = Path(args.archive_root)
    progress = archive_root / "progress.json"
    tmp_root.mkdir(parents=True, exist_ok=True)
    archive_root.mkdir(parents=True, exist_ok=True)
    done = set()
    if progress.exists():
        done.update(read_json(progress).get("completed_episodes", []))

    for batch_start in range(0, args.total, args.batch_size):
        batch_end = min(args.total - 1, batch_start + args.batch_size - 1)
        accepted = []
        for idx in range(batch_start, batch_end + 1):
            if idx in done:
                accepted.append(idx)
                continue
            ep_dir = tmp_root / f"episode_{idx:06d}" / "thrown_ball_cube_catch"
            for attempt in range(args.max_retries):
                if ep_dir.exists():
                    shutil.rmtree(ep_dir.parent)
                seed = args.seed + idx * 1009 + attempt * 17
                cmd = [
                    sys.executable, str(script), "--out-root", str(tmp_root), "--episodes", "1",
                    "--seed", str(seed), "--width", str(args.width), "--height", str(args.height),
                    "--fps", str(args.fps), "--seconds", str(args.seconds),
                    "--max-penetration-m", str(args.max_penetration_m),
                ]
                env = None
                subprocess.run(cmd, check=True, env=env)
                produced = tmp_root / "episode_000000"
                target = tmp_root / f"episode_{idx:06d}"
                if produced.exists() and produced != target:
                    if target.exists():
                        shutil.rmtree(target)
                    produced.rename(target)
                val_path = ep_dir / "state_replay_validation.json"
                if val_path.exists() and read_json(val_path).get("validation_pass"):
                    accepted.append(idx)
                    done.add(idx)
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError(f"episode {idx:06d} failed validation after {args.max_retries} retries")
            write_json(progress, {
                "task": "franka_revo2_thrown_ball_cube_catch",
                "completed": len(done),
                "total": args.total,
                "completed_episodes": sorted(done),
                "latest_episode": idx,
                "tmp_root": str(tmp_root),
                "archive_root": str(archive_root),
            })
        archive = archive_batch(tmp_root, archive_root, batch_start, batch_end)
        if not args.keep_batch_tmp:
            for idx in range(batch_start, batch_end + 1):
                shutil.rmtree(tmp_root / f"episode_{idx:06d}", ignore_errors=True)
        write_json(progress, {
            "task": "franka_revo2_thrown_ball_cube_catch",
            "completed": len(done),
            "total": args.total,
            "completed_episodes": sorted(done),
            "latest_archive": str(archive),
            "tmp_root": str(tmp_root),
            "archive_root": str(archive_root),
        })
        print(json.dumps({"archived": str(archive), "completed": len(done), "total": args.total}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
