#!/usr/bin/env python3
"""Shared video2sim export schema checks for MuJoCo episodes.

This module is intentionally engine-neutral. MuJoCo runners should export their
state into the same field family used by the IsaacLab/IsaacGym datasets, then
call validate_episode_dir() before accepting an episode.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

SCHEMA_NAME = "video2sim_unified_state_schema_v1"
VALIDATION_SCHEMA = "video2sim_state_replay_validation_v3"
CAMERA_VIEWS = (
    "front", "top", "left", "right", "left_oblique", "right_oblique",
)
VIDEO_VARIANTS = ("raw", "bbox2d", "bbox3d")
VAE_MESH_VARIANTS = ("point_cloud", "voxel", "implicit_field")

# Common fields required to match the IsaacLab/IsaacGym video2sim data family.
REQUIRED_NPZ_FIELDS = (
    "schema", "task", "physics_source", "dt", "camera_view_names",
    "camera_intrinsics_json", "scene_text_description", "scene_standard_description",
    "task_text_description", "task_standard_description",
    "state_replay_import_contract_json",
    "object_name", "object_category", "object_asset_source",
    "object_text_description", "object_standard_description",
    "object_pos_w", "object_quat_wxyz", "object_pose_w", "object_trajectory_w",
    "object_lin_vel_w", "object_ang_vel_w", "object_vel_w",
    "object_momentum_kg_m_s", "object_angular_momentum_kg_m2_rad_s",
    "object_mass_kg", "object_center_of_mass_local_m", "object_center_of_mass_w",
    "object_inertia_diag_kg_m2", "object_color_rgba", "object_size_m",
    "object_static_friction", "object_dynamic_friction", "object_restitution",
    "object_long_axis_w", "object_rotation_angle_from_initial_rad", "object_tilt_angle_rad",
    "initial_object_position_w", "initial_object_quat_wxyz", "initial_object_pose_w",
    "initial_object_lin_vel_w", "initial_object_ang_vel_w",
    "final_pose_w", "object_visual_asset_path", "object_collision_asset_path",
    "object_visual_mesh_export_path", "object_collision_mesh_export_path",
    "object_mesh_export_dir", "object_mesh_export_format", "object_metadata_json",
    "object_2d_bbox_xyxy_path_json", "object_3d_bbox_corners_path_json",
    "object_pixel_mask_rle_path_json", "contains_2d_bboxes", "contains_3d_bboxes",
    "contains_pixel_masks", "mesh_assets_manifest_path",
)

REQUIRED_STATE_JSON_KEYS = (
    "objects", "cameras", "coordinate_system", "videos", "assets", "validation",
)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_npz(path: str | Path) -> Mapping[str, np.ndarray]:
    return np.load(Path(path), allow_pickle=True)


def quat_norm_error(quats: np.ndarray) -> float:
    q = np.asarray(quats, dtype=np.float64).reshape(-1, 4)
    if q.size == 0:
        return 0.0
    return float(np.max(np.abs(np.linalg.norm(q, axis=1) - 1.0)))


def finite_error_count(fields: Iterable[np.ndarray]) -> int:
    total = 0
    for arr in fields:
        a = np.asarray(arr)
        if np.issubdtype(a.dtype, np.number):
            total += int(np.size(a) - np.count_nonzero(np.isfinite(a)))
    return total


def required_episode_paths(episode_dir: str | Path) -> Dict[str, Path]:
    root = Path(episode_dir)
    return {
        "dataset": root / "dataset.npz",
        "episode_state": root / "episode_state.json",
        "mesh_manifest": root / "mesh_assets_manifest.json",
        "annotation_manifest": root / "annotation_manifest.json",
        "validation": root / "state_replay_validation.json",
    }


def expected_video_paths(episode_dir: str | Path, views: Sequence[str] = CAMERA_VIEWS) -> List[Path]:
    root = Path(episode_dir)
    paths: List[Path] = []
    for view in views:
        for variant in VIDEO_VARIANTS:
            paths.append(root / "videos" / variant / f"{view}.mp4")
    return paths


def validate_episode_dir(
    episode_dir: str | Path,
    *,
    task_name: str,
    physics_engine: str = "MuJoCo",
    strict_videos: bool = True,
    write_validation: bool = True,
) -> Dict[str, Any]:
    episode_dir = Path(episode_dir)
    paths = required_episode_paths(episode_dir)
    failure_reasons: List[str] = []
    warnings: List[str] = []

    if not paths["dataset"].exists():
        failure_reasons.append("missing dataset.npz")
        data = None
    else:
        data = load_npz(paths["dataset"])
        missing = [k for k in REQUIRED_NPZ_FIELDS if k not in data.files]
        if missing:
            failure_reasons.append("missing dataset fields: " + ", ".join(missing))

    for key in ("episode_state", "mesh_manifest", "annotation_manifest"):
        if not paths[key].exists():
            failure_reasons.append(f"missing {paths[key].name}")

    raw_video_count = annotated_2d_count = annotated_3d_count = 0
    if strict_videos:
        for video_path in expected_video_paths(episode_dir):
            if not video_path.exists() or video_path.stat().st_size <= 0:
                failure_reasons.append(f"missing_or_empty_video:{video_path.relative_to(episode_dir)}")
            elif "/raw/" in str(video_path):
                raw_video_count += 1
            elif "/bbox2d/" in str(video_path):
                annotated_2d_count += 1
            elif "/bbox3d/" in str(video_path):
                annotated_3d_count += 1

    frame_count = object_count = 0
    nonfinite_count = 0
    max_qerr = 0.0
    max_linear_speed = 0.0
    max_angular_speed = 0.0
    max_initial_position_error = None
    max_final_position_error = None
    contains_2d = contains_3d = contains_masks = False

    if data is not None and not failure_reasons:
        pos = np.asarray(data["object_pos_w"])
        quat = np.asarray(data["object_quat_wxyz"])
        lin = np.asarray(data["object_lin_vel_w"])
        ang = np.asarray(data["object_ang_vel_w"])
        frame_count = int(pos.shape[0])
        object_count = int(pos.shape[1]) if pos.ndim >= 2 else 0
        nonfinite_count = finite_error_count([pos, quat, lin, ang, data["object_mass_kg"], data["object_inertia_diag_kg_m2"]])
        max_qerr = quat_norm_error(quat)
        max_linear_speed = float(np.max(np.linalg.norm(lin.reshape(-1, 3), axis=1))) if lin.size else 0.0
        max_angular_speed = float(np.max(np.linalg.norm(ang.reshape(-1, 3), axis=1))) if ang.size else 0.0
        max_initial_position_error = float(np.max(np.linalg.norm(pos[0] - data["initial_object_position_w"], axis=-1)))
        max_final_position_error = float(np.max(np.linalg.norm(data["final_pose_w"][:, :3] - pos[-1], axis=-1)))
        contains_2d = bool(np.asarray(data["contains_2d_bboxes"]).item())
        contains_3d = bool(np.asarray(data["contains_3d_bboxes"]).item())
        contains_masks = bool(np.asarray(data["contains_pixel_masks"]).item())
        if nonfinite_count:
            failure_reasons.append(f"nonfinite numeric values: {nonfinite_count}")
        if max_qerr > 1e-3:
            failure_reasons.append(f"quaternion norm error too large: {max_qerr}")
        if not contains_2d:
            failure_reasons.append("contains_2d_bboxes is false")
        if not contains_3d:
            failure_reasons.append("contains_3d_bboxes is false")
        if not contains_masks:
            failure_reasons.append("contains_pixel_masks is false")

    mesh_manifest_count = 0
    vae_npz_count = 0
    missing_mesh_files: List[str] = []
    if paths["mesh_manifest"].exists():
        manifest = json.loads(paths["mesh_manifest"].read_text(encoding="utf-8"))
        objects = manifest.get("objects", [])
        mesh_manifest_count = len(objects)
        for obj in objects:
            for attr in ("visual_mesh", "collision_mesh"):
                p = obj.get(attr)
                if p and not (episode_dir / p).exists() and not Path(p).exists():
                    missing_mesh_files.append(str(p))
            enc = obj.get("vae_encodings", {})
            for attr in VAE_MESH_VARIANTS:
                p = enc.get(attr)
                if p and ((episode_dir / p).exists() or Path(p).exists()):
                    vae_npz_count += 1
                else:
                    missing_mesh_files.append(f"missing_vae_{attr}:{obj.get('name','unknown')}")
    if missing_mesh_files:
        failure_reasons.append("missing mesh/vae files: " + "; ".join(missing_mesh_files[:20]))

    result: Dict[str, Any] = {
        "schema": VALIDATION_SCHEMA,
        "task": task_name,
        "episode_dir": str(episode_dir),
        "validation_pass": not failure_reasons,
        "failure_reasons": failure_reasons,
        "warnings": warnings,
        "validation_scope": {
            "physics_engine": physics_engine,
            "state_sequence_importable": data is not None and not missing_mesh_files,
            "engine_replay_verified": False,
            "note": "This file validates export completeness and state-sequence importability. Engine deterministic replay should set engine_replay_verified=true after reloading MuJoCo state and comparing trajectories/contacts.",
        },
        "state_sequence": {
            "frame_count": frame_count,
            "object_count": object_count,
            "nonfinite_field_count": nonfinite_count,
            "max_quaternion_norm_error": max_qerr,
            "max_linear_speed_m_s": max_linear_speed,
            "max_angular_speed_rad_s": max_angular_speed,
        },
        "trajectory_error": {
            "max_initial_position_error_m": max_initial_position_error,
            "max_final_position_error_m": max_final_position_error,
        },
        "contact_error": {
            "contact_events_available": data is not None and ("contact_flags" in getattr(data, "files", [])),
            "contact_error_pending_engine_replay": True,
        },
        "penetration_check": {
            "penetration_available": data is not None and ("max_penetration_m" in getattr(data, "files", [])),
            "threshold_m": 0.005,
            "requires_mujoco_contact_replay": True,
        },
        "task_constraint_check": {
            "requires_task_specific_validator": True,
            "auto_rebuild_if_failed": True,
        },
        "visual_supervision": {
            "raw_video_count": raw_video_count,
            "bbox2d_video_count": annotated_2d_count,
            "bbox3d_video_count": annotated_3d_count,
            "contains_2d_bboxes": contains_2d,
            "contains_3d_bboxes": contains_3d,
            "contains_pixel_masks": contains_masks,
        },
        "assets": {
            "mesh_manifest_object_count": mesh_manifest_count,
            "missing_mesh_files": missing_mesh_files,
            "vae_npz_count": vae_npz_count,
        },
    }
    if write_validation:
        write_json(paths["validation"], result)
    return result


def schema_smoke_test(reference_episode_dirs: Sequence[str | Path]) -> Dict[str, Any]:
    reports = []
    for episode in reference_episode_dirs:
        episode = Path(episode)
        task = episode.name
        reports.append(validate_episode_dir(episode, task_name=task, physics_engine="Reference", strict_videos=False, write_validation=False))
    return {"schema": SCHEMA_NAME, "reference_count": len(reports), "reports": reports}
