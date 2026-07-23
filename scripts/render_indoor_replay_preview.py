import json
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT = Path("/autodl-fs/data/mingyu/Mujoco/projects/franka_revo2_thrown_ball_cube_catch")
sys.path.insert(0, str(PROJECT / "scripts"))

import run_thrown_ball_cube_episode as sim


def main():
    src_ep = Path(sys.argv[1])
    out_root = Path(sys.argv[2])
    fps = int(sys.argv[3])
    width = int(sys.argv[4])
    height = int(sys.argv[5])
    out_ep = out_root / src_ep.parent.name / src_ep.name
    out_ep.mkdir(parents=True, exist_ok=True)

    z = np.load(src_ep / "dataset.npz", allow_pickle=True)
    obj_type = str(z["object_category"][0])
    size = float(np.max(z["object_size_m"][0]))
    mass = float(z["object_mass_kg"][0])
    color = z["object_color_rgba"][0].astype(float).tolist()

    scene_xml = out_ep / "scene.xml"
    sim.build_scene(scene_xml, obj_type, size, mass, color)
    for name in ("dataset.npz", "episode_state.json", "state_replay_validation.json", "mesh_assets_manifest.json", "annotation_manifest.json"):
        src = src_ep / name
        if src.exists():
            shutil.copy2(src, out_ep / name)

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    seg_renderer = mujoco.Renderer(model, height=height, width=width)
    seg_renderer.enable_segmentation_rendering()
    obj_geom_id = int(model.geom("incoming_object_geom").id)
    geom_obj_type = int(mujoco.mjtObj.mjOBJ_GEOM)
    obj_qadr = model.jnt_qposadr[model.joint("incoming_object_free").id]
    obj_vadr = model.jnt_dofadr[model.joint("incoming_object_free").id]

    views = [str(v) for v in z["camera_view_names"]]
    frames_raw = {v: [] for v in views}
    frames_2d = {v: [] for v in views}
    frames_3d = {v: [] for v in views}
    frames_mask = {v: [] for v in views}

    n = int(z["object_pos_w"].shape[0])
    for i in range(n):
        if model.nq >= 18:
            data.qpos[:18] = z["robot_qpos_trajectory"][i]
        if model.nv >= 18:
            data.qvel[:18] = z["robot_qvel_trajectory"][i]
        if model.nu >= 18:
            data.ctrl[:18] = z["robot_ctrl_trajectory"][i]
        data.qpos[obj_qadr:obj_qadr + 3] = z["object_pos_w"][i, 0]
        data.qpos[obj_qadr + 3:obj_qadr + 7] = z["object_quat_wxyz"][i, 0]
        data.qvel[obj_vadr:obj_vadr + 3] = z["object_lin_vel_w"][i, 0]
        data.qvel[obj_vadr + 3:obj_vadr + 6] = z["object_ang_vel_w"][i, 0]
        mujoco.mj_forward(model, data)

        corners = sim.bbox_corners(obj_type, size, z["object_pos_w"][i, 0], z["object_quat_wxyz"][i, 0])
        for view in views:
            cam_id = model.camera(view).id
            renderer.update_scene(data, camera=cam_id)
            raw = renderer.render()
            seg_renderer.update_scene(data, camera=cam_id)
            seg = seg_renderer.render()
            mask = ((seg[:, :, 0] == obj_geom_id) & (seg[:, :, 1] == geom_obj_type)).astype(np.uint8)
            K, cam_pos, cam_rot = sim.camera_matrix(model, data, cam_id, width=width, height=height)
            proj = sim.project_points(corners, K, cam_pos, cam_rot)
            ys, xs = np.where(mask > 0)
            if len(xs):
                box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
            else:
                valid = np.isfinite(proj[:, 0]) & (proj[:, 2] > 1e-4)
                if valid.any():
                    xy = proj[valid, :2]
                    x1, y1 = np.maximum(np.floor(xy.min(axis=0)), [0, 0])
                    x2, y2 = np.minimum(np.ceil(xy.max(axis=0)), [width - 1, height - 1])
                    box = [float(x1), float(y1), float(x2), float(y2)]
                else:
                    box = [0.0, 0.0, 0.0, 0.0]
            img2 = raw.copy()
            sim.draw_bbox2d(img2, box)
            img3 = raw.copy()
            sim.draw_bbox3d(img3, proj[:, :2])
            imgm = sim.draw_mask_overlay(raw, mask)
            frames_raw[view].append(raw)
            frames_2d[view].append(img2)
            frames_3d[view].append(img3)
            frames_mask[view].append(imgm)

    for view in views:
        sim.write_video(out_ep / "videos/raw" / f"{view}.mp4", frames_raw[view], fps)
        sim.write_video(out_ep / "videos/bbox2d" / f"{view}.mp4", frames_2d[view], fps)
        sim.write_video(out_ep / "videos/bbox3d" / f"{view}.mp4", frames_3d[view], fps)
        sim.write_video(out_ep / "videos/mask" / f"{view}.mp4", frames_mask[view], fps)

    (out_root / "render_replay_summary.json").write_text(json.dumps({"source_episode": str(src_ep), "episode": str(out_ep)}, indent=2), encoding="utf-8")
    print(out_ep)


if __name__ == "__main__":
    main()
