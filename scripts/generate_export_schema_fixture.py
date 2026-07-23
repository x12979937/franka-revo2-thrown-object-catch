#!/usr/bin/env python3
from __future__ import annotations

import json, math, sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from mesh_vae_encodings import export_primitive_bundle
from video2sim_schema import CAMERA_VIEWS, validate_episode_dir, write_json


def rel(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def make_video(path: Path, color, label_offset=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = 240, 320
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 12.0, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f'failed to open video writer: {path}')
    for i in range(12):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = np.array([35, 38, 42], dtype=np.uint8)
        x = 40 + i * 16 + label_offset
        y = 90 + i // 2
        img[y:y + 42, x:x + 42] = np.array(color, dtype=np.uint8)
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    writer.release()
    if not path.exists() or path.stat().st_size <= 1024:
        raise RuntimeError(f'empty or invalid video output: {path}')


def main():
    out_root = Path('/root/autodl-tmp/mingyu/video2sim/franka_revo2_thrown_ball_cube_catch/schema_fixture')
    ep = out_root / 'episode_000000' / 'thrown_ball_cube_catch'
    if ep.exists():
        import shutil; shutil.rmtree(ep)
    ep.mkdir(parents=True, exist_ok=True)

    T=12; N=2; dt=np.float32(1/60)
    names=np.array(['incoming_sphere','incoming_cube'])
    cats=np.array(['sphere','cube'])
    sizes=np.array([[0.08,0.08,0.08],[0.10,0.10,0.10]], dtype=np.float32)
    masses=np.array([0.08,0.12], dtype=np.float32)
    pos=np.zeros((T,N,3), dtype=np.float32)
    quat=np.zeros((T,N,4), dtype=np.float32); quat[...,0]=1
    lin=np.zeros((T,N,3), dtype=np.float32)
    ang=np.zeros((T,N,3), dtype=np.float32)
    for t in range(T):
        a=t/(T-1)
        pos[t,0]=[-0.45+0.5*a, 0.18, 0.85-0.2*a]
        pos[t,1]=[-0.42+0.45*a,-0.16, 0.78-0.18*a]
        lin[t,0]=[1.8,0,-0.7]; lin[t,1]=[1.5,0.1,-0.6]
        ang[t,0]=[0,6,1]; ang[t,1]=[4,2,3]
    pose=np.concatenate([pos, quat], axis=-1)
    inertia=np.array([[2/5*masses[0]*(0.04**2)]*3, [1/6*masses[1]*(0.10**2)]*3], dtype=np.float32)
    momentum=lin*masses[None,:,None]
    angular_momentum=ang*inertia[None,:,:]

    mesh_dir=ep/'meshes'
    sphere=export_primitive_bundle(mesh_dir/'incoming_sphere','incoming_sphere','sphere',(0.08,0.08,0.08),seed=1)
    cube=export_primitive_bundle(mesh_dir/'incoming_cube','incoming_cube','cube',(0.10,0.10,0.10),seed=2)
    objects=[]
    for i,(name,cat,bundle) in enumerate([(names[0],cats[0],sphere),(names[1],cats[1],cube)]):
        odir=mesh_dir/name
        objects.append({
            'name': str(name), 'category': str(cat), 'asset_source': 'mujoco_generated_primitive',
            'visual_mesh': rel(ep, odir/bundle['visual_mesh']),
            'collision_mesh': rel(ep, odir/bundle['collision_mesh']),
            'mesh_pose': pose[0,i].tolist(),
            'asset_local_coordinate_frame': {'origin_xyz':[0,0,0], 'axes':'x-right y-forward z-up'},
            'vae_encodings': {
                'point_cloud': rel(ep, odir/bundle['point_cloud']),
                'voxel': rel(ep, odir/bundle['voxel']),
                'implicit_field': rel(ep, odir/bundle['implicit_field']),
            }
        })
    write_json(ep/'mesh_assets_manifest.json', {'schema':'video2sim_mesh_assets_manifest_v2','objects':objects})

    bbox2d={view:{str(n):[[40+t*16,90+t//2,82+t*16,132+t//2] for t in range(T)] for n in names} for view in CAMERA_VIEWS}
    bbox3d={view:{str(n):np.zeros((T,8,3),dtype=float).tolist() for n in names} for view in CAMERA_VIEWS}
    masks={view:{str(n):['rle_placeholder_training_schema_fixture']*T for n in names} for view in CAMERA_VIEWS}
    ann_dir=ep/'annotations'; ann_dir.mkdir(exist_ok=True)
    write_json(ann_dir/'bbox2d_xyxy.json', bbox2d)
    write_json(ann_dir/'bbox3d_corners.json', bbox3d)
    write_json(ann_dir/'pixel_masks_rle.json', masks)
    write_json(ep/'annotation_manifest.json', {
        'schema':'video2sim_annotation_manifest_v2',
        'bbox2d_xyxy': 'annotations/bbox2d_xyxy.json',
        'bbox3d_corners': 'annotations/bbox3d_corners.json',
        'pixel_masks_rle': 'annotations/pixel_masks_rle.json',
    })
    for view in CAMERA_VIEWS:
        make_video(ep/'videos/raw'/f'{view}.mp4', (80,140,220), 0)
        make_video(ep/'videos/bbox2d'/f'{view}.mp4', (70,210,120), 4)
        make_video(ep/'videos/bbox3d'/f'{view}.mp4', (230,170,60), 8)

    camera_json=json.dumps({v:{'intrinsics':[[300,0,160],[0,300,120],[0,0,1]], 'extrinsics_front_camera_world':'identity_for_front_static_canonical'} for v in CAMERA_VIEWS})
    desc='Franka FR3 with a Revo2-style dexterous hand catches incoming thrown sphere and cube objects.'
    std='SCENE[robot=Franka_FR3+Revo2_hand; task=catch_in_flight; objects={sphere,cube}; cameras=static_front_canonical+multi_view; physics=MuJoCo]'
    np.savez_compressed(
        ep/'dataset.npz',
        schema=np.array('video2sim_unified_state_schema_v1'), task=np.array('thrown_ball_cube_catch'), physics_source=np.array('MuJoCo'), dt=dt,
        camera_view_names=np.array(CAMERA_VIEWS), camera_intrinsics_json=np.array(camera_json),
        scene_text_description=np.array(desc), scene_standard_description=np.array(std), task_text_description=np.array(desc), task_standard_description=np.array(std),
        state_replay_import_contract_json=np.array(json.dumps({'engine':'MuJoCo','replay_mode':'state_sequence_then_deterministic_resim'})),
        object_name=names, object_category=cats, object_asset_source=np.array(['mujoco_generated_primitive']*N),
        object_text_description=np.array(['Thrown smooth ping-pong-sized sphere.','Thrown rigid cube with equal square faces.']),
        object_standard_description=np.array(['OBJECT[type=sphere; role=incoming_catch_target]','OBJECT[type=cube; role=incoming_catch_target]']),
        object_pos_w=pos, object_quat_wxyz=quat, object_pose_w=pose, object_trajectory_w=pos, object_lin_vel_w=lin, object_ang_vel_w=ang,
        object_vel_w=np.concatenate([lin,ang],axis=-1), object_momentum_kg_m_s=momentum, object_angular_momentum_kg_m2_rad_s=angular_momentum,
        object_mass_kg=masses, object_center_of_mass_local_m=np.zeros((N,3),dtype=np.float32), object_center_of_mass_w=pos,
        object_inertia_diag_kg_m2=inertia, object_color_rgba=np.array([[0.2,0.6,1,1],[1,0.45,0.2,1]],dtype=np.float32), object_size_m=sizes,
        object_static_friction=np.array([0.7,0.8],dtype=np.float32), object_dynamic_friction=np.array([0.5,0.55],dtype=np.float32), object_restitution=np.array([0.7,0.45],dtype=np.float32),
        object_long_axis_w=np.tile(np.array([[1,0,0],[1,0,0]],dtype=np.float32),(T,1,1)), object_rotation_angle_from_initial_rad=np.zeros((T,N),dtype=np.float32), object_tilt_angle_rad=np.zeros((T,N),dtype=np.float32),
        initial_object_position_w=pos[0], initial_object_quat_wxyz=quat[0], initial_object_pose_w=pose[0], initial_object_lin_vel_w=lin[0], initial_object_ang_vel_w=ang[0], final_pose_w=pose[-1],
        object_visual_asset_path=np.array(['meshes/incoming_sphere/incoming_sphere_visual.obj','meshes/incoming_cube/incoming_cube_visual.obj']),
        object_collision_asset_path=np.array(['meshes/incoming_sphere/incoming_sphere_collision.obj','meshes/incoming_cube/incoming_cube_collision.obj']),
        object_visual_mesh_export_path=np.array(['meshes/incoming_sphere/incoming_sphere_visual.obj','meshes/incoming_cube/incoming_cube_visual.obj']),
        object_collision_mesh_export_path=np.array(['meshes/incoming_sphere/incoming_sphere_collision.obj','meshes/incoming_cube/incoming_cube_collision.obj']),
        object_mesh_export_dir=np.array(['meshes/incoming_sphere','meshes/incoming_cube']), object_mesh_export_format=np.array(['obj','obj']),
        object_metadata_json=np.array([json.dumps(o) for o in objects]),
        object_2d_bbox_xyxy_path_json=np.array('annotations/bbox2d_xyxy.json'), object_3d_bbox_corners_path_json=np.array('annotations/bbox3d_corners.json'), object_pixel_mask_rle_path_json=np.array('annotations/pixel_masks_rle.json'),
        contains_2d_bboxes=np.array(True), contains_3d_bboxes=np.array(True), contains_pixel_masks=np.array(True), mesh_assets_manifest_path=np.array('mesh_assets_manifest.json'),
    )
    write_json(ep/'episode_state.json', {
        'schema':'video2sim_episode_state_v2', 'objects':objects,
        'coordinate_system':{'type':'static_camera_canonical','world_origin':'front_camera_first_frame','front_camera_extrinsics':'identity'},
        'cameras':{v:{'static':True} for v in CAMERA_VIEWS},
        'videos':{variant:{v:f'videos/{variant}/{v}.mp4' for v in CAMERA_VIEWS} for variant in ('raw','bbox2d','bbox3d')},
        'assets':{'mesh_manifest':'mesh_assets_manifest.json'}, 'validation':{'path':'state_replay_validation.json'}
    })
    result=validate_episode_dir(ep, task_name='thrown_ball_cube_catch', strict_videos=True, write_validation=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['validation_pass'] else 2

if __name__ == '__main__':
    raise SystemExit(main())
