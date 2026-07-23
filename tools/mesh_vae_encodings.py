#!/usr/bin/env python3
"""Small deterministic mesh encodings for VAE-ready inputs.

For primitive MuJoCo objects this creates explicit mesh assets plus three numeric
representations: point cloud, voxel occupancy, and implicit signed-distance
samples. The arrays are intentionally plain npz so Isaac and MuJoCo datasets can
share consumers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import math
import numpy as np


def cube_vertices(edge: float) -> np.ndarray:
    h = edge / 2.0
    return np.array([[x,y,z] for x in (-h,h) for y in (-h,h) for z in (-h,h)], dtype=np.float32)


def cube_faces() -> np.ndarray:
    return np.array([
        [0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],
        [2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]
    ], dtype=np.int32)


def sphere_mesh(radius: float, rings: int = 16, sectors: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    verts=[]; faces=[]
    for r in range(rings + 1):
        theta = math.pi * r / rings
        for s in range(sectors):
            phi = 2.0 * math.pi * s / sectors
            verts.append([radius*math.sin(theta)*math.cos(phi), radius*math.sin(theta)*math.sin(phi), radius*math.cos(theta)])
    for r in range(rings):
        for s in range(sectors):
            a = r*sectors + s; b = r*sectors + (s+1)%sectors; c = (r+1)*sectors + s; d = (r+1)*sectors + (s+1)%sectors
            faces.append([a,c,b]); faces.append([b,c,d])
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def write_obj(path: str | Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for v in vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


def sample_point_cloud(vertices: np.ndarray, count: int = 2048, seed: int = 0) -> np.ndarray:
    rng=np.random.default_rng(seed)
    idx=rng.integers(0, len(vertices), size=count)
    pts=vertices[idx].astype(np.float32)
    if len(vertices) >= 8:
        pts += rng.normal(0, 0.002, pts.shape).astype(np.float32)
    return pts


def voxelize_primitive(kind: str, size: Tuple[float,float,float], resolution: int = 32) -> np.ndarray:
    xs=np.linspace(-0.5,0.5,resolution,dtype=np.float32)
    grid=np.stack(np.meshgrid(xs,xs,xs,indexing='ij'),axis=-1)
    if kind == 'sphere':
        occ=(np.linalg.norm(grid,axis=-1) <= 0.5).astype(np.uint8)
    else:
        occ=(np.max(np.abs(grid),axis=-1) <= 0.5).astype(np.uint8)
    return occ


def implicit_samples(kind: str, size: Tuple[float,float,float], count: int = 8192, seed: int = 0) -> np.ndarray:
    rng=np.random.default_rng(seed)
    pts=rng.uniform(-0.75,0.75,size=(count,3)).astype(np.float32)
    if kind == 'sphere':
        sdf=(np.linalg.norm(pts,axis=1) - 0.5).astype(np.float32)
    else:
        q=np.abs(pts)-0.5
        outside=np.linalg.norm(np.maximum(q,0),axis=1)
        inside=np.minimum(np.maximum.reduce(q,axis=1),0)
        sdf=(outside+inside).astype(np.float32)
    return np.concatenate([pts, sdf[:,None]], axis=1)


def export_primitive_bundle(out_dir: str | Path, name: str, kind: str, size: Tuple[float,float,float], seed: int = 0) -> Dict[str,str]:
    out=Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if kind == 'sphere':
        vertices, faces=sphere_mesh(float(size[0]) / 2.0)
    elif kind == 'cube':
        vertices, faces=cube_vertices(float(size[0])), cube_faces()
    else:
        raise ValueError(f'unsupported primitive kind: {kind}')
    visual=out / f'{name}_visual.obj'
    collision=out / f'{name}_collision.obj'
    write_obj(visual, vertices, faces); write_obj(collision, vertices, faces)
    pc=sample_point_cloud(vertices, seed=seed)
    vox=voxelize_primitive(kind, size)
    imp=implicit_samples(kind, size, seed=seed)
    pc_path=out / f'{name}_vae_point_cloud.npz'
    vox_path=out / f'{name}_vae_voxel.npz'
    imp_path=out / f'{name}_vae_implicit_field.npz'
    np.savez_compressed(pc_path, points=pc, schema='video2sim_vae_point_cloud_v1')
    np.savez_compressed(vox_path, occupancy=vox, resolution=np.array([vox.shape[0]], dtype=np.int32), schema='video2sim_vae_voxel_v1')
    np.savez_compressed(imp_path, samples=imp, columns=np.array(['x','y','z','sdf']), schema='video2sim_vae_implicit_field_v1')
    return {
        'visual_mesh': visual.name,
        'collision_mesh': collision.name,
        'point_cloud': pc_path.name,
        'voxel': vox_path.name,
        'implicit_field': imp_path.name,
    }
