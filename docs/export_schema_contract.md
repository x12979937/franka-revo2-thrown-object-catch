# Unified video2sim export contract for MuJoCo tasks

This task must export MuJoCo episodes in the same format family as the IsaacLab/IsaacGym video2sim datasets.

Required per episode:

- `dataset.npz`: dense numeric state and metadata fields.
- `episode_state.json`: readable object/camera/coordinate/video/asset manifest summary.
- `mesh_assets_manifest.json`: visual mesh, collision mesh, mesh pose, local asset coordinate frame, and VAE encodings for each object.
- `annotation_manifest.json`: 2D boxes, 3D boxes, pixel masks, and per-view annotation paths.
- `state_replay_validation.json`: generated before accepting the episode.
- `videos/raw/{front,top,left,right,left_oblique,right_oblique}.mp4`.
- `videos/bbox2d/{front,top,left,right,left_oblique,right_oblique}.mp4`.
- `videos/bbox3d/{front,top,left,right,left_oblique,right_oblique}.mp4`.

Coordinate rule for static-camera tasks:

- The front camera first-frame camera coordinate system is the canonical world frame.
- Front camera extrinsics are identity.
- All camera poses, point maps, 3D points, object poses, mesh poses, and boxes are expressed in this canonical frame.

Episode acceptance rule:

- Generate one episode under tmp.
- Export full state, videos, annotations, mesh bundle, and VAE mesh matrices.
- Import/replay the state sequence in MuJoCo.
- Write `state_replay_validation.json`.
- If validation fails, delete that tmp episode and regenerate it with a new seed.
- Only validated episodes are packed into compressed archives on fs.
