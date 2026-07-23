# Implementation plan

1. Start from the copied v730/front110 MuJoCo runner and keep robot/controller/camera setup unchanged.
2. Replace the rod/tool asset creation with parameterized sphere and cube MuJoCo bodies using real collision geoms and visual geoms.
3. Sample object type, size, mass, launch pose, launch velocity, spin, release time, and incoming sector per episode.
4. Reuse front110 hand intercept timing and grasp/closure logic, adapting target pose to predicted thrown-object intercept.
5. Export states, object mesh/geometry description, object pose, camera views, 2D/3D boxes, masks, replay validation, and compressed archives according to current video2sim baseline requirements.
6. Keep raw per-frame outputs on tmp; package completed batches into archives under fs.

## IsaacLab/IsaacGym schema compatibility

All MuJoCo-generated data for this task must be exported in the same format family as the IsaacLab/IsaacGym datasets already produced for demo1, demo2, demo3, and arc-rods. The MuJoCo runner may keep engine-specific diagnostics, but the public training/replay data must expose the same logical structure: episode manifest, per-object state, camera metadata, RGB/raw video, boxed video, 2D boxes, 3D boxes, pixel masks, mesh bundle, VAE-ready mesh numeric encodings, text descriptions, and state replay validation.

Before large-scale generation, run a schema smoke test that compares one MuJoCo episode against one Isaac-style reference episode and fails if required shared fields or paths are missing.
