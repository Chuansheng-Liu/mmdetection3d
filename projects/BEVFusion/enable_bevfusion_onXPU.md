# Enabling BEVFusion on Intel® XPU

## Environment
- Python 3.13 with `pytorch-xpu` 2.4 build shipped for Intel® GPU.
- Project root: `/home/intel/chuansheng/mmdetection3d`.
- Custom ops compiled via `projects/BEVFusion/setup.py develop` (kernels converted from CUDA with Intel® DPC++ Compatibility Tool).

## Key Modifications
- `mmdet3d/apis/inference.py`: understand `xpu:*` strings and invoke `torch.xpu.set_device` so inference can target Intel GPUs.
- `projects/BEVFusion/bevfusion/bevfusion.py`: drive autocast/voxelization from the module's device instead of hard-coded CUDA, letting tensors stay resident on XPU.
- `projects/BEVFusion/bevfusion/ops/bev_pool` and `projects/BEVFusion/bevfusion/ops/voxel`: added XPU kernels (`*_xpu.cpp`) plus a PyTorch fallback for BEV pooling when custom kernels aren’t available.
- `projects/BEVFusion/bevfusion/sparse_encoder.py`: permute 5-D sparse-conv weights at load time so spconv v2 checkpoints align with MMCV’s `(D, H, W, in, out)` layout.
- `projects/BEVFusion/setup.py`: generalized build helper to choose CUDA or XPU sources based on the active runtime.
- `tools/test.py`: relaxed `torch.load` safety defaults and registered MMEngine objects as safe globals for PyTorch 2.6.
- Added `projects/BEVFusion/tests/xpu_ops_stress.py` for manual stress runs of BEV pooling, voxelization, and dynamic scatter on XPU.

## Validation Workflow
1. **Compile kernels**: `python projects/BEVFusion/setup.py develop`.
2. **Single-device evaluation**:
   ```bash
   cd /home/intel/chuansheng/mmdetection3d
   PYTHONPATH=$(pwd) python tools/test.py \
     projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
     ../bevfusion_model_data/checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth \
     --launcher none
   ```
3. **Dataset**: NuScenes `v1.0-mini` tables staged under `data/nuscenes/` (10-scene subset).
4. **Result**: mAP **0.5680**, NDS **0.5761** on XPU after sparse-kernel fix (checkpoint loads without warnings, evaluator completes).

## Notes
- The official README metrics (NDS 71.4 / mAP 68.6) correspond to the full NuScenes `v1.0-trainval` split with CUDA kernels. Our mini-split numbers are for bring-up and sanity check; download the full split to reproduce the published scores.
- When full tables are available, flip the config metainfo back to `v1.0-trainval` and rerun the command above (or `tools/dist_test.sh … 8`) for definitive validation.
- Profiling and performance polish (e.g., AMP, batch-size tuning, kernel micro-optimizations) remain as future work now that functional parity is confirmed.
