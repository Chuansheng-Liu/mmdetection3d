# Enabling BEVFusion on Intel® XPU

## Environment
- Python 3.13 with `pytorch-xpu` 2.4 build shipped for Intel® GPU.
- Project root: `${MMDET3D_ROOT}` (export this to point at your checkout).
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
   cd "$MMDET3D_ROOT"
   PYTHONPATH=$(pwd) python tools/test.py \
     projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
     "${BEVFUSION_MODEL_DIR}/checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth" \
     --launcher none
   ```
3. **Dataset**: NuScenes `v1.0-mini` tables staged under `data/nuscenes/` (10-scene subset).
4. **Result**: mAP **0.5680**, NDS **0.5761** on XPU after sparse-kernel fix (checkpoint loads without warnings, evaluator completes).

### Demo QA (No Open3D)
- Ran `projects/BEVFusion/demo/multi_modality_demo.py` on `n015-2018-07-24-11-22-45+0800` with the converted BEVFusion checkpoint on `xpu:0`. Visualization is skipped, but predictions are serialized to `work_dirs/bevfusion_demo_result.json`.
- `projects/BEVFusion/demo/export_predictions.py` converts the serialized `Det3DDataSample` into NuScenes-style detections plus a per-class summary. With `--score-thr 0.3` the run keeps 10 boxes (truck, car, pedestrians, barriers) and writes:
  - `work_dirs/bevfusion_demo_detection.json`
  - `work_dirs/bevfusion_demo_summary.txt`
- `projects/BEVFusion/demo/batch_export_predictions.py` aggregates multiple samples. On the demo result with `--score-thr 0.1` it retains 19 boxes and emits:
  - `work_dirs/bevfusion_batch_detection.json`
  - `work_dirs/bevfusion_batch_summary.txt`
- These helpers unblock quick QA while we defer building Open3D for Python 3.13. They accept alternative `--result-json`/`--ann` inputs, so additional samples can be exported in bulk once their serialized outputs are collected.

#### Step-by-step commands

```bash
# 1. Activate the PyTorch XPU environment and run a single-sample demo inference
. "${PYTORCH_XPU_ENV}/bin/activate"
cd "$MMDET3D_ROOT"
PYTHONPATH=$(pwd) SYCL_DEVICE_FILTER=level_zero:gpu \
  python projects/BEVFusion/demo/multi_modality_demo.py \
    projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
    "${BEVFUSION_MODEL_DIR}/bevfusion_converted.pth" \
    --pcd demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin \
    --image-root demo/data/nuscenes/ \
    --ann demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl \
    --device xpu:0 \
    --save-json work_dirs/bevfusion_demo_result.json

# 2. Convert that JSON into NuScenes-format detections and a class summary
python projects/BEVFusion/demo/export_predictions.py \
  --result-json work_dirs/bevfusion_demo_result.json \
  --ann demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl \
  --out-json work_dirs/bevfusion_demo_detection.json \
  --summary-txt work_dirs/bevfusion_demo_summary.txt \
  --score-thr 0.3

# 3. Optionally batch more samples (glob pattern) and relax the score filter
python projects/BEVFusion/demo/batch_export_predictions.py \
  --result-glob 'work_dirs/*.json' \
  --ann-root demo/data/nuscenes \
  --out-json work_dirs/bevfusion_batch_detection.json \
  --summary-txt work_dirs/bevfusion_batch_summary.txt \
  --score-thr 0.1
```

#### Result snapshot
- Single-sample demo (step 1) produces 200 raw 3D detections; after applying `--score-thr 0.3` (step 2) 10 remain, dominated by pedestrians plus one truck and one car. The summary file reports mean scores per class (`truck 0.697`, `car 0.550`, `pedestrian 0.485`, `barrier 0.304`).
- Batch export (step 3) on the same result with `--score-thr 0.1` yields 19 boxes and updates the summary accordingly (`barrier` count rises to 9, `pedestrian` to 8). These JSON files can now feed NuScenes evaluation tooling or further QA scripts without Open3D.

### XPU Ops Stress Script
`projects/BEVFusion/tests/xpu_ops_stress.py` dynamically imports the BEV pooling and voxelization extensions and exercises them on an Intel XPU. It runs multiple configurations of `bev_pool`, voxelization, and `dynamic_scatter`, performing forward/backward passes while timing each case. Use it after building the extensions:

```bash
cd "$MMDET3D_ROOT"
python projects/BEVFusion/tests/xpu_ops_stress.py
```

Successful completion indicates the ported kernels are functional on XPU and provides quick sanity timings.

## BEVFusion Ops Coverage

| Operator | Python entry point | Native sources | XPU implementation | CPU fallback | Test / status |
| --- | --- | --- | --- | --- | --- |
| BEV pooling | `bevfusion/ops/bev_pool/bev_pool.py` | `bevfusion/ops/bev_pool/src` | `bevfusion/ops/bev_pool/src/bev_pool_xpu.cpp` | Python fallback inside `bev_pool.py` when the extension is unavailable | Covered by `projects/BEVFusion/tests/xpu_ops_stress.py` (configs #1-3) |
| Voxelization | `bevfusion/ops/voxel/voxelize.py` | `bevfusion/ops/voxel/src` | `bevfusion/ops/voxel/src/voxelization_xpu.cpp` | `bevfusion/ops/voxel/src/voxelization_cpu.cpp` | Stress tested via `xpu_ops_stress.py` alongside BEV pooling |
| Dynamic scatter (point-to-voxel) | Imported from MMCV (`mmcv.ops.dynamic_scatter`) | `mmcv/ops/csrc/pytorch/{cpu,xpu}/scatter_points*.cpp` | `mmcv/ops/csrc/pytorch/xpu/scatter_points_xpu.cpp` | `mmcv/ops/csrc/pytorch/cpu/scatter_points_cpu.cpp` | Validated by MMCV `pytest tests/test_ops/test_spconv.py -k dynamic_scatter` and BEVFusion stress configs |

All three operators build through `projects/BEVFusion/setup.py develop` when the Intel XPU toolchain is active. The stress script exercises forward/backward passes on XPU, and the `tests/test_ops -k xpu` suite in MMCV provides additional regression coverage for the shared kernels.

## Notes
- The official README metrics (NDS 71.4 / mAP 68.6) correspond to the full NuScenes `v1.0-trainval` split with CUDA kernels. Our mini-split numbers are for bring-up and sanity check; download the full split to reproduce the published scores.
- When full tables are available, flip the config metainfo back to `v1.0-trainval` and rerun the command above (or `tools/dist_test.sh … 8`) for definitive validation.
- Profiling and performance polish (e.g., AMP, batch-size tuning, kernel micro-optimizations) remain as future work now that functional parity is confirmed.
