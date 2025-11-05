# BEVFusion INT8 Post-Training Quantization on Intel XPU

This note documents the workflow for generating and validating an INT8
quantized BEVFusion detector that runs on Intel XPU. The flow keeps modules
without native XPU kernels in FP32 while quantizing the rest of the model using
PyTorch FX.

## Prerequisites

- Intel PyTorch-XPU stack with oneAPI/oneDNN backend enabled.
- NuScenes v1.0 dataset prepared per the main BEVFusion README.
- Calibration subset generated with
  `tools/quantization/build_nuscenes_calib_subset.py` and stored at
  `data/nuscenes/calib_subset/nuscenes_infos_calib.pkl`.
- FP32 fusion checkpoint (e.g.
  `bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth`).

All commands below assume the `mmdetection3d` repository root as the working
directory.

## Step 1: Produce the Quantized Checkpoint

```bash
python tools/quantization/bevfusion_ptq.py \
  --config projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
  --checkpoint /path/to/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth \
  --calib-ann-file data/nuscenes/calib_subset/nuscenes_infos_calib.pkl \
  --device xpu \
  --batch-size 1 \
  --num-workers 2 \
  --calibrate-batches 300 \
  --output work_dirs/bevfusion_quantized_ptq.pth
```

Key details:

- The script deep-copies the runner model before calling `prepare_fx` so the
  original FP32 instance remains untouched.
- `--skip-modules` defaults keep `pts_backbone`, image branches, and custom
  sparse ops in FP32. This avoids unsupported kernels such as
  `quantized::conv2d_relu.new` on XPU.
- Conversion happens on CPU (`convert_fx(prepared.cpu())`) to match torch.ao FX
  expectations, and the resulting INT8 weights are saved at the path provided
  via `--output`.

## Step 2: Rehydrate and Evaluate the Quantized Graph

```bash
python tools/quantization/bevfusion_ptq_eval.py \
  --config projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
  --checkpoint /path/to/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth \
  --quantized-checkpoint work_dirs/bevfusion_quantized_ptq.pth \
  --calib-ann-file data/nuscenes/calib_subset/nuscenes_infos_calib.pkl \
  --device xpu \
  --batch-size 1 \
  --num-workers 2 \
  --run-eval --eval-split val
```

What happens under the hood:

1. The helper rebuilds the FX graph with the same skip list used in Step 1.
2. FP32 versions of `pts_backbone`, voxel ops, and fusion blocks are restored on
   top of the quantized graph to guarantee XPU execution.
3. The script swaps the runner model to the restored graph and runs NuScenes
   evaluation. An optional `--output /path/to/graph.pth` argument can be used to
   export the FX `GraphModule` snapshot for deployment.

### Optional Kernel Diagnostics

The default skip list keeps execution entirely on XPU. To double-check, you can
trace fallback decisions by setting `PYTORCH_DEBUG_XPU_FALLBACK=1` for a short
subset, or simply leave all fallback environment variables unset and confirm the
absence of warning messages.

## Observed Metrics

- Quantized INT8 run: `NDS = 0.7120`, `mAP = 0.6837` on NuScenes validation.
- FP32 baseline (XPU single device): `NDS = 0.712`, `mAP = 0.684`.

The quantized model matches the FP32 baseline within rounding error while
preserving XPU-only execution.

### Latency Benchmark

Two quick options are available to inspect throughput without burning through the
entire NuScenes validation split:

```bash
# INT8 graph (evaluate ~200 samples from the calibration subset)
python tools/analysis_tools/benchmark.py \
  projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
  work_dirs/bevfusion_quantized_ptq.pth \
  --device xpu:0 \
  --samples 200 \
  --num-warmup 20 \
  --cfg-options test_dataloader.dataset.ann_file=data/nuscenes/calib_subset/nuscenes_infos_calib.pkl

# Inline benchmarking during graph restoration (shares the same dataset overrides)
python tools/quantization/bevfusion_ptq_eval.py \
  --config projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
  --checkpoint ${FP32_CHECKPOINT} \
  --quantized-checkpoint work_dirs/bevfusion_quantized_ptq.pth \
  --calib-ann-file data/nuscenes/calib_subset/nuscenes_infos_calib.pkl \
  --device xpu:0 \
  --benchmark-samples 200 \
  --benchmark-warmup 20 \
  --cfg-options val_dataloader.dataset.ann_file=data/nuscenes/calib_subset/nuscenes_infos_calib.pkl
```

Adjust `--samples`/`--benchmark-samples` and the calibration subset path to fit
your environment; both commands keep execution on XPU and stop once the requested
sample count is measured.

## Outputs

- Quantized checkpoint: `work_dirs/bevfusion_quantized_ptq.pth`.
- Optional FX graph export: provide `--output` to
  `tools/quantization/bevfusion_ptq_eval.py`.
- Evaluation results: printed to console and written under the temporary NuScenes
  evaluation directory (e.g. `/tmp/.../results_nusc.json`).

## Troubleshooting

- **Missing calibration subset:** run
  `tools/quantization/build_nuscenes_calib_subset.py --help` to create the
  subset from NuScenes info tables.
- **Unsupported kernel errors:** ensure the skip list includes the module name
  (for example, `pts_backbone`). Keeping those modules in FP32 avoids the
  unsupported `quantized::conv2d_relu.new` path.
- **Observer warnings:** torch.ao FX emits deprecation notes; they can be safely
  ignored for PyTorch 2.6 era stacks, but migrating to `torchao` PT2E APIs is a
  future task.
