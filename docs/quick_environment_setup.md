# Quick Environment Setup

These notes cover the minimum steps to stand up a working MMDetection3D development environment on Linux. Adjust versions to match your GPU drivers and project needs.

## Reference Versions

| Module  | Version                      |
|---------|------------------------------|
| Python  | 3.13                         |
| torch   | 2.10.0.dev20251105+xpu       |
| oneAPI  | 2025.2                       |

## 1. Create and Activate a Virtual Environment

Always isolate dependencies. In this guide we first create a `venv` named `bevfusion_p3.13_venv`.

### Using `python -m venv`
```bash
python3 -m venv ~/envs/bevfusion_p3.13_venv
source ~/envs/bevfusion_p3.13_venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## 2. Load Intel oneAPI Toolchain

For Intel XPU builds you must load the oneAPI environment before compiling dependencies.

```bash
source /opt/intel/oneapi/setvars.sh
```

This exports compiler, MPI, and library paths such that `icx`, `icpx`, and `dpcpp` become available to `pip` builds. Re-run the command whenever you open a new shell.

## 3. Install pytorch

```bash
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu
```
## 4. install packages

```bash
pip install --upgrade setuptools wheel
pip install mmdet numba pyquaternion lyft_dataset_sdk nuscenes-devkit trimes mmengine scikit-image trimesh build
```
## 5. install mmcv packages

```bash
git clone https://github.com/Chuansheng-Liu/mmdetection3d.git -b xpu_bevfusion_porting 
cd mmdetection3d
pip install ./p3.13_packages/mmcv-2.1.0-cp313-cp313-linux_x86_64.whl
```

## 6. Prepare the nuScenes Dataset

Assuming you already downloaded and extracted the dataset to `$FULL_DATASET`, wire it into the new clone at `mmdetection3d`:

```bash
mkdir -p data
ln -s $FULL_DATASET data/nuscenes

# Ensure the directory layout matches nuScenes expectations
ls data/nuscenes
# Expected folders: maps/ samples/ sweeps/ v1.0-trainval/ (or v1.0-mini/)
# If you see an extra nesting level, adjust the symlink target accordingly, e.g.
# ln -s $FULL_DATASET/nuscenes data/nuscenes

# If the test split was downloaded as tarballs, extract it so that v1.0-test exists
tar -xzf data/nuscenes/v1.0-test_meta.tgz -C data/nuscenes
tar -xzf data/nuscenes/v1.0-test_blobs.tgz -C data/nuscenes

# Generate nuScenes metadata (pass v1.0 so the script expands to v1.0-trainval/test)
PYTHONPATH=. python tools/create_data.py nuscenes \
	--root-path data/nuscenes \
	--out-dir data/nuscenes \
	--version v1.0
```


```{note}
If you encounter `AssertionError: Database version not found`, it means `data/nuscenes/<version>` is missing. Revisit the symlink so that the root contains the `v1.0-trainval` (or `v1.0-mini`) folder alongside `samples/`, `maps/`, and `sweeps/`, and make sure you pass `--version v1.0` (not `v1.0-trainval`) before rerunning the command.
If you downloaded the test split as archives, unpack `v1.0-test_meta.tgz` and `v1.0-test_blobs.tgz` first.
```
After the script finishes, verify that files such as `data/nuscenes/nuscenes_infos_train.pkl` are created. Update any config files to point at `data/nuscenes` if you use a custom path.

## 7. Install mmengine/mmdetection3d

```bash
pip install ./p3.13_packages/mmengine-0.10.7-py3-none-any.whl --force-reinstall --no-deps
pip install ./p3.13_packages/mmdet3d-1.4.0-py3-none-any.whl --no-deps --force-reinstall
```

## 8. Run the End2End test
```bash
    SYCL_DEVICE_FILTER=level_zero:gpu \
    python tools/test.py \
    projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
    "${BEVFUSION_MODEL_DIR}/checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth" \
    --cfg-options device=xpu:0
```

## 9. benchmark test
```bash
SYCL_DEVICE_FILTER=level-zero:gpu python tools/analysis_tools/benchmark.py projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py ${BEVFUSION_MODEL_DIR}/check
points/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth --device xpu:0
```