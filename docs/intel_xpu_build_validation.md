# Intel XPU Build and Validation Guide

## Fetch Latest Sources

```bash
# mmdetection3d
git clone https://github.com/Chuansheng-Liu/mmdetection3d.git -b xpu_bevfusion_porting

# mmcv
git clone https://github.com/Chuansheng-Liu/mmcv.git -b xpu_ops_porting
```

## Prepare Environment

Intel oneAPI and the PyTorch XPU environment must be active before any build or test steps. The oneAPI bundle only ships `setvars.sh`, so we call it directly.

```bash
source /opt/intel/oneapi/setvars.sh --force
source ~/chuansheng/pytorch_xpu/bin/activate
```

## Clean Previous Builds(optional)

Start from a pristine tree so the rebuild picks up the latest kernels.

```bash
# mmcv artifacts
cd ~/chuansheng/mmcv
rm -rf build mmcv.egg-info mmcv/_ext.cpython-*.so

# mmdetection3d artifacts (including BEVFusion custom build dir)
cd ~/chuansheng/mmdetection3d
rm -rf build mmdet3d.egg-info projects/BEVFusion/build
```

## Build mmcv (XPU backend)

Running the editable install compiles the C++/SYCL ops (`mmcv/_ext*.so`) against the active PyTorch XPU toolchain.

```bash
cd ~/chuansheng/mmcv
CC=icx CXX=icpx python -m pip install -e . --no-build-isolation
# Disable isolation and point CC/CXX at the oneAPI toolchain so the SYCL ops compile against torch XPU.
# The setup hook copies the compiled `_ext` library back into mmcv/ for development use.
```

## Package mmcv for deployment(standalone usage)

When you need to distribute mmcv without relying on an editable checkout, build a wheel with the oneAPI toolchain and install it like any other package.

```bash
cd ~/chuansheng/mmcv
rm -rf build mmcv.egg-info mmcv/_ext.cpython-*.so dist
CC=icx CXX=icpx python -m build --wheel
# Wheel lands in dist/mmcv-<version>-py2.py3-none-any.whl
python -m pip install dist/mmcv-<version>-py2.py3-none-any.whl
```

The wheel ships the full package tree (including `mmcv/__init__.py` and `version.py`), so downstream environments pick up `mmcv.__version__` without manual PYTHONPATH tweaks.

## Build mmdetection3d

`open3d` wheels are still missing for Python 3.13, so leave the dependency graph as-is and reuse the pre-installed torch stack. Disable build isolation to ensure torch stays visible.

```bash
cd ~/chuansheng/mmdetection3d
FORCE_XPU=1 CC=icx CXX=icpx python -m pip install -e . --no-build-isolation --no-deps
# FORCE_XPU enables XPU-specific kernels while icx/icpx ensure oneAPI compilers are used.
FORCE_XPU=1 CC=icx CXX=icpx python projects/BEVFusion/setup.py build_ext --inplace
# Compile the BEVFusion custom op immediately so SYCL mistakes fail the build instead of lingering until runtime.
```

## Package mmdetection3d for deployment(standalone usage)

To ship mmdetection3d without depending on the editable checkout, build a wheel that bakes in the XPU custom ops. The build backend must run inside the pre-configured XPU environment, so we disable isolation and force the oneAPI toolchain.

```bash
cd ~/chuansheng/mmdetection3d
rm -rf build mmdet3d.egg-info projects/BEVFusion/build dist
FORCE_XPU=1 CC=icx CXX=icpx python -m build --wheel --no-isolation
# Wheel lands in dist/mmdet3d-<version>-py3-none-any.whl
python -m pip install dist/mmdet3d-<version>-py3-none-any.whl --no-deps
# --no-deps keeps the existing torch/mmcv stack, which already targets Intel XPU.
```

## Smoke Tests

The mmcv installation needs a CPU voxelization sanity check, and mmdetection3d must import with mmcv’s Python shim in scope. For XPU we rely on the BEVFusion stress harness (since upstream MMCV does not ship XPU voxelization kernels).

```bash
python -m pytest tests/test_utils/test_setup_env.py -q
python -c "import mmcv, mmdet3d; print(mmcv.__version__, mmdet3d.__version__)"

# mmcv voxelization on CPU
cd ~/chuansheng/mmcv
python -m pytest tests/test_ops/test_voxelization.py -k cpu --maxfail=1 --disable-warnings -q

# mmdetection3d environment setup test
cd ~/chuansheng/mmdetection3d
PYTHONPATH=~/chuansheng/mmcv python -m pytest tests/test_utils/test_setup_env.py -q
# Drop the PYTHONPATH export only if mmcv is installed from the wheel; editable installs still need it.

# Quick import verification (optional)
PYTHONPATH=~/chuansheng/mmcv python -c "import mmcv, mmdet3d; print(mmcv.__version__, mmdet3d.__version__)"

# BEVFusion custom kernels on XPU (development checkout)
cd ~/chuansheng/mmdetection3d
PYTHONPATH=~/chuansheng/mmcv python projects/BEVFusion/tests/xpu_ops_stress.py
```

## End-to-End Sanity
[BEVFusion test commands](https://github.com/Chuansheng-Liu/mmdetection3d/blob/xpu_bevfusion_porting/projects/BEVFusion/README.md#testing-commands)

The full NuScenes evaluation is long; to confirm the loop works on XPU, watch for two progress lines and then Ctrl+C.

```bash
cd ~/chuansheng/mmdetection3d
PYTHONPATH=/home/intel/chuansheng/mmcv:/home/intel/chuansheng/mmdetection3d:${PYTHONPATH} \
    SYCL_DEVICE_FILTER=level_zero:gpu \
    python tools/test.py \
    projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
    /home/intel/chuansheng/bevfusion_model_data/checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth \
    --cfg-options device=xpu:0
# Stop once you see two "mmengine - INFO - Epoch(test)" lines.
```

