#!/usr/bin/env python3
"""XPU stress tests for BEVFusion custom ops."""

import importlib.util
import os
import sys
import time
from contextlib import contextmanager

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OPS_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "bevfusion", "ops"))


def load_module(name: str, relative_path: str, is_package: bool = False):
    module_path = os.path.join(OPS_ROOT, relative_path)
    search_locations = [os.path.dirname(module_path)] if is_package else None
    spec = importlib.util.spec_from_file_location(
        name, module_path, submodule_search_locations=search_locations)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bev_pool_module = load_module("bev_pool_pkg", os.path.join("bev_pool", "__init__.py"), is_package=True)
voxel_pkg = load_module("voxel_pkg", os.path.join("voxel", "__init__.py"), is_package=True)

bev_pool = bev_pool_module.bev_pool
Voxelization = voxel_pkg.Voxelization
dynamic_scatter = voxel_pkg.dynamic_scatter


@contextmanager
def timer(label: str):
    start = time.time()
    yield
    end = time.time()
    print(f"{label}: {end - start:.3f}s")


def run_bev_pool_tests(device: torch.device):
    configs = [
        (1, 4, 3, 4, 6, 256),
        (2, 6, 4, 6, 8, 2048),
        (3, 8, 6, 8, 12, 4096),
    ]
    for idx, (B, D, H, W, C, N) in enumerate(configs, start=1):
        feats = torch.randn(N, C, device=device, requires_grad=True)
        coords = torch.randint(0, max(B, D, H, W), (N, 4), device=device, dtype=torch.int32)
        coords[:, 0] %= B
        coords[:, 1] %= H
        coords[:, 2] %= D
        coords[:, 3] %= W
        dummy_head = torch.nn.Conv3d(C, C // 2, kernel_size=1).to(device)
        opt = torch.optim.Adam(dummy_head.parameters(), lr=1e-3)
        with timer(f"bev_pool config#{idx}"):
            opt.zero_grad()
            out = bev_pool(feats, coords, B, D, H, W)
            pred = dummy_head(out).mean()
            pred.backward()
            opt.step()
        print(f"  shape -> B{B} D{D} H{H} W{W} C{C} N{N}")


def run_voxel_tests(device: torch.device):
    voxel_scenarios = [
        dict(voxel_size=[0.2, 0.2, 0.4], range=[-10, -10, -3, 10, 10, 1], max_points=5, max_voxels=400),
        dict(voxel_size=[0.1, 0.1, 0.2], range=[-20, -20, -5, 20, 20, 3], max_points=10, max_voxels=800),
        dict(voxel_size=[0.05, 0.05, 0.1], range=[-30, -30, -6, 30, 30, 4], max_points=20, max_voxels=1600),
    ]
    batch_sizes = [2, 3, 4]
    point_counts = [20000, 40000, 60000]
    for idx, (cfg, B, N) in enumerate(zip(voxel_scenarios, batch_sizes, point_counts), start=1):
        layer = Voxelization(
            voxel_size=cfg['voxel_size'],
            point_cloud_range=cfg['range'],
            max_num_points=cfg['max_points'],
            max_voxels=(cfg['max_voxels'], cfg['max_voxels']),
            deterministic=True,
        ).to(device)
        pts = torch.rand(B, N, 5, device=device)
        mins = torch.tensor(cfg['range'][:3], device=device)
        maxs = torch.tensor(cfg['range'][3:], device=device)
        pts[..., :3] = mins + (maxs - mins) * pts[..., :3]
        feats_all, coors_all = [], []
        with timer(f"voxelization config#{idx}"):
            for b in range(B):
                voxels, coors, _ = layer(pts[b])
                feats_all.append(voxels.mean(dim=1))
                coors_all.append(torch.nn.functional.pad(coors, (1, 0), value=b))
        feats = torch.cat(feats_all, dim=0)
        coors = torch.cat(coors_all, dim=0)
        with timer(f"dynamic_scatter config#{idx}"):
            scatter_feats, scatter_coors = dynamic_scatter(feats, coors, 'mean')
            head = torch.nn.Sequential(
                torch.nn.Linear(scatter_feats.shape[1], 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1),
            ).to(device)
            out = head(scatter_feats).sum()
            out.backward()
        print(f"  scatter features -> {scatter_feats.shape}")


def main():
    assert torch.xpu.is_available(), "XPU device not available"
    device = torch.device('xpu')
    torch.manual_seed(123)
    print("Running BEVFusion XPU ops stress suite...")
    run_bev_pool_tests(device)
    run_voxel_tests(device)
    print("All stress tests completed successfully.")


if __name__ == "__main__":
    main()
