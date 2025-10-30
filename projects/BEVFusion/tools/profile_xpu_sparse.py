#!/usr/bin/env python3

"""Profile BEVFusion sparse ops on Intel® XPU hardware.

This helper runs a short single-device evaluation loop under ``torch.profiler``
so we can inspect host/device hotspots (e.g. CPU fallbacks inside
``mmcv.ops.sparse_ops.indice_conv``). It mirrors the launcher logic from
``tools/test.py`` but swaps in a custom test loop that limits the number of
iterations and records a trace to TensorBoard-compatible JSON files.
"""

from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch

from mmengine.config import Config, DictAction
from mmengine.logging.history_buffer import HistoryBuffer
from mmengine.registry import LOOPS
from mmengine.runner import Runner
from mmengine.runner.loops import TestLoop, _parse_losses

from mmdet3d.utils import replace_ceph_backend


def _patch_cuda_with_xpu() -> None:
    """Mirror common ``torch.cuda`` helpers onto ``torch.xpu`` when available."""

    if not hasattr(torch, 'xpu'):
        return

    xpu = torch.xpu
    cuda = torch.cuda

    cuda.set_device = lambda index: xpu.set_device(index)  # type: ignore[attr-defined]
    cuda.device_count = lambda: xpu.device_count()  # type: ignore[attr-defined]
    cuda.current_device = lambda: xpu.current_device()  # type: ignore[attr-defined]
    cuda.is_available = lambda: xpu.device_count() > 0  # type: ignore[attr-defined]
    cuda.empty_cache = lambda: getattr(xpu, 'empty_cache', lambda: None)()  # type: ignore[attr-defined]
    cuda.get_device_name = lambda index: xpu.get_device_name(index)  # type: ignore[attr-defined]
    cuda.get_device_properties = lambda index: xpu.get_device_properties(index)  # type: ignore[attr-defined]
    cuda.get_device_capability = lambda index: xpu.get_device_capability(index)  # type: ignore[attr-defined]
    cuda.max_memory_allocated = lambda device=None: xpu.max_memory_allocated(device=device)  # type: ignore[attr-defined]
    cuda.max_memory_reserved = lambda device=None: xpu.max_memory_reserved(device=device)  # type: ignore[attr-defined]
    cuda.memory_allocated = lambda device=None: xpu.memory_allocated(device=device)  # type: ignore[attr-defined]
    cuda.memory_reserved = lambda device=None: xpu.memory_reserved(device=device)  # type: ignore[attr-defined]
    cuda.reset_peak_memory_stats = (  # type: ignore[attr-defined]
        lambda device=None: xpu.reset_peak_memory_stats(device=device))


_patch_cuda_with_xpu()

_ORIG_TORCH_LOAD = torch.load
_TORCH_LOAD_SIGNATURE = inspect.signature(_ORIG_TORCH_LOAD)


def _torch_load_allow_weights(*args, **kwargs):
    if 'weights_only' in _TORCH_LOAD_SIGNATURE.parameters:
        kwargs.setdefault('weights_only', False)
    return _ORIG_TORCH_LOAD(*args, **kwargs)


torch.load = _torch_load_allow_weights

if hasattr(torch.serialization, 'add_safe_globals'):
    safe_globals = [HistoryBuffer]
    reconstruct = getattr(np.core.multiarray, '_reconstruct', None)
    if reconstruct is not None:
        safe_globals.append((reconstruct, 'numpy.core.multiarray._reconstruct'))
    torch.serialization.add_safe_globals(safe_globals)


def _ensure_tensorboard_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _normalise_activities(names: Iterable[str]) -> List[torch.profiler.ProfilerActivity]:
    from torch.profiler import ProfilerActivity

    activities: List[torch.profiler.ProfilerActivity] = []
    for name in names:
        lower = name.lower()
        if lower == 'cpu':
            activities.append(ProfilerActivity.CPU)
        elif lower in {'cuda', 'xpu'}:
            attr = getattr(ProfilerActivity, lower.upper(), None)
            if attr is None:
                raise RuntimeError(
                    f'ProfilerActivity.{lower.upper()} unavailable in this PyTorch build')
            activities.append(attr)
        else:
            raise ValueError(f'Unsupported profiler activity: {name}')
    if not activities:
        activities.append(ProfilerActivity.CPU)
    return activities


@LOOPS.register_module()
class ProfileTestLoop(TestLoop):
    """A test loop that records a short ``torch.profiler`` trace."""

    def __init__(self,
                 runner,
                 dataloader,
                 evaluator,
                 max_iters: int = 50,
                 profiler: dict | None = None,
                 fp16: bool = False):
        super().__init__(runner, dataloader, evaluator, fp16=fp16)
        self.max_iters = max(1, int(max_iters))
        self.profiler_cfg = profiler or {}
        self._processed = 0

    def _make_profiler(self):
        from torch.profiler import (ProfilerActivity, profile, schedule,
                                    tensorboard_trace_handler)

        activities = _normalise_activities(
            self.profiler_cfg.get('activities', ('cpu',)))
        wait = int(self.profiler_cfg.get('wait', 1))
        warmup = int(self.profiler_cfg.get('warmup', 1))
        active = int(self.profiler_cfg.get('active', self.max_iters))
        repeat = int(self.profiler_cfg.get('repeat', 1))
        sched = schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)

        log_dir = _ensure_tensorboard_dir(
            self.profiler_cfg.get('log_dir', 'work_dirs/xpu_profile'))
        handler = tensorboard_trace_handler(str(log_dir))

        return profile(
            activities=activities,
            schedule=sched,
            record_shapes=bool(self.profiler_cfg.get('record_shapes', True)),
            profile_memory=bool(self.profiler_cfg.get('profile_memory', True)),
            with_stack=bool(self.profiler_cfg.get('with_stack', False)),
            on_trace_ready=handler)

    def run(self):  # type: ignore[override]
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()

        self.test_loss.clear()
        prof = self._make_profiler()
        self._processed = 0
        processed_batches = 0

        prof.__enter__()
        try:
            for idx, data_batch in enumerate(self.dataloader):
                if idx >= self.max_iters:
                    break
                self.run_iter(idx, data_batch)
                prof.step()
                processed_batches = idx + 1
                batch_size = len(data_batch.get('data_samples', []))
                self._processed += batch_size
        finally:
            prof.__exit__(None, None, None)

        metrics: dict = {}
        if self.test_loss:
            loss_dict = _parse_losses(self.test_loss, 'test')
            metrics.update(loss_dict)

        metrics.update(
            profiled_batches=float(processed_batches),
            profiled_samples=float(self._processed))

        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')
        return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description='Profile BEVFusion sparse encoder on Intel XPU')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument(
        '--work-dir',
        default='work_dirs/xpu_profile_run',
        help='work directory for runner state and logs')
    parser.add_argument(
        '--log-dir',
        default='work_dirs/xpu_profile/trace',
        help='TensorBoard trace output directory')
    parser.add_argument(
        '--max-iters',
        type=int,
        default=64,
        help='number of test iterations to profile')
    parser.add_argument(
        '--sweeps-num',
        type=int,
        default=3,
        help='override LoadPointsFromMultiSweeps.sweeps_num for quicker runs')
    parser.add_argument(
        '--activities',
        default='cpu',
        help='comma separated profiler activities (cpu,xpu)')
    parser.add_argument(
        '--device',
        default='xpu:0',
        help='device string recognised by torch (e.g. xpu:0)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override config options, key=value')
    parser.add_argument(
        '--wait',
        type=int,
        default=1,
        help='profiler schedule wait steps')
    parser.add_argument(
        '--warmup',
        type=int,
        default=1,
        help='profiler schedule warmup steps')
    parser.add_argument(
        '--active',
        type=int,
        default=0,
        help='profiler schedule active steps (defaults to --max-iters)')
    return parser.parse_args()


def _override_sweeps(pipeline, sweeps_num: int) -> None:
    for step in pipeline:
        if isinstance(step, dict):
            if step.get('type') == 'LoadPointsFromMultiSweeps':
                step['sweeps_num'] = sweeps_num


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.work_dir = args.work_dir
    cfg.launcher = 'none'
    cfg.load_from = args.checkpoint
    cfg.setdefault('default_scope', 'mmdet3d')
    cfg.setdefault('env_cfg', dict())
    cfg.env_cfg.setdefault('dist_cfg', dict(backend='nccl'))

    try:
        replace_ceph_backend(cfg)
    except SyntaxError:
        # Config prettifier can struggle after runtime mutations; skip in that case.
        pass

    # lighten the point aggregation stage to speed up profiling runs
    test_dataset = cfg.test_dataloader.dataset
    _override_sweeps(test_dataset.pipeline, args.sweeps_num)
    cfg.test_dataloader.num_batch_per_epoch = args.max_iters

    activities = [name.strip() for name in args.activities.split(',') if name.strip()]
    active = args.active if args.active > 0 else args.max_iters

    cfg.test_cfg = dict(
        type='ProfileTestLoop',
        max_iters=args.max_iters,
        profiler=dict(
            log_dir=args.log_dir,
            activities=activities,
            wait=args.wait,
            warmup=args.warmup,
            active=active,
            profile_memory=True,
            record_shapes=True,
            with_stack=False))

    cfg.device = args.device

    os.makedirs(cfg.work_dir, exist_ok=True)

    runner = Runner.from_cfg(cfg)
    runner.test()


if __name__ == '__main__':
    main()
