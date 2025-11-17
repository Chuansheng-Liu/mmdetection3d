"""Benchmark script with local path bootstrapping for direct execution."""

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from contextlib import nullcontext
import os
import sys
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
TOOLS_DIR = os.path.join(PROJECT_ROOT, 'tools')

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import torch
from mmengine import Config
from mmengine.config import DictAction
from mmengine.device import get_device
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, autocast, load_checkpoint

from mmdet3d.registry import MODELS

try:
    from tools.misc.fuse_conv_bn import fuse_module
except ModuleNotFoundError:
    from misc.fuse_conv_bn import fuse_module

_MMENGINE_AUTOD_TYPES = {'cuda', 'cpu', 'mlu', 'npu', 'musa'}


def _ensure_torch_device(device_like):
    if isinstance(device_like, torch.device):
        return device_like
    if isinstance(device_like, str):
        return torch.device(device_like)
    raise TypeError(f'Unsupported device type: {type(device_like)}')


def _resolve_amp_dtype(device_like):
    torch_device = _ensure_torch_device(device_like)
    if torch_device.type in ('xpu', 'cpu'):
        return torch.bfloat16
    if torch_device.type == 'cuda':
        return torch.float16
    return torch.float16


def _autocast_context(device_like, dtype, enabled):
    if not enabled:
        return nullcontext()
    torch_device = _ensure_torch_device(device_like)
    device_type = torch_device.type
    if device_type in _MMENGINE_AUTOD_TYPES:
        return autocast(device_type=device_type, dtype=dtype, enabled=True)
    return torch.autocast(device_type=device_type, dtype=dtype)


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--samples',
        type=int,
        default=2000,
        help='Number of inference iterations (after warmup) to benchmark.',
    )
    parser.add_argument(
        '--num-warmup',
        type=int,
        default=5,
        help='Warm-up iterations to skip before timing.',
    )
    parser.add_argument(
        '--log-interval',
        type=int,
        default=50,
        help='Interval of logging progress.',
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        help='Whether to use automatic mixed precision inference')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--device',
        default=None,
        help='Device to benchmark on, e.g. "cuda:0" or "xpu:0". Defaults to auto-detection.',
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override settings in the config (key=value).',
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    init_default_scope('mmdet3d')

    # build config and set cudnn_benchmark
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.samples <= args.num_warmup:
        raise ValueError('--samples must be greater than --num-warmup.')

    if cfg.env_cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # build dataloader
    dataloader = Runner.build_dataloader(cfg.test_dataloader)

    # build model and load checkpoint
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_module(model)
    raw_device = args.device if args.device is not None else get_device()
    device = _ensure_torch_device(raw_device)
    amp_dtype = _resolve_amp_dtype(device)
    model.to(device)
    model.eval()

    # the first several iterations may be very slow so skip them
    pure_inf_time = 0.0
    processed = 0

    def _synchronize(dev) -> None:
        torch_device = _ensure_torch_device(dev)
        if torch_device.type == 'cuda' and hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch_device.type == 'xpu' and hasattr(torch, 'xpu'):
            torch.xpu.synchronize()
        elif torch_device.type == 'mps' and hasattr(torch, 'mps'):
            torch.mps.synchronize()

    with torch.inference_mode():
        for i, data in enumerate(dataloader):
            if i >= args.samples:
                break

            _synchronize(device)
            start_time = time.perf_counter()

            with _autocast_context(device, amp_dtype, args.amp):
                model.test_step(data)

            _synchronize(device)
            elapsed = time.perf_counter() - start_time

            if i >= args.num_warmup:
                pure_inf_time += elapsed
                processed = i + 1 - args.num_warmup
                should_log = processed > 0 and (
                    processed % args.log_interval == 0 or i + 1 == args.samples)
                if should_log:
                    fps = processed / pure_inf_time
                    print(
                        f'Done sample [{i + 1:<3}/{args.samples}], '
                        f'fps: {fps:.2f} sample/s')

    if processed <= 0:
        print('Warning: not enough samples processed after warm-up to compute FPS.')
    else:
        fps = processed / pure_inf_time
        print(f'Overall fps: {fps:.2f} sample/s over {processed} samples (warm-up skipped).')


if __name__ == '__main__':
    main()
