#!/usr/bin/env python3
"""Rehydrate a quantized BEVFusion FX graph for evaluation.

Given the original BEVFusion config/checkpoint and the PTQ state_dict dumped by
``tools/quantization/bevfusion_ptq.py``, this helper rebuilds the FX graph,
loads the quantized weights, restores modules kept in FP32, and optionally
exports the ready-to-run GraphModule. This avoids re-running ``prepare_fx``
when you want to inspect or deploy the quantized model.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from contextlib import nullcontext
from typing import Iterable, Sequence

import torch  # type: ignore[import]
from mmengine.config import Config, DictAction  # type: ignore[import]
from mmengine.registry import init_default_scope  # type: ignore[import]
from mmengine.runner import Runner  # type: ignore[import]
from mmengine.runner.checkpoint import _load_checkpoint_to_model  # type: ignore[import]

# Allow running the script without manually exporting PYTHONPATH
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.quantization.bevfusion_ptq import (
    QuantizedBEVFusion,
    _freeze_quantization,
    _pack_batch_for_fx,
    _update_calib_dataset,
)

# TODO(xpu): Remove once quantized::conv2d_relu.new supports XPU execution natively.

RESTORE_ATTRS: Sequence[str] = (
    'pts_voxel_layer',
    'pts_voxel_encoder',
    'pts_middle_encoder',
    'pts_backbone',
    'fusion_layer',
    'img_backbone',
    'img_neck',
    'view_transform',
    'bbox_head',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Load BEVFusion PTQ checkpoint for evaluation.')
    parser.add_argument('--config', required=True, help='Config file path.')
    parser.add_argument('--checkpoint', required=True, help='Original FP32 checkpoint path.')
    parser.add_argument('--quantized-checkpoint', required=True, help='Quantized state dict emitted by bevfusion_ptq.py.')
    parser.add_argument('--calib-ann-file', required=True, help='NuScenes infos file used to trace the model.')
    parser.add_argument('--device', default='cpu', help='Device used while preparing the FX graph (default: cpu).')
    parser.add_argument('--work-dir', default='work_dirs/ptq_eval', help='Temporary work_dir for Runner instantiation.')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for the tracing dataloader.')
    parser.add_argument('--num-workers', type=int, default=2, help='Worker count for the tracing dataloader.')
    parser.add_argument(
        '--skip-modules',
        nargs='*',
        default=(
            'bev_pool',
            'voxel_layer',
            'fusion_layer',
            'img_backbone',
            'img_neck',
            'view_transform',
            'pts_backbone',
            'deblock',
            'deblocks',
        ),
        help='Module name substrings to keep in FP32 (must match the PTQ run).',
    )
    parser.add_argument('--output', help='Optional path to save the restored GraphModule.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override settings in the config. The key-pattern is xxx=yyy.',
    )
    parser.add_argument('--run-eval', action='store_true', help='Run runner.test() with the quantized model before exiting.')
    parser.add_argument('--eval-split', default='val', choices=('val', 'test'), help='Dataloader split to use when --run-eval is set.')
    parser.add_argument('--benchmark-samples', type=int, default=0, help='Number of samples for latency benchmarking (0 disables benchmarking).')
    parser.add_argument('--benchmark-warmup', type=int, default=5, help='Warm-up iterations skipped before measuring latency.')
    parser.add_argument('--benchmark-log-interval', type=int, default=50, help='Logging interval for benchmark progress.')
    return parser.parse_args()


def _load_base_checkpoint(path: str) -> dict:
    safe_allow_list: list[object] = []
    try:
        from mmengine.logging.history_buffer import HistoryBuffer  # type: ignore

        safe_allow_list.append(HistoryBuffer)
    except ImportError:
        pass

    try:
        from torch.serialization import safe_globals  # type: ignore

        safe_ctx = safe_globals(safe_allow_list)
    except (ImportError, AttributeError):
        safe_ctx = nullcontext()

    with safe_ctx:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    return checkpoint


def _restore_fp32_modules(quantized: torch.nn.Module, reference: torch.nn.Module, extra_attrs: Iterable[str] = ()) -> None:
    for attr in RESTORE_ATTRS:
        if hasattr(reference, attr):
            setattr(quantized.model, attr, getattr(reference, attr))
    for attr in extra_attrs:
        if hasattr(reference, attr):
            setattr(quantized.model, attr, getattr(reference, attr))


def _synchronize(device: torch.device) -> None:
    if device.type == 'cuda' and torch.cuda.is_available():  # type: ignore[attr-defined]
        torch.cuda.synchronize()  # type: ignore[no-untyped-call]
    elif device.type == 'xpu' and hasattr(torch, 'xpu') and torch.xpu.is_available():  # type: ignore[attr-defined]
        torch.xpu.synchronize()  # type: ignore[no-untyped-call]


def main() -> None:
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.work_dir = args.work_dir

    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    runner = Runner.from_cfg(cfg)
    checkpoint = _load_base_checkpoint(args.checkpoint)
    runner.call_hook('after_load_checkpoint', checkpoint=checkpoint)
    _load_checkpoint_to_model(runner.model, checkpoint, strict=False)
    runner.model.eval()

    device = torch.device(args.device if ':' in args.device else args.device)

    quant_model = copy.deepcopy(runner.model)
    quant_model.eval().to(device)
    if hasattr(quant_model, 'data_preprocessor'):
        quant_model.data_preprocessor.to(device)  # type: ignore[attr-defined]

    calib_loader_cfg = _update_calib_dataset(cfg, args.calib_ann_file, args.batch_size, args.num_workers)
    calib_loader = runner.build_dataloader(calib_loader_cfg, seed=cfg.get('seed'))

    _freeze_quantization(quant_model, args.skip_modules)

    from torch.ao.quantization import QConfigMapping, get_default_qconfig  # type: ignore
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx  # type: ignore
    from torch.ao.quantization.fx.custom_config import PrepareCustomConfig  # type: ignore

    qconfig = get_default_qconfig(torch.backends.quantized.engine or 'onednn')
    qconfig_mapping = (
        QConfigMapping()
        .set_global(qconfig)
        .set_object_type(torch.nn.modules.conv._ConvTransposeNd, None)
        .set_object_type(torch.nn.ConvTranspose2d, None)
        .set_object_type(torch.ops.aten.conv_transpose2d, None)
        .set_module_name_regex(r'.*deblocks.*', None)
    )

    prepare_config = PrepareCustomConfig().set_non_traceable_module_names([
        'model.img_backbone',
        'model.img_neck',
        'model.img_backbone.patch_embed.adap_padding',
    ])

    example_batch = next(iter(calib_loader))
    data_preprocessor = getattr(quant_model, 'data_preprocessor', None)
    if data_preprocessor is not None:
        processed = data_preprocessor(example_batch, training=False)
    else:
        processed = example_batch
    packed_example = _pack_batch_for_fx(processed, torch.device('cpu'))
    example_inputs = (packed_example,)

    wrapped = QuantizedBEVFusion(quant_model)
    preserved_dp = getattr(wrapped, 'data_preprocessor', None)
    prepared = prepare_fx(
        wrapped,
        qconfig_mapping,
        example_inputs=example_inputs,
        prepare_custom_config=prepare_config,
    )
    if preserved_dp is not None:
        setattr(prepared, 'data_preprocessor', preserved_dp)

    quantized = convert_fx(prepared.cpu())
    state = torch.load(args.quantized_checkpoint, map_location='cpu')
    if 'state_dict' not in state:
        raise KeyError('Quantized checkpoint does not contain a "state_dict" entry.')
    quantized.load_state_dict(state['state_dict'], strict=False)

    _restore_fp32_modules(
        quantized,
        runner.model,
        extra_attrs=('voxelize_reduce', 'data_preprocessor'),
    )

    quantized.eval()
    quantized.model.eval()

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        torch.save({'model': quantized}, args.output)
        print(f'[INFO] Quantized GraphModule exported to {args.output}')
    else:
        print('[INFO] Quantized GraphModule ready in memory (not exported).')

    if args.benchmark_samples > 0:
        if args.eval_split == 'val':
            dataloader_cfg = cfg.val_dataloader
        else:
            dataloader_cfg = cfg.test_dataloader
        benchmark_loader = runner.build_dataloader(dataloader_cfg, seed=cfg.get('seed'))

        runner.model = quantized.model
        runner.model.to(device)
        runner.model.eval()
        if hasattr(runner.model, 'data_preprocessor'):
            runner.model.data_preprocessor.to(device)  # type: ignore[attr-defined]

        warmup = max(args.benchmark_warmup, 0)
        total = min(max(args.benchmark_samples, 0), len(benchmark_loader))
        if total <= warmup:
            warmup = max(total - 1, 0)
        print(f'[INFO] Benchmarking latency on {total} samples (warm-up: {warmup}).')

        pure_inf = 0.0
        measured = 0
        interval = max(args.benchmark_log_interval, 1)

        with torch.no_grad():
            for idx, batch in enumerate(benchmark_loader):
                if idx >= total:
                    break
                _synchronize(device)
                start = time.perf_counter()
                runner.model.test_step(batch)
                _synchronize(device)
                elapsed = time.perf_counter() - start
                if idx >= warmup:
                    pure_inf += elapsed
                    measured += 1
                    if measured % interval == 0:
                        fps = measured / pure_inf if pure_inf > 0 else 0.0
                        print(f'[INFO] Benchmark progress: {idx + 1} samples processed, FPS={fps:.2f}')

        if measured > 0 and pure_inf > 0:
            fps = measured / pure_inf
            print(f'[INFO] Benchmark complete: {measured} samples, FPS={fps:.2f}')
        else:
            print('[WARN] Benchmark skipped; insufficient measured samples after warm-up.')

    if args.run_eval:
        if cfg.get('default_scope'):
            init_default_scope(cfg.default_scope)
        quantized.model.to(device)
        quantized.model.eval()
        if hasattr(quantized.model, 'data_preprocessor'):
            quantized.model.data_preprocessor.to(device)  # type: ignore[attr-defined]
        runner.model = quantized.model
        runner.model.eval()
        if args.eval_split == 'val':
            results = runner.val()
        else:
            results = runner.test()
        print('[INFO] Evaluation metrics:', results)


if __name__ == '__main__':
    main()
