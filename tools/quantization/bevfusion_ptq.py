#!/usr/bin/env python3
"""Post-training quantization for BEVFusion using PyTorch FX.

This helper mirrors the CUDA-BEVFusion PTQ workflow but stays entirely inside
the mmdetection3d/Intel XPU environment. The command expects a calibration
subset (e.g. ``data/nuscenes/calib_subset/nuscenes_infos_calib.pkl``) created
from the NuScenes v1.0-trainval tables.

Example:

.. code-block:: bash

    PYTHONPATH="$MMCV_ROOT" python tools/quantization/bevfusion_ptq.py \\
        --config projects/BEVFusion/configs/\
bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \\
        --checkpoint "$BEVFUSION_MODEL_DIR"/checkpoints/\
bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth \\
        --calib-ann-file data/nuscenes/calib_subset/nuscenes_infos_calib.pkl \\
        --output work_dirs/bevfusion_quantized_ptq.pth

ResNet layers, fusion blocks, and image branches are quantized; custom SYCL
extensions (voxelization, bev_pool, scatter) remain in FP32 by default to avoid
accuracy regressions.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from contextlib import nullcontext
from typing import List, Sequence, Tuple

import numpy as np  # type: ignore[import]
import torch  # type: ignore[import]
import torch.nn.functional as F  # type: ignore[import]
from mmengine.config import Config, ConfigDict, DictAction  # type: ignore[import]
from mmengine.registry import init_default_scope  # type: ignore[import]
from mmengine.runner import Runner  # type: ignore[import]
from mmengine.runner.checkpoint import _load_checkpoint_to_model  # type: ignore[import]
from torch.fx import wrap  # type: ignore[import]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='PTQ for BEVFusion (PyTorch FX)')
    parser.add_argument('--config', required=True, help='Config file path.')
    parser.add_argument('--checkpoint', required=True, help='FP32 checkpoint.')
    parser.add_argument(
        '--calib-ann-file',
        required=True,
        help='Calibration NuScenes info file (e.g. data/nuscenes/calib_subset/*.pkl).')
    parser.add_argument(
        '--output',
        default='work_dirs/bevfusion_quantized_ptq.pth',
        help='Output path for the quantized checkpoint.',
    )
    parser.add_argument('--batch-size', type=int, default=1, help='Calibration batch size.')
    parser.add_argument('--num-workers', type=int, default=2, help='Calibration dataloader workers.')
    parser.add_argument('--calibrate-batches', type=int, default=300, help='Number of calibration batches.')
    parser.add_argument(
        '--device',
        default='cpu',
        help='Execution device for calibration/inference (default: cpu).',
    )
    parser.add_argument(
        '--work-dir',
        default='work_dirs/ptq_bevfusion',
        help='Working directory for runner checkpoints/logs.',
    )
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
        help='Module name substrings to keep in FP32.',
    )
    parser.add_argument('--save-observer-stats', action='store_true', help='Dump observer histograms alongside the checkpoint.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override settings in the config. The key-pattern is xxx=yyy.',
    )
    return parser.parse_args()


def _update_calib_dataset(cfg: Config, ann_file: str, batch_size: int, num_workers: int) -> ConfigDict:
    dataloader = cfg.val_dataloader.copy()
    dataset_cfg = dataloader['dataset'].copy()
    data_root = dataset_cfg.get('data_root', '')
    if os.path.isabs(ann_file):
        dataset_cfg['ann_file'] = ann_file
        if data_root:
            dataset_cfg['data_root'] = ''
    elif data_root and ann_file.startswith(data_root):
        dataset_cfg['ann_file'] = os.path.relpath(ann_file, data_root)
    else:
        dataset_cfg['ann_file'] = ann_file
    dataset_cfg.setdefault('test_mode', True)
    dataloader['dataset'] = dataset_cfg
    dataloader['batch_size'] = batch_size
    dataloader['num_workers'] = num_workers
    dataloader.setdefault('persistent_workers', False)
    # Disable sampler shuffling for deterministic calibration order.
    if 'sampler' in dataloader:
        sampler_cfg = dataloader['sampler'].copy()
        sampler_cfg['shuffle'] = False
        dataloader['sampler'] = sampler_cfg
    else:
        dataloader['shuffle'] = False
    return ConfigDict(dataloader)

class QuantizedBEVFusion(torch.nn.Module):
    """FX-friendly wrapper that runs BEVFusion in inference mode."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        if hasattr(model, 'data_preprocessor'):
            self.data_preprocessor = model.data_preprocessor  # type: ignore[assignment]

    def forward(self, batch):  # type: ignore[override]
        points = batch.get('points')
        pts_feature = _run_extract_pts_feat(self.model, {'points': points})
        fused = pts_feature

        fused = self.model.pts_backbone(fused)
        fused = self.model.pts_neck(fused)

        if self.model.with_bbox_head:
            head_input = fused if isinstance(fused, list) else [fused]
            metas = batch.get('metas', [])
            _run_bbox_head(self.model.bbox_head, head_input, metas)

        return fused


def _pack_batch_for_fx(processed_batch: dict, device: torch.device) -> dict:
    inputs = processed_batch['inputs']
    data_samples = processed_batch.get('data_samples', [])
    packed = dict(inputs)

    if not data_samples:
        packed.setdefault('lidar2img', torch.empty(0, device=device))
        packed.setdefault('cam2img', torch.empty(0, device=device))
        packed.setdefault('cam2lidar', torch.empty(0, device=device))
        packed.setdefault('img_aug_matrix', torch.empty(0, device=device))
        packed.setdefault('lidar_aug_matrix', torch.empty(0, device=device))
        return packed

    metas = [sample.metainfo for sample in data_samples]

    def _stack(key: str, default: np.ndarray | None = None) -> torch.Tensor:
        reference = metas[0].get(key, default)
        if reference is None:
            raise KeyError(f'Missing required meta key: {key}')
        values = []
        for meta in metas:
            value = meta.get(key, reference)
            values.append(np.asarray(value, dtype=np.float32))
        return torch.as_tensor(np.stack(values, axis=0), device=device)

    if 'imgs' in packed:
        img_device = packed['imgs'].device
    elif packed.get('points'):
        img_device = packed['points'][0].device
    else:
        img_device = device

    packed['lidar2img'] = _stack('lidar2img')
    packed['cam2img'] = _stack('cam2img')
    packed['cam2lidar'] = _stack('cam2lidar')
    packed['img_aug_matrix'] = _stack('img_aug_matrix')
    identity = np.eye(4, dtype=np.float32)
    packed['lidar_aug_matrix'] = _stack('lidar_aug_matrix', identity)
    packed['metas'] = metas

    if 'imgs' in packed and packed['imgs'].device != img_device:
        packed['imgs'] = packed['imgs'].to(img_device)

    return packed


def _run_extract_pts_feat(model: torch.nn.Module, batch_inputs: dict) -> torch.Tensor:
    return model.extract_pts_feat(batch_inputs)


wrap('_run_extract_pts_feat')


def _run_bbox_head(head: torch.nn.Module, feats, metas):
    return head(feats, metas)


wrap('_run_bbox_head')


def _freeze_quantization(model: torch.nn.Module, skip_tokens: Sequence[str]) -> None:
    matched: List[str] = []
    for name, module in model.named_modules():
        if any(token in name for token in skip_tokens):
            module.qconfig = None  # type: ignore[attr-defined]
            matched.append(name)
    if matched:
        print(f'[INFO] Skipping quantization for modules: {matched}')


def _clear_qconfig_for_conv_transpose(gm: torch.fx.GraphModule) -> Tuple[List[str], List[str]]:
    cleared_modules: List[str] = []
    cleared_nodes: List[str] = []
    for node in gm.graph.nodes:
        if node.op == 'call_module':
            module = gm.get_submodule(node.target)
            if isinstance(module, torch.nn.modules.conv._ConvTransposeNd):
                module.qconfig = None  # type: ignore[attr-defined]
                if 'qconfig' in node.meta:
                    node.meta['qconfig'] = None
                cleared_modules.append(node.target)
        elif node.op == 'call_function':
            target = node.target
            if target is F.conv_transpose2d or target is torch.ops.aten.conv_transpose2d:
                if 'qconfig' in node.meta:
                    node.meta['qconfig'] = None
                cleared_nodes.append(str(target))
    return cleared_modules, cleared_nodes


def main() -> None:
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.work_dir = args.work_dir

    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    if not torch.backends.quantized.engine:
        # Prefer oneDNN backend where available (Intel® GPUs / CPUs).
        torch.backends.quantized.engine = 'onednn'

    if args.device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(args.device if ':' in args.device else f'{args.device}:0')

    # PyTorch 2.6+ blocks loading pickled states unless types are explicitly allow-listed.
    calib_dataloader_cfg = _update_calib_dataset(cfg, args.calib_ann_file, args.batch_size, args.num_workers)

    runner = Runner.from_cfg(cfg)
    safe_allow_list: List[object] = []
    try:
        from mmengine.logging.history_buffer import HistoryBuffer  # type: ignore[import]

        safe_allow_list.append(HistoryBuffer)
    except ImportError:  # pragma: no cover - optional dependency
        pass

    safe_allow_list.extend([
        np.ndarray,
        np.dtype,
        np.core.multiarray._reconstruct,  # type: ignore[attr-defined]
    ])

    try:
        from torch.serialization import safe_globals  # type: ignore[import]

        safe_ctx = safe_globals(safe_allow_list)
    except (ImportError, AttributeError):  # pragma: no cover - PyTorch < 2.6
        safe_ctx = nullcontext()

    with safe_ctx:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    runner.call_hook('after_load_checkpoint', checkpoint=checkpoint)
    _load_checkpoint_to_model(runner.model, checkpoint, strict=False)
    runner.logger.info(f'Loaded checkpoint from {args.checkpoint}')
    fp32_model = runner.model
    fp32_model.eval()

    quant_model = copy.deepcopy(fp32_model)
    quant_model.eval().to(device)
    if hasattr(quant_model, 'data_preprocessor'):
        quant_model.data_preprocessor.to(device)  # type: ignore[call-arg]

    calib_loader = runner.build_dataloader(
        calib_dataloader_cfg,
        seed=cfg.get('seed'),
    )

    _freeze_quantization(quant_model, args.skip_modules)

    from torch.ao.quantization import (  # type: ignore[import]
        QConfigMapping,
        get_default_qconfig,
    )
    try:
        from torch.ao.quantization.quantize_fx import (  # type: ignore[import]
            convert_fx,
            prepare_fx,
        )
    except ImportError:  # pragma: no cover - compatibility fallback
        try:
            from torch.ao.quantization.fx import (  # type: ignore[import]
                convert_fx,
                prepare_fx,
            )
        except ImportError:
            from torch.ao.quantization import (  # type: ignore[import]
                convert_fx,
                prepare_fx,
            )

    from torch.ao.quantization.fx.custom_config import (  # type: ignore[import]
        PrepareCustomConfig,
    )
    qconfig = get_default_qconfig(torch.backends.quantized.engine)
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
    packed_example = _pack_batch_for_fx(processed, device)
    example_inputs = (packed_example,)

    wrapped_model = QuantizedBEVFusion(quant_model)
    data_preprocessor = getattr(wrapped_model, 'data_preprocessor', None)
    prepared_model = prepare_fx(
        wrapped_model,
        qconfig_mapping,
        example_inputs=example_inputs,
        prepare_custom_config=prepare_config,
    )
    if data_preprocessor is not None:
        setattr(prepared_model, 'data_preprocessor', data_preprocessor)

    processed = None
    with torch.no_grad():
        for idx, batch in enumerate(calib_loader):
            if idx >= args.calibrate_batches:
                break
            if data_preprocessor is not None:
                processed = data_preprocessor(batch, training=False)
            else:
                processed = batch
            packed = _pack_batch_for_fx(processed, device)
            prepared_model(packed)

    prepared_model_cpu = prepared_model.cpu()
    cleared_mods, cleared_nodes = _clear_qconfig_for_conv_transpose(prepared_model_cpu)
    if cleared_mods or cleared_nodes:
        print('[INFO] Cleared qconfig for transposed conv modules:', cleared_mods)
        print('[INFO] Cleared qconfig for transposed conv function nodes:', cleared_nodes)
    quantized_model = convert_fx(prepared_model_cpu)

    state_dict = quantized_model.state_dict()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({'state_dict': state_dict}, args.output)
    print(f'[INFO] Quantized checkpoint saved to {args.output}')

    if args.save_observer_stats and hasattr(prepared_model, 'activation_post_process_map'):  # type: ignore[attr-defined]
        stats_path = os.path.splitext(args.output)[0] + '_observers.json'
        stats = {
            name: obs.calculate_qparams()  # type: ignore[attr-defined]
            for name, obs in prepared_model.activation_post_process_map.items()  # type: ignore[attr-defined]
        }
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, default=lambda x: x.tolist())
        print(f'[INFO] Observer statistics dumped to {stats_path}')


if __name__ == '__main__':
    main()