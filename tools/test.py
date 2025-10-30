# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import inspect
import numpy as np
import torch


def _patch_cuda_with_xpu() -> None:
    """Redirect common torch.cuda entry points to torch.xpu when available.

    MMEngine still calls ``torch.cuda`` helpers while initialising the
    distributed environment. On Intel® XPU builds those functions are either
    absent or stubbed, so we mirror them to the XPU equivalents to keep the
    upstream launcher logic working without further changes.
    """

    if not hasattr(torch, 'xpu'):
        return

    xpu = torch.xpu
    cuda = torch.cuda

    def _set_device(index):
        xpu.set_device(index)

    def _device_count():
        return xpu.device_count()

    def _current_device():
        return xpu.current_device()

    def _is_available():
        return xpu.device_count() > 0

    def _empty_cache():
        if hasattr(xpu, 'empty_cache'):
            xpu.empty_cache()

    def _get_device_name(index):
        return xpu.get_device_name(index)

    def _get_device_properties(index):
        return xpu.get_device_properties(index)

    def _get_device_capability(index):
        return xpu.get_device_capability(index)

    def _max_memory_allocated(device=None):
        return xpu.max_memory_allocated(device=device)

    def _max_memory_reserved(device=None):
        return xpu.max_memory_reserved(device=device)

    def _memory_allocated(device=None):
        return xpu.memory_allocated(device=device)

    def _memory_reserved(device=None):
        return xpu.memory_reserved(device=device)

    def _reset_peak_memory_stats(device=None):
        return xpu.reset_peak_memory_stats(device=device)

    cuda.set_device = _set_device  # type: ignore[attr-defined]
    cuda.device_count = _device_count  # type: ignore[attr-defined]
    cuda.current_device = _current_device  # type: ignore[attr-defined]
    cuda.is_available = _is_available  # type: ignore[attr-defined]
    cuda.empty_cache = _empty_cache  # type: ignore[attr-defined]
    cuda.get_device_name = _get_device_name  # type: ignore[attr-defined]
    cuda.get_device_properties = _get_device_properties  # type: ignore[attr-defined]
    cuda.get_device_capability = _get_device_capability  # type: ignore[attr-defined]
    cuda.max_memory_allocated = _max_memory_allocated  # type: ignore[attr-defined]
    cuda.max_memory_reserved = _max_memory_reserved  # type: ignore[attr-defined]
    cuda.memory_allocated = _memory_allocated  # type: ignore[attr-defined]
    cuda.memory_reserved = _memory_reserved  # type: ignore[attr-defined]
    cuda.reset_peak_memory_stats = _reset_peak_memory_stats  # type: ignore[attr-defined]


_patch_cuda_with_xpu()

_orig_torch_load = torch.load


_torch_load_signature = inspect.signature(_orig_torch_load)


def _torch_load_allow_weights(*args, **kwargs):
    if 'weights_only' in _torch_load_signature.parameters:
        kwargs.setdefault('weights_only', False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_allow_weights

from mmengine.config import Config, ConfigDict, DictAction
from mmengine.logging.history_buffer import HistoryBuffer
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

# Allow checkpoints saved with mmengine metadata to load with PyTorch 2.6 safety defaults.
if hasattr(torch.serialization, 'add_safe_globals'):
    safe_globals = [HistoryBuffer]
    reconstruct = getattr(np.core.multiarray, '_reconstruct', None)
    if reconstruct is not None:
        safe_globals.append((reconstruct, 'numpy.core.multiarray._reconstruct'))
    torch.serialization.add_safe_globals(safe_globals)

from mmdet3d.utils import replace_ceph_backend


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet3D test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--ceph', action='store_true', help='Use ceph as data storage backend')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--score-thr', type=float, default=0.1, help='bbox score threshold')
    parser.add_argument(
        '--task',
        type=str,
        choices=[
            'mono_det', 'multi-view_det', 'lidar_det', 'lidar_seg',
            'multi-modality_det'
        ],
        help='Determine the visualization method depending on the task.')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/test.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualization_hook['test_out_dir'] = args.show_dir
        all_task_choices = [
            'mono_det', 'multi-view_det', 'lidar_det', 'lidar_seg',
            'multi-modality_det'
        ]
        assert args.task in all_task_choices, 'You must set '\
            f"'--task' in {all_task_choices} in the command " \
            'if you want to use visualization hook'
        visualization_hook['vis_task'] = args.task
        visualization_hook['score_thr'] = args.score_thr
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)

    # TODO: We will unify the ceph support approach with other OpenMMLab repos
    if args.ceph:
        cfg = replace_ceph_backend(cfg)

    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        # Currently, we only support tta for 3D segmentation
        # TODO: Support tta for 3D detection
        assert 'tta_model' in cfg, 'Cannot find ``tta_model`` in config.'
        assert 'tta_pipeline' in cfg, 'Cannot find ``tta_pipeline`` in config.'
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start testing
    runner.test()


if __name__ == '__main__':
    main()
