import os
from setuptools import setup

import torch
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)


def make_backend_ext(name,
                     module,
                     sources,
                     sources_cuda=None,
                     sources_xpu=None,
                     extra_args=None,
                     extra_include_path=None,
                     extra_link_args=None):

    sources_cuda = sources_cuda or []
    sources_xpu = sources_xpu or []
    extra_args = extra_args or []
    extra_include_path = extra_include_path or []
    extra_link_args = extra_link_args or []

    define_macros = []
    extra_compile_args = {'cxx': list(extra_args)}
    selected_sources = list(sources)

    force_cuda = os.getenv('FORCE_CUDA', '0') == '1'
    force_xpu = os.getenv('FORCE_XPU', '0') == '1'
    disable_xpu = os.getenv('DISABLE_XPU', '0') == '1'

    has_cuda = torch.cuda.is_available() or force_cuda
    xpu_runtime_available = hasattr(torch, 'xpu') and torch.xpu.is_available()
    has_xpu = (force_xpu or xpu_runtime_available) and not disable_xpu

    if has_cuda:
        define_macros.append(('WITH_CUDA', None))
        extension = CUDAExtension
        extra_compile_args['nvcc'] = list(extra_args) + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
            '-gencode=arch=compute_70,code=sm_70',
            '-gencode=arch=compute_75,code=sm_75',
            '-gencode=arch=compute_80,code=sm_80',
            '-gencode=arch=compute_86,code=sm_86',
        ]
        selected_sources += sources_cuda
    elif has_xpu:
        define_macros.append(('WITH_XPU', None))
        extension = CppExtension
        extra_compile_args['cxx'] += ['-fsycl', '-fsycl-unnamed-lambda']
        selected_sources += sources_xpu
        extra_link_args.append('-fsycl')
        print(f'Compiling {name} with XPU backend')
    else:
        extension = CppExtension
        print(f'Compiling {name} without GPU acceleration')

    return extension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in selected_sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )


if __name__ == '__main__':
    setup(
        name='bev_pool',
        ext_modules=[
            make_backend_ext(
                name='bev_pool_ext',
                module='projects.BEVFusion.bevfusion.ops.bev_pool',
                sources=[
                    'src/bev_pool.cpp',
                ],
                sources_cuda=['src/bev_pool_cuda.cu'],
                sources_xpu=['src/bev_pool_xpu.cpp'],
            ),
            make_backend_ext(
                name='voxel_layer',
                module='projects.BEVFusion.bevfusion.ops.voxel',
                sources=[
                    'src/voxelization.cpp',
                    'src/scatter_points_cpu.cpp',
                    'src/voxelization_cpu.cpp',
                ],
                sources_cuda=[
                    'src/scatter_points_cuda.cu',
                    'src/voxelization_cuda.cu',
                ],
                sources_xpu=[
                    'src/voxelization_xpu.cpp',
                    'src/scatter_points_xpu.cpp',
                ],
            ),
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False,
    )
