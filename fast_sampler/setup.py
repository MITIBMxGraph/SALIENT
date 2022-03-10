import os
import sys
from pathlib import Path
from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CppExtension
from torch.__config__ import parallel_info


def flags_to_list(flagstring):
    return list(filter(bool, flagstring.split(' ')))


WITH_SYMBOLS = True if os.getenv('WITH_SYMBOLS', '0') == '1' else False
CXX_FLAGS = flags_to_list(os.getenv('CXX_FLAGS', ''))
ROOT_PATH = Path(__file__).resolve().parent


def get_extensions():
    define_macros = []
    libraries = []
    extra_compile_args = {
        'cxx': ['-O3', '-mcpu=native', '-std=c++17', '-g'] + CXX_FLAGS}
    extra_link_args = [] if WITH_SYMBOLS else ['-s']

    info = parallel_info()
    if 'backend: OpenMP' in info and 'OpenMP not found' not in info:
        extra_compile_args['cxx'] += ['-DAT_PARALLEL_OPENMP']
        if sys.platform == 'win32':
            extra_compile_args['cxx'] += ['/openmp']
        else:
            extra_compile_args['cxx'] += ['-fopenmp']
    else:
        print('Compiling without OpenMP...')

    include_dirs = [ROOT_PATH, ROOT_PATH.joinpath('parallel-hashmap')]

    return [
        CppExtension(
            'fast_sampler',
            ['fast_sampler.cpp'],
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            libraries=libraries,
        ),
    ]


setup(
    name='fast_sampler',
    ext_modules=get_extensions(),
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=False)
    })
