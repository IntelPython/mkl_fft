project(
    'mkl_fft',
    ['c', 'cython'],
    version: run_command(
        'python', '-c',
        'import os; exec(open("mkl_fft/_version.py").read()); print(__version__)',
        check: true
    ).stdout().strip(),
    default_options: [
        'buildtype=release',
    ]
)

py = import('python').find_installation(pure: false)

# numpy includes
np_dir = run_command(py,
    ['-c', 'import numpy; print(numpy.get_include())'],
    check: true
).stdout().strip()

inc_np = include_directories(np_dir, 'mkl_fft/src', 'mkl_fft')

# compiler/linker
cc = meson.get_compiler('c')
c_args = ['-DNDEBUG']

mkl_dep = dependency('MKL', method: 'cmake',
    modules: ['MKL::MKL'],
    cmake_args: [
        '-DMKL_ARCH=intel64',
        '-DMKL_LINK=dynamic',
        '-DMKL_THREADING=intel_thread',
        '-DMKL_INTERFACE=lp64'
    ],
    required: true
)

# code gen
src_file_cli = files('_vendored/process_src_template.py')

gen_mklfft_c = custom_target(
    'mklfft_c_generated',
    output: 'mklfft.c',
    input: 'mkl_fft/src/mklfft.c.src',
    command: [py, src_file_cli, '@INPUT@', '-o', '@OUTPUT@'],
)

# cython
py.extension_module(
    '_pydfti',
    sources: [
        'mkl_fft/_pydfti.pyx',
        gen_mklfft_c
    ],
    include_directories: inc_np,
    dependencies: [mkl_dep],
    c_args: c_args + [
        '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
        '-DPY_ARRAY_UNIQUE_SYMBOL=mkl_fft_ext'
    ],
    install: true,
    subdir: 'mkl_fft'
)

# python
py.install_sources(
    [
        'mkl_fft/__init__.py',
        'mkl_fft/_fft_utils.py',
        'mkl_fft/_init_helper.py',
        'mkl_fft/_mkl_fft.py',
        'mkl_fft/_patch_numpy.py',
        'mkl_fft/_version.py',
    ],
    subdir: 'mkl_fft'
)

py.install_sources(
    [
        'mkl_fft/interfaces/__init__.py',
        'mkl_fft/interfaces/_float_utils.py',
        'mkl_fft/interfaces/_numpy_fft.py',
        'mkl_fft/interfaces/_numpy_helper.py',
        'mkl_fft/interfaces/_scipy_fft.py',
        'mkl_fft/interfaces/numpy_fft.py',
        'mkl_fft/interfaces/scipy_fft.py',
    ],
    subdir: 'mkl_fft/interfaces'
)

install_subdir(
    'mkl_fft/tests',
    install_dir: py.get_install_dir() / 'mkl_fft'
)
