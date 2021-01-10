import os
import shutil
from distutils.command.build_ext import build_ext
from distutils.core import Distribution, Extension

from Cython.Build import cythonize
import numpy

from version import get_version_from_pyproject

ffm2_include_dir = os.getenv('FFM_INCLUDE_DIR', 'fastfm-core2/fastfm/')
ffm2_solvers_include_dir = os.getenv('FFM_INCLUDE_SOLVERS_DIR',
                                     'fastfm-core2/fastfm/solvers/')
ffm2_library_dir = os.getenv('FFM_LIBRARY_DIR', 'fastfm-core2/_lib/fastfm/')
ffm2_library_solvers_dir = os.getenv('FFM_LIBRARY_SOLVERS_DIR',
                                     'fastfm-core2/_lib/fastfm/solvers')


def build():
    extensions = [
        Extension('ffm2', ['fastfm2/core/ffm2.pyx'],
                  libraries=['fastfm', 'solvers'],
                  library_dirs=[ffm2_library_dir, ffm2_library_solvers_dir],
                  include_dirs=['fastfm2/core',
                                ffm2_include_dir,
                                ffm2_solvers_include_dir,
                                numpy.get_include()
                                ],
                  extra_compile_args=['-std=c++11', '-Wall'],
                  extra_link_args=['-std=c++11', '-mstackrealign'],
                  language="c++")
    ]
    ext_modules = cythonize(
        extensions,
        compile_time_env=dict(EXTERNAL_RELEASE=True),
        compiler_directives={'binding': True, 'language_level': 3},
    )

    distribution = Distribution(
        {'name': 'fastfm2',
         'ext_modules': ext_modules,
         'package_data': {'fastfm2': ['fastfm2/core/*.pxd']},
         'version': get_version_from_pyproject(),
         })

    distribution.package_dir = 'fastfm2'

    cmd = build_ext(distribution)
    cmd.ensure_finalized()
    cmd.run()

    # Copy built lib to project root
    for output in cmd.get_outputs():
        relative_extension = os.path.relpath(output, cmd.build_lib)
        shutil.copyfile(output, relative_extension)
        mode = os.stat(relative_extension).st_mode
        mode |= (mode & 0o444) >> 2
        os.chmod(relative_extension, mode)


if __name__ == '__main__':
    build()
