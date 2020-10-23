import os

import numpy
from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Distutils import build_ext

from version import get_version_from_txt

ffm2_include_dir = os.getenv("FFM_INCLUDE_DIR", 'fastFM-core2/fastfm/')
ffm2_solvers_include_dir = os.getenv("FFM_INCLUDE_SOLVERS_DIR", 'fastFM-core2/fastfm/solvers/')
ffm2_library_dir = os.getenv("FFM_LIBRARY_DIR", 'fastFM-core2/_lib/fastfm/')
ffm2_library_solvers_dir = os.getenv("FFM_LIBRARY_SOLVERS_DIR", 'fastFM-core2/_lib/fastfm/solvers')


ext_modules = [
    Extension('ffm2', ['fastfm2/ffm2.pyx'],
              libraries=['fastfm', 'solvers'],
              library_dirs=[ffm2_library_dir, ffm2_library_solvers_dir],
              include_dirs=['fastfm2/',
                            ffm2_include_dir,
                            ffm2_solvers_include_dir,
                            numpy.get_include()],
              extra_compile_args=['-std=c++11', '-Wall', '-pedantic'],
              extra_link_args=['-std=c++11', '-mstackrealign'],
              language="c++",
              cython_directives = {'language_level': "3"})]


setup(
    name='fastfm2',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,

    packages=find_packages(),

    package_data={'fastfm2': ['fastfm2/*.pxd']},

    version=get_version_from_txt(),
    url='http://ibayer.github.io/fastFM',
    author='Immanuel Bayer',
    author_email='immanuel.bayer@uni-konstanz.de',

    # Choose your license
    license='BSD',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',

        'License :: OSI Approved :: BSD License',
        'Operating System :: Unix',

        'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.5',
        # 'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
    ],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['numpy', 'scikit-learn', 'scipy']
)
