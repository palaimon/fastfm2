#!/usr/bin/env bash
set -ex

rm -rf __pycache__
rm -rf .pytest_cache
rm -rf build
rm -rf dist
rm -rf notebooks

rm -rf fastfm2
rm -rf fastfm2.egg*
rm -rf ffm2*.so

rm -rf Makefile
rm -rf poetry.lock
rm -rf pyproject.toml
rm -rf build.py
rm -rf version.py
rm -rf version.txt
rm -rf .dockerignore
rm -rf .flake8
rm -rf generate_requierments.sh

rm -rf fastfm-core2/_lib
rm -rf fastfm-core2/CMakeLists.txt
rm -rf fastfm-core2/fastfm
rm -rf fastfm-core2/fastfm_tests

rm -rf docker
