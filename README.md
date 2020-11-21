# Welcome to fastfm v2
> NB: This is still in early development. Use [v1](https://github.com/ibayer/fastFM) unless you want to contribute to the next version of fastai


To learn more about the library, read our introduction in the [paper](http://arxiv.org/abs/1505.00641) presenting it.

Note that c++ dependencies are in a submodule, so to clone with all dependencies included, you should use:

     git clone --recurse-submodules https://github.com/palaimon/fastfm2

## Installing

We infrequently push wheels to pypi that you can install with `pip install fastfm2`.

### Source Install

You can build the latest version from source (requires `cmake>=3.12`) by first compiling the c++ library by running

```bash
make
```
from the root dir and then install the lib locally for dev

```bash
poetry install
```

or build the **python weehls**.

`macos`:
```
poetry run python setup.py bdist_wheel                  && \
poetry run delocate-wheel -w fixed_whl_macos dist/*.whl && \
rm -rf build && rm -rf dist
```

`linux` [auditwheel instead delocate]:

```
poetry run python setup.py bdist_wheel                  && \
poetry run auditwheel repair dist/fastfm-*.whl
```

## Tests

To run the tests launch:

```bash
pytest
```

For the tests to run, you'll need to install the following optional dependency:

```
pip install pytest
```
