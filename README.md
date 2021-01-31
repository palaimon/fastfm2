# Welcome to fastfm v2
![CI-Badge](https://github.com/palaimon/fastfm2/workflows/WHL/badge.svg)

> NB: This is still in early development. Use [v1](https://github.com/ibayer/fastFM) unless you want to contribute to the next version of fastfm


To learn more about the library, read our introduction in the [paper](http://arxiv.org/abs/1505.00641) presenting it.

Note that c++ dependencies are in a submodule, so to clone with all dependencies included, you should use:

     git clone --recurse-submodules https://github.com/palaimon/fastfm2

## Installing

We infrequently push wheels to pypi that you can install with `pip install fastfm2`.

### Source Install

You can build the latest version from source (requires `cmake>=3.12`) by first compiling the c++ library from the project root directory:

```bash
make
```
then install fastfm2 python lib locally:

#### User install

```bash
pip install .
```
Also you can build **python wheels**:

`macos`:
```shell
pip wheel . --no-deps -w wheelhouse
delocate-wheel -w fixed_macos_whls wheelhouse/fastfm*.whl
```

`linux`:
```shell
pip wheel . --no-deps -w wheelhouse
auditwheel repair wheelhouse/fastfm-*.whl
```
(`auditwheel` used instead `delocate`).

#### Dev install

For development we use poetry as dependency managmer:
```bash
poetry install
```

and wheels:
```shell
poetry build
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
