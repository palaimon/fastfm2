## HOWTO update 

#### Clone fastfm repo:
```ssh://git@gitlab.palaimon.io:34892/collab/fastfm.git```

#### Run extraction script: (todo: script outdated atm!)
```
cd fastfm
./copybara.py
```
- as a result you will gain `_external_release` directory. 
- merge/update all files files from `_external_release` to the root directory of `fastfm2` repo

NB! Do not cleanup/remove existing directory, ci stuff and eigen submodule stay unchanged. They are not autogenerated.  

#### Version
`version_ex.txt` contains git hash for extracted release

## HOWTO:

### build lib
Run `make` from root dir to build the lib

### build whls

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

### install lib locally

`poetry install`

