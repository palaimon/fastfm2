#!/usr/bin/env bash

for PYBIN in /opt/python/*/bin;
	do "${PYBIN}/pip" install --no-cache-dir -r src/requirements.txt
	   "${PYBIN}/pip" install pytest
done
