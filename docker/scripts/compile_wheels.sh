#!/usr/bin/env bash

for PYBIN in /opt/python/*/bin;
	do
      "${PYBIN}/pip" wheel -w wheelhouse/ src/
      "${PYBIN}/pip" install --no-index --no-cache-dir --find-links=/wheelhouse fastfm
      "${PYBIN}/python" -m pytest src/fastfm2
done

for whl in wheelhouse/fastfm*.whl
	do  auditwheel show "$whl"
	    auditwheel repair "$whl"
done

rclone copy wheelhouse/fasfm*.whl data:palaimon-releases/fastfm/wheelhouse