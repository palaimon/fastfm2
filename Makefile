PYTHON ?= python3

all:
	$(PYTHON) version.py
	( cd fastFM-core2 ; \
	  cmake -H. -B_lib -DEXTERNAL_RELEASE=1 \
	                   -DCMAKE_BUILD_TYPE=Release \
	                   -DCMAKE_DEBUG_POSTFIX=d; \
	  cmake --build _lib )
	$(PYTHON) setup.py build_ext --inplace

.PHONY : pyclean
pyclean:
	rm -f *.so
	rm -f fastfm2/core/ffm2.cpp

.PHONY : clean
clean:
	rm -rf fastFM-core2/_lib/
	rm -f *.so
	rm -rf build/
	rm -f fastfm2/core/ffm2.cpp
