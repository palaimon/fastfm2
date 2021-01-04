all:
	( cd fastfm-core2 ; \
	  cmake -H. -B_lib -DEXTERNAL_RELEASE=1 \
	                   -DCMAKE_BUILD_TYPE=Release \
	                   -DCMAKE_DEBUG_POSTFIX=d; \
	  cmake --build _lib )

.PHONY : pyclean
pyclean:
	rm -f *.so
	rm -f fastfm2/core/ffm2.cpp

.PHONY : clean
clean:
	rm -rf fastfm-core2/_lib/
	rm -f *.so
	rm -rf build/
	rm -f fastfm2/core/ffm2.cpp
