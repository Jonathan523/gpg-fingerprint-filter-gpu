NVCC = nvcc
NVCCFLAGS += -O3 -std=c++14 --compiler-options -Wall,-Wextra
LIBS = -lgcrypt -lnvrtc -lcuda

.PHONY: all clean

all: gpg-fingerprint-filter-gpu

key_test_sha1.o: key_test_sha1.cu
	$(NVCC) -c -o $@ $(NVCCFLAGS) $^

key_test_pattern.o: key_test_pattern.cpp
	$(NVCC) -c -o $@ $(NVCCFLAGS) $^

key_test.o: key_test.cpp
	$(NVCC) -c -o $@ $(NVCCFLAGS) $^

gpg_helper.o: gpg_helper.cpp
	$(NVCC) -c -o $@ $(NVCCFLAGS) $^

gpg-fingerprint-filter-gpu: main.cpp key_test.o key_test_sha1.o key_test_pattern.o gpg_helper.o
	$(NVCC) -o $@ $(NVCCFLAGS) $(LIBS) $^

clean:
	-rm -f *.o gpg-fingerprint-filter-gpu
