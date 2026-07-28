NVCC := nvcc
NVCC_ARCH := -arch=sm_80
CXXSTD := -std=c++17
OPT := -O3

SRC_CU := main.cu cuda_gemm.cu
SRC_CPP := cpu_gemm.cpp
HEADERS := common.h cpu_gemm.h cuda_gemm.cuh

TARGET := gemm_bench

all: $(TARGET)

$(TARGET): $(SRC_CU) $(SRC_CPP) $(HEADERS)
	$(NVCC) $(OPT) $(CXXSTD) $(NVCC_ARCH) $(SRC_CU) $(SRC_CPP) -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean
