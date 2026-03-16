NVCC      := nvcc
NVCCFLAGS := -O3 -arch=sm_80 --expt-relaxed-constexpr \
             -Xcompiler -Wall \
             --expt-extended-lambda

TARGET := nf4_test
SRCS   := main.cu nf4_dequant.cu nf4_dequant_v2.cu nf4_gemv.cu

all: $(TARGET)

$(TARGET): $(SRCS) nf4_dequant.cuh nf4_dequant_v2.cuh nf4_gemv.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRCS)

clean:
	rm -f $(TARGET)