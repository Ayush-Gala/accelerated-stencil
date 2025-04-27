NVCC = nvcc
NVFLAGS = -O3 -std=c++11 -I./include/ -diag-suppress 177,2464

all: spmv-cuda spmv-mpi-cuda

spmv-cuda: spmv-cuda.cu mmio.c
	$(NVCC) $(NVFLAGS) -x c mmio.c -x cu spmv-cuda.cu -o $@

spmv-mpi-cuda: spmv-mpi-cuda.cu mmio.c
	$(NVCC) $(NVFLAGS) -x c mmio.c -x cu spmv-mpi-cuda.cu -ccbin mpicxx -Xcompiler "-DMPICH_SKIP_MPICXX" -o $@ -lmpi

clean:
	rm -f spmv-cuda spmv-mpi-cuda test_COO test_x test_y_cpu test_y_cuda

retest: 
	rm -f test_COO test_x test_y_cpu test_y_cuda