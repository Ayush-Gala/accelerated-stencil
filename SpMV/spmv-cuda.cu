#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cmdline.h"
#include "input.h"
#include "config.h"
#include "timer.h"
#include "formats.h"

#define max(a,b) \
({ __typeof__ (a) _a = (a); \
   __typeof__ (b) _b = (b); \
 _a > _b ? _a : _b; })

#define min(a,b) \
({ __typeof__ (a) _a = (a); \
   __typeof__ (b) _b = (b); \
 _a < _b ? _a : _b; })

void usage(int argc, char** argv)
{
    printf("Usage: %s [my_matrix.mtx]\n", argv[0]);
    printf("Note: my_matrix.mtx must be real-valued sparse matrix in the MatrixMarket file format.\n"); 
}

// CUDA kernel for COO-SpMV
__global__ void coo_spmv_kernel(int num_nonzeros, const int* __restrict__ rows, 
                               const int* __restrict__ cols, 
                               const float* __restrict__ vals, 
                               const float* __restrict__ x, 
                               float* __restrict__ y)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_nonzeros) {
        atomicAdd(&y[rows[idx]], vals[idx] * x[cols[idx]]);
    }
}

double benchmark_cuda_coo_spmv(coo_matrix* coo, float* h_x, float* h_y, int iter)
{
    int num_nonzeros = coo->num_nonzeros;
    int num_rows = coo->num_rows;
    int num_cols = coo->num_cols;
    
    // Allocate device memory
    int *d_rows, *d_cols;
    float *d_vals, *d_x, *d_y;
    
    cudaMalloc((void**)&d_rows, num_nonzeros * sizeof(int));
    cudaMalloc((void**)&d_cols, num_nonzeros * sizeof(int));
    cudaMalloc((void**)&d_vals, num_nonzeros * sizeof(float));
    cudaMalloc((void**)&d_x, num_cols * sizeof(float));
    cudaMalloc((void**)&d_y, num_rows * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_rows, coo->rows, num_nonzeros * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, coo->cols, num_nonzeros * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, coo->vals, num_nonzeros * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, num_cols * sizeof(float), cudaMemcpyHostToDevice);
    
    // Calculate grid and block dimensions
    int blockSize = 256;
    int gridSize = (num_nonzeros + blockSize - 1) / blockSize;
    
    // Zero output vector
    cudaMemset(d_y, 0, num_rows * sizeof(float));

    // Warmup run
    timer time_one_iteration;
    timer_start(&time_one_iteration);
    
    coo_spmv_kernel<<<gridSize, blockSize>>>(num_nonzeros, d_rows, d_cols, d_vals, d_x, d_y);
    
    cudaDeviceSynchronize();
    double estimated_time = seconds_elapsed(&time_one_iteration);
    
    // number of iterations is the same as serial argument passed
    int num_iterations;
    num_iterations = iter;
    printf("\tPerforming %d iterations\n", num_iterations);
    
    // Benchmark SpMV iterations
    timer t;
    timer_start(&t);
    
    for (int j = 0; j < num_iterations; j++) {
        // Launch kernel
        coo_spmv_kernel<<<gridSize, blockSize>>>(num_nonzeros, d_rows, d_cols, d_vals, d_x, d_y);
        // Check for errors
        cudaGetLastError();
    }
    
    cudaDeviceSynchronize();

    double msec_per_iteration = milliseconds_elapsed(&t) / (double)num_iterations;
    double sec_per_iteration = msec_per_iteration / 1000.0;
    double GFLOPs = (sec_per_iteration == 0) ? 0 : (2.0 * (double)coo->num_nonzeros / sec_per_iteration) / 1e9;
    double GBYTEs = (sec_per_iteration == 0) ? 0 : ((double)bytes_per_coo_spmv(coo) / sec_per_iteration) / 1e9;
    printf("\tbenchmarking CUDA-COO-SpMV: %8.4f ms ( %5.2f GFLOP/s %5.1f GB/s)\n", 
           msec_per_iteration, GFLOPs, GBYTEs);
    
    // Copy result back to host for verification
    cudaMemcpy(h_y, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_rows);
    cudaFree(d_cols);
    cudaFree(d_vals);
    cudaFree(d_x);
    cudaFree(d_y);
    
    return msec_per_iteration;
}

// Sequential implementation for comparison
double benchmark_coo_spmv(coo_matrix * coo, float* x, float* y, int& iter)
{
    int num_nonzeros = coo->num_nonzeros;

    // warmup    
    timer time_one_iteration;
    timer_start(&time_one_iteration);
    for (int i = 0; i < num_nonzeros; i++){   
        y[coo->rows[i]] += coo->vals[i] * x[coo->cols[i]];
    }

    double estimated_time = seconds_elapsed(&time_one_iteration); 

    // determine # of seconds dynamically
    int num_iterations;
    num_iterations = MAX_ITER;

    if (estimated_time == 0)
        num_iterations = MAX_ITER;
    else {
        num_iterations = min(MAX_ITER, max(MIN_ITER, (int) (TIME_LIMIT / estimated_time)) ); 
    }
    printf("\tPerforming %d iterations\n", num_iterations);
    iter = num_iterations;

    // time several SpMV iterations
    timer t;
    timer_start(&t);
    for(int j = 0; j < num_iterations; j++) {
        for (int i = 0; i < num_nonzeros; i++){   
            y[coo->rows[i]] += coo->vals[i] * x[coo->cols[i]];
        }
    }
    double msec_per_iteration = milliseconds_elapsed(&t) / (double) num_iterations;
    double sec_per_iteration = msec_per_iteration / 1000.0;
    double GFLOPs = (sec_per_iteration == 0) ? 0 : (2.0 * (double) coo->num_nonzeros / sec_per_iteration) / 1e9;
    double GBYTEs = (sec_per_iteration == 0) ? 0 : ((double) bytes_per_coo_spmv(coo) / sec_per_iteration) / 1e9;
    printf("\tbenchmarking CPU-COO-SpMV: %8.4f ms ( %5.2f GFLOP/s %5.1f GB/s)\n", msec_per_iteration, GFLOPs, GBYTEs); 

    return msec_per_iteration;
}

int main(int argc, char** argv)
{
    if (get_arg(argc, argv, "help") != NULL){
        usage(argc, argv);
        return 0;
    }

    char * mm_filename = NULL;
    if (argc == 1) {
        printf("Give a MatrixMarket file.\n");
        return -1;
    } else 
        mm_filename = argv[1];

    coo_matrix coo;
    read_coo_matrix(&coo, mm_filename);

    // fill matrix with random values
    srand(13);
    for(int i = 0; i < coo.num_nonzeros; i++) {
        coo.vals[i] = 1.0 - 2.0 * (rand() / (RAND_MAX + 1.0)); 
    }
    
    printf("\nfile=%s rows=%d cols=%d nonzeros=%d\n", mm_filename, coo.num_rows, coo.num_cols, coo.num_nonzeros);
    fflush(stdout);

#ifdef TESTING
    // Print matrix in COO format for testing
    printf("Writing matrix in COO format to test_COO ...");
    FILE *fp = fopen("test_COO", "w");
    fprintf(fp, "%d\t%d\t%d\n", coo.num_rows, coo.num_cols, coo.num_nonzeros);
    fprintf(fp, "coo.rows:\n");
    for (int i=0; i<coo.num_nonzeros; i++)
    {
      fprintf(fp, "%d  ", coo.rows[i]);
    }
    fprintf(fp, "\n\n");
    fprintf(fp, "coo.cols:\n");
    for (int i=0; i<coo.num_nonzeros; i++)
    {
      fprintf(fp, "%d  ", coo.cols[i]);
    }
    fprintf(fp, "\n\n");
    fprintf(fp, "coo.vals:\n");
    for (int i=0; i<coo.num_nonzeros; i++)
    {
      fprintf(fp, "%f  ", coo.vals[i]);
    }
    fprintf(fp, "\n");
    fclose(fp);
    printf("... done!\n");
#endif 

    // Initialize host arrays
    float * x = (float*)malloc(coo.num_cols * sizeof(float));
    float * y = (float*)malloc(coo.num_rows * sizeof(float));
    float * y_cuda = (float*)malloc(coo.num_rows * sizeof(float));  // For CUDA results

    for(int i = 0; i < coo.num_cols; i++) {
        x[i] = rand() / (RAND_MAX + 1.0); 
    }
    
    // Reset output vectors
    for(int i = 0; i < coo.num_rows; i++) {
        y[i] = 0;
        y_cuda[i] = 0;
    }

    //to equalize the number of iterations between CUDA and CPU runs
    int iter;

    /* Benchmarking */
    printf("\n--- CPU Implementation ---\n");
    double cpu_ms = benchmark_coo_spmv(&coo, x, y, iter);
    
    printf("\n--- CUDA Implementation ---\n");
    double cuda_ms = benchmark_cuda_coo_spmv(&coo, x, y_cuda, iter);
    
    printf("\nPerformance Comparison:\n");
    printf("CPU: %8.4f ms\n", cpu_ms);
    printf("CUDA: %8.4f ms\n", cuda_ms);
    printf("Speedup: %8.2fx\n", cpu_ms / cuda_ms);
    
    /* Test correctness */
#ifdef TESTING
    // Verify CUDA and CPU results match
    float max_diff = 0.0f;
    for(int i = 0; i < coo.num_rows; i++) {
        float diff = fabsf(y[i] - y_cuda[i]);
        max_diff = max_diff > diff ? max_diff : diff;
    }
    printf("\nMaximum difference between CPU and CUDA results: %e\n", max_diff);
    
    // Write test vectors
    printf("Writing x and y vectors ...");
    fp = fopen("test_x", "w");
    for (int i=0; i<coo.num_cols; i++)
    {
      fprintf(fp, "%f\n", x[i]);
    }
    fclose(fp);
    
    fp = fopen("test_y_cpu", "w");
    for (int i=0; i<coo.num_rows; i++)
    {
      fprintf(fp, "%f\n", y[i]);
    }
    fclose(fp);
    
    fp = fopen("test_y_cuda", "w");
    for (int i=0; i<coo.num_rows; i++)
    {
      fprintf(fp, "%f\n", y_cuda[i]);
    }
    fclose(fp);
    printf("... done!\n");
#endif

    delete_coo_matrix(&coo);
    free(x);
    free(y);
    free(y_cuda);

    return 0;
}