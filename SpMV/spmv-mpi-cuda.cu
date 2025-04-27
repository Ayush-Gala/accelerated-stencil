#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cmdline.h"
#include "input.h"
#include "config.h"
#include "timer.h"
#include "formats.h"
#include <mpi.h>

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

double benchmark_cuda_mpi_coo_spmv(coo_matrix* coo, float* h_x, float* h_y, int iter, int rank)
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
    if(rank == 0){
        printf("\tPerforming %d iterations\n", num_iterations);
    }
    
    
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
    if(rank == 0){
        printf("\tbenchmarking CUDA-MPI-COO-SpMV: %8.4f ms ( %5.2f GFLOP/s %5.1f GB/s)\n", 
           msec_per_iteration, GFLOPs, GBYTEs);
    }
    
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

    int rank, world_size;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    char * mm_filename = NULL;
    coo_matrix coo;
    int global_num_nonzeros;

    if(rank == 0){
        if (get_arg(argc, argv, "help") != NULL){
            usage(argc, argv);
            return 0;
        }

        //checking if matrix file has been provided
        if (argc == 1) {
            printf("Give a MatrixMarket file.\n");
            return -1;
        } else 
            mm_filename = argv[1];

        //reading the input matrix
        read_coo_matrix(&coo, mm_filename);

        // fill matrix with random values
        srand(13);
        for(int i = 0; i < coo.num_nonzeros; i++) {
            coo.vals[i] = 1.0 - 2.0 * (rand() / (RAND_MAX + 1.0)); 
        }

        global_num_nonzeros = coo.num_nonzeros;
        
        printf("\nfile=%s rows=%d cols=%d nonzeros=%d\n", mm_filename, coo.num_rows, coo.num_cols, coo.num_nonzeros);
        fflush(stdout);
    }

    //broadcast dimension to all nodes
    MPI_Bcast(&coo.num_nonzeros, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&coo.num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&coo.num_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //allocate x and y and y_cuda on every node
    float *x = (float*) malloc(coo.num_cols * sizeof(float));
    float *y = (float*) calloc(coo.num_rows, sizeof(float));
    float *y_cuda = (float*) calloc(coo.num_rows, sizeof(float));

    if(rank == 0) {
    
        for(int i = 0; i < coo.num_cols; i++) {
            x[i] = rand() / (RAND_MAX + 1.0); 
            // x[i] = 1;
        }
    }

    //to equalize the number of iterations between CUDA and CPU runs
    int iter;
    double cpu_ms, cuda_mpi_ms;

    if(rank == 0){
        /* Benchmarking */
        printf("\n--- CPU Implementation ---\n");
        cpu_ms = benchmark_coo_spmv(&coo, x, y, iter);
    }

    //serializing
    MPI_Barrier(MPI_COMM_WORLD);

    //broadcasting number of iterations to each node
    MPI_Bcast(&iter, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //TO-DO broadcast x to all nodes
    MPI_Bcast(x, coo.num_cols, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if(rank == 0)
    {
        // Arrays to store send counts and displacements
        // Calculate chunk size for each node
        int chunk_size = coo.num_nonzeros / world_size;
        int remainder = coo.num_nonzeros % world_size;
                
        int* sendcounts = (int*)malloc(world_size * sizeof(int));
        int* displs = (int*)malloc(world_size * sizeof(int));

        // Calculate send counts and displacements
        int offset = 0;
        for (int i = 0; i < world_size; i++) {
            sendcounts[i] = chunk_size + (i < remainder ? 1 : 0);
            displs[i] = offset;
            offset += sendcounts[i];
        }

        //TO-DO send i-th chunk to i-th node
        // Scatter chunks of coo matrix data
        MPI_Scatterv(coo.vals, sendcounts, displs, MPI_FLOAT, 
            NULL, 0, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatterv(coo.rows, sendcounts, displs, MPI_INT, 
            NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatterv(coo.cols, sendcounts, displs, MPI_INT, 
            NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);

        int local_size = chunk_size + (rank < remainder ? 1 : 0);
        global_num_nonzeros = coo.num_nonzeros;
        coo.num_nonzeros = local_size;

        free(sendcounts);
        free(displs);

    }
    else {
        //TO-DO initialize buffer for x and chunk
        // Calculate local chunk size
        int chunk_size = coo.num_nonzeros / world_size;
        int remainder = coo.num_nonzeros % world_size;
        int local_size = chunk_size + (rank < remainder ? 1 : 0);

        //TO-DO receive data from root
        coo.num_nonzeros = local_size;
        coo.vals = (float*)malloc(local_size * sizeof(float));
        coo.rows = (int*)malloc(local_size * sizeof(int));
        coo.cols = (int*)malloc(local_size * sizeof(int));

        // Receive scattered chunks
        MPI_Scatterv(NULL, NULL, NULL, MPI_FLOAT, 
                     coo.vals, local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatterv(NULL, NULL, NULL, MPI_INT, 
                     coo.rows, local_size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatterv(NULL, NULL, NULL, MPI_INT, 
                     coo.cols, local_size, MPI_INT, 0, MPI_COMM_WORLD);
    
    }

    // calling the core computation function
    if(rank == 0){
        printf("\n--- CUDA-MPI Implementation ---\n");
    }
    cuda_mpi_ms = benchmark_cuda_mpi_coo_spmv(&coo, x, y_cuda, iter, rank);

    // Reduce all local results to root node
    float* global_y = NULL;
    if (rank == 0) {
        global_y = (float*)malloc(coo.num_rows * sizeof(float));
    }

    //serializing
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(y_cuda, global_y, coo.num_rows, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Update y pointer in root node
    if (rank == 0) {
        coo.num_nonzeros = global_num_nonzeros;
        free(y_cuda);
        y_cuda = global_y;
    }
    
    if(rank == 0){
        printf("\nPerformance Comparison:\n");
        printf("CPU: %8.4f ms\n", cpu_ms);
        printf("CUDA-MPI: %8.4f ms\n", cuda_mpi_ms);
        printf("Speedup: %8.2fx\n", cpu_ms / cuda_mpi_ms);
    }
    
    /* Test correctness */
#ifdef TESTING
    if(rank == 0){
        // Verify CUDA and CPU results match
        float max_diff = 0.0f;
        for(int i = 0; i < coo.num_rows; i++) {
            float diff = fabsf(y[i] - y_cuda[i]);
            max_diff = max_diff > diff ? max_diff : diff;
            if(diff == max_diff){
                printf("\n Max diff at i = %d \n", i);
            }
        }
        printf("\nMaximum difference between CPU and CUDA results: %e\n", max_diff);
        
        // Write test vectors
        printf("Writing x and y vectors ...");
        FILE *fp = fopen("test_x", "w");
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
    }
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    delete_coo_matrix(&coo);
    free(x);
    free(y);
    free(y_cuda);

    return 0;
}