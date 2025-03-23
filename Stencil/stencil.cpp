#include <iostream>
#include <chrono>
#include <numeric>
#include <assert.h>
#include <math.h>
#include <vector>
#include <memory>
#include <iomanip>
#include <xmmintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

#ifndef __GNUC__
#define __restrict__
#endif

// Partial load. Load n elements and set the rest to 0
// You can call this function for boudary cases.
void load_partial(__m128 * xmm, int n, float const * p) {
    __m128 t1, t2;
    switch (n) {
    case 1:
        *xmm = _mm_load_ss(p); break;
    case 2:
        *xmm = _mm_castpd_ps(_mm_load_sd((double const*)p)); break;
    case 3:
        t1 = _mm_castpd_ps(_mm_load_sd((double const*)p));
        t2 = _mm_load_ss(p + 2);
        *xmm = _mm_movelh_ps(t1, t2); break;
    case 4:
        *xmm = _mm_loadu_ps(p); break;
    default:
        *xmm = _mm_setzero_ps();
    }
    return;
}

// Print the grid for debugging purposes
template <typename T>
void print_grid(int N, T vec) {
    for (int x = 0; x < N; ++x) {
        for (int y = 0; y < N; ++y) {
            cout << setprecision(6) << vec[x*N + y] << " ";     //changed precision to 6
        }
        cout << endl;
    }
}

// Test if one grid equals another with a small margin of error epsilon
bool test_grids(int N, float* __restrict__ grid_1, float* __restrict__ grid_2) {
    const double epsilon = 0.0002;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (!(fabs(grid_1[i*N + j] - grid_2[i*N + j]) < epsilon)) {
                cout << "ERROR: " << grid_1[i*N + j] << " != " << grid_2[i*N + j] << endl;
                return false;
            }
        }
    }
    return true;
}

// Return a random variable between fMin and fMax
float fRand(float fMin, float fMax) {
    float f = (float)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

// Initialize a N * N grid with random variables coming from a seed
float* initialize_grid(int N, float fMin, float fMax) {
    float* vec = new(std::align_val_t(64)) float[N * N];
    const unsigned int seed = 42;
    srand(seed);
    for (unsigned int x = 0; x < N; ++x) {
        for (unsigned int y = 0; y < N; ++y) {
            // For better debugging
            // vec[x*N + y] = x*N + y + 100;
            vec[x*N + y] = fRand(fMin, fMax);
        }
    }
    return vec;
}

// Transpose a given grid and return that transposed one
float* transpose(int N, float* src) {
    float* tar = new(std::align_val_t(64)) float[N * N];
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            tar[i*N + k] = src[k*N + i];
        }
    }
    return tar;
}

// Basic solution for the local mean kernel, allocate a tmp array of size 
// N * N for temporary write and later read backs
void basic_solution(int N, int K, float* vec) {
    // Allocate tmp grid
    float* tmp = new(std::align_val_t(64)) float[N*N];

    // Find neighbors
    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++) {
            int left_boundary = max(y - K, 0), right_boundary = min(y + K, N - 1);
            int bottom_boundary = max(x - K, 0), upper_boundary = min(x + K, N - 1);
            int neighbors_count = (upper_boundary - bottom_boundary) + (right_boundary - left_boundary);
            float sum = 0.0;

            for (int i = left_boundary; i <= right_boundary; i++) {
                if (i != y) {
                    sum += vec[x*N + i];
                }
            }

            for (int i = bottom_boundary; i <= upper_boundary; i++) {
                if (i != x) {
                    sum += vec[i*N + y];
                }
            }

            // Replace current value at u(x, y) with local mean
            tmp[x*N + y] = sum / neighbors_count;
        }
    }

    // Write back
    for (int x = 0; x < N; ++x) {
        for (int y = 0; y < N; ++y) {
            vec[x*N + y] = tmp[x*N + y];
        }
    }
}

// Block size is compile parameter B
/*****  Based on this code to implement your SIMD and OMP code  *****/
template<int B>
void blocked_solution(int N, int K, float* __restrict__ vec, float* __restrict__ trans) {
    // Divisibility contraints
    if (N % B != 0) {
        cout << "N must be divisible through B" << endl;
        exit(-1);
    }
    // Allocate tmp grid
    float* tmp = new(std::align_val_t(64)) float[N*N];

    // Find neighbors
    for (int I = 0; I < N; I+=B) {
        for (int J = 0; J < N; J+=B) {
            for (int x = I; x < I + B; ++x) {
                for (int y = J; y < J + B; ++y) {
                    int left_boundary = max(y - K, 0), right_boundary = min(y + K, N - 1);
                    int bottom_boundary = max(x - K, 0), upper_boundary = min(x + K, N - 1);
                    int neighbors_count = (right_boundary - left_boundary) + (upper_boundary - bottom_boundary);
                    float sum = 0.0;

                    for (int i = left_boundary; i <= right_boundary; i++) {
                        if (i != y) {
                            sum += vec[x*N + i];
                        }
                    }

                    for (int i = bottom_boundary; i <= upper_boundary; i++) {
                        if (i != x) {
                            sum += trans[y*N+i]; // Makes it a lot faster due to consecutive reads
                        }
                    }

                    // Replace current value at u(x, y) with local mean
                    tmp[x*N + y] = sum / neighbors_count;
                }
            }
        }
    }

    // Write back
    for (int x = 0; x < N; ++x) {
        for (int y = 0; y < N; ++y) {
            vec[x*N + y] = tmp[x*N + y];
        }
    }
}

template<int B>
void blocked_simd(int N, int K, float* __restrict__ vec, float* __restrict__ trans) {
    // Allocate aligned tmp grid for results
    float* tmp = new(std::align_val_t(64)) float[N*N];
    
    // SIMD width is 4 floats with SSE
    const int SIMD_WIDTH = 4;
    
    // Find neighbors
    for (int I = 0; I < N; I+=B) {
        for (int J = 0; J < N; J+=B) {
            // Process block
            for (int x = I; x < I + B; ++x) {
                for (int y = J; y < J + B; ++y) {
                    int left_boundary = max(y - K, 0), right_boundary = min(y + K, N - 1);
                    int bottom_boundary = max(x - K, 0), upper_boundary = min(x + K, N - 1);
                    
                    int neighbors_count = (right_boundary - left_boundary) + (upper_boundary - bottom_boundary);
                    
                    // Get center element value (we'll subtract it twice later)
                    float center_value = vec[x*N + y];

                    /* ----------------------------------------------
                    PROCESSING HORIZONTAL ELEMENTS OF THE INPUT GRID
                    ------------------------------------------------- */
                    
                    // Vector to accumulate horizontal sums
                    __m128 h_sum_vec = _mm_setzero_ps();
                    float h_sum_scalar = 0.0f;
                    
                    // Process horizontal neighbors (row-wise) INCLUDING center
                    // Process initial non-aligned elements
                    int i = left_boundary;
                    for (; i <= right_boundary && (i % SIMD_WIDTH) != 0; ++i) {
                        h_sum_scalar += vec[x*N + i];
                    }
                    
                    // Process aligned SIMD chunks
                    for (; i + SIMD_WIDTH - 1 <= right_boundary; i += SIMD_WIDTH) {
                        __m128 row_vec = _mm_loadu_ps(&vec[x*N + i]);
                        h_sum_vec = _mm_add_ps(h_sum_vec, row_vec);
                    }
                    
                    // Process remaining elements
                    for (; i <= right_boundary; ++i) {
                        h_sum_scalar += vec[x*N + i];
                    }

                    /* ----------------------------------------------
                    PROCESSING VERTICAL ELEMENTS USING TRANSPOSE GRID
                    ------------------------------------------------- */

                    __m128 v_sum_vec = _mm_setzero_ps();
                    float v_sum_scalar = 0.0f;

                    // Process initial non-aligned elements
                    i = bottom_boundary;
                    for (; i <= upper_boundary && (i % SIMD_WIDTH) != 0; ++i) {
                        v_sum_scalar += trans[y*N + i];
                    }

                    // Process aligned SIMD chunks
                    for (; i + SIMD_WIDTH - 1 <= upper_boundary; i += SIMD_WIDTH) {
                        __m128 col_vec = _mm_loadu_ps(&trans[y*N + i]);
                        v_sum_vec = _mm_add_ps(v_sum_vec, col_vec);
                    }

                    // Process remaining elements
                    for (; i <= upper_boundary; ++i) {
                        v_sum_scalar += trans[y*N + i];
                    }

                    // Horizontally add the 4 floats in the vertical SIMD vector
                    alignas(16) float v_sum_array[4];
                    _mm_store_ps(v_sum_array, v_sum_vec);
                    float v_simd_sum = v_sum_array[0] + v_sum_array[1] + v_sum_array[2] + v_sum_array[3];
                    
                    // Horizontally add the 4 floats in the SIMD vector
                    alignas(16) float h_sum_array[4];
                    _mm_store_ps(h_sum_array, h_sum_vec);
                    float h_simd_sum = h_sum_array[0] + h_sum_array[1] + h_sum_array[2] + h_sum_array[3];
                    
                    // Combine all sums horizontal and vertical - scalar and simd
                    float total_sum = h_simd_sum + h_sum_scalar + v_simd_sum + v_sum_scalar;
                    
                    // Subtract the center element twice (included once in horizontal and once in vertical sum)
                    total_sum -= 2 * center_value;
                    
                    // Replace current value with local mean
                    tmp[x*N + y] = total_sum / neighbors_count;
                }
            }
        }
    }
    
    // Write back results - vectorize this operation as well
    for (int x = 0; x < N; ++x) {
        int y = 0;
        // Process 4 elements at a time
        for (; y <= N - SIMD_WIDTH; y += SIMD_WIDTH) {
            __m128 tmp_vec = _mm_loadu_ps(&tmp[x*N + y]);
            _mm_storeu_ps(&vec[x*N + y], tmp_vec);
        }
        // Handle remaining elements
        for (; y < N; ++y) {
            vec[x*N + y] = tmp[x*N + y];
        }
    }
    
    // Free memory
    delete[] tmp;
}
/**********************/

/*****  OMP code  *****/
// Put the explanation of your code here

template<int B>
void blocked_simd_omp(int N, int K, float* __restrict__ vec, float* __restrict__ trans) {

    const int SIMD_WIDTH = 4;

    // Each thread gets its own temporary array to avoid race conditions
    float* tmp = new(std::align_val_t(64)) float[N*N];

    #pragma omp parallel
    {    
        // Find neighbors - parallelize the outer I loop with dynamic scheduling
        #pragma omp for schedule(dynamic)
        for (int I = 0; I < N; I+=B) {
            for (int J = 0; J < N; J+=B) {
                // Process block
                for (int x = I; x < I + B && x < N; ++x) {
                    for (int y = J; y < J + B && y < N; ++y) {
                        int left_boundary = std::max(y - K, 0), right_boundary = std::min(y + K, N - 1);
                        int bottom_boundary = std::max(x - K, 0), upper_boundary = std::min(x + K, N - 1);
                        
                        int neighbors_count = (right_boundary - left_boundary + 1) + (upper_boundary - bottom_boundary + 1) - 2;
                        
                        // Get center element value (we'll subtract it twice later)
                        float center_value = vec[x*N + y];

                        /* ----------------------------------------------
                        PROCESSING HORIZONTAL ELEMENTS OF THE INPUT GRID
                        ------------------------------------------------- */
                        
                        // Vector to accumulate horizontal sums
                        __m128 h_sum_vec = _mm_setzero_ps();
                        float h_sum_scalar = 0.0f;
                        
                        // Process horizontal neighbors (row-wise) INCLUDING center
                        // Process initial non-aligned elements
                        int i = left_boundary;
                        for (; i <= right_boundary && (i % SIMD_WIDTH) != 0; ++i) {
                            h_sum_scalar += vec[x*N + i];
                        }
                        
                        // Process aligned SIMD chunks
                        for (; i + SIMD_WIDTH - 1 <= right_boundary; i += SIMD_WIDTH) {
                            __m128 row_vec = _mm_loadu_ps(&vec[x*N + i]);
                            h_sum_vec = _mm_add_ps(h_sum_vec, row_vec);
                        }
                        
                        // Process remaining elements
                        for (; i <= right_boundary; ++i) {
                            h_sum_scalar += vec[x*N + i];
                        }

                        /* ----------------------------------------------
                        PROCESSING VERTICAL ELEMENTS USING TRANSPOSE GRID
                        ------------------------------------------------- */

                        __m128 v_sum_vec = _mm_setzero_ps();
                        float v_sum_scalar = 0.0f;

                        // Process initial non-aligned elements
                        i = bottom_boundary;
                        for (; i <= upper_boundary && (i % SIMD_WIDTH) != 0; ++i) {
                            v_sum_scalar += trans[y*N + i];
                        }

                        // Process aligned SIMD chunks
                        for (; i + SIMD_WIDTH - 1 <= upper_boundary; i += SIMD_WIDTH) {
                            __m128 col_vec = _mm_loadu_ps(&trans[y*N + i]);
                            v_sum_vec = _mm_add_ps(v_sum_vec, col_vec);
                        }

                        // Process remaining elements
                        for (; i <= upper_boundary; ++i) {
                            v_sum_scalar += trans[y*N + i];
                        }

                        // Horizontally add the 4 floats in the vertical SIMD vector
                        alignas(16) float v_sum_array[4];
                        _mm_store_ps(v_sum_array, v_sum_vec);
                        float v_simd_sum = v_sum_array[0] + v_sum_array[1] + v_sum_array[2] + v_sum_array[3];
                        
                        // Horizontally add the 4 floats in the SIMD vector
                        alignas(16) float h_sum_array[4];
                        _mm_store_ps(h_sum_array, h_sum_vec);
                        float h_simd_sum = h_sum_array[0] + h_sum_array[1] + h_sum_array[2] + h_sum_array[3];
                        
                        // Combine all sums horizontal and vertical - scalar and simd
                        float total_sum = h_simd_sum + h_sum_scalar + v_simd_sum + v_sum_scalar;
                        
                        // Subtract the center element twice (included once in horizontal and once in vertical sum)
                        total_sum -= 2 * center_value;
                        
                        // Replace current value with local mean in thread-local array
                        tmp[x*N + y] = total_sum / neighbors_count;
                    }
                }
            }
        }
        
        // Write back results from thread-local arrays to the global array
        // Use a different parallelization strategy for this phase
        #pragma omp for schedule(static)
        for (int x = 0; x < N; ++x) {
            int y = 0;
            // Process 4 elements at a time
            for (; y <= N - SIMD_WIDTH; y += SIMD_WIDTH) {
                __m128 tmp_vec = _mm_loadu_ps(&tmp[x*N + y]);
                _mm_storeu_ps(&vec[x*N + y], tmp_vec);
            }
            // Handle remaining elements
            for (; y < N; ++y) {
                vec[x*N + y] = tmp[x*N + y];
            }
        }
    }

    // Free thread-local memory
    delete[] tmp;
}

/**********************/


int main(int argc, char* argv[]) {
    // Block size for blocked implementations
    const int B = 32; //32

    // Grid size N and amount of neighbors K
    int N = 8192;  //1024
    int K = 8;  //8
    
    // Make it possible to initialize the variables while the program is running
    if (argc == 3) {
        N = atoi(argv[1]);
        K = atoi(argv[2]);
    }

    if (argc != 1 && argc != 3) {
        cout << "Usage:" << endl;
        cout << "./program" << endl;
        cout << "./program <N> <K>" << endl;
        exit(-1);
    }

    cout << "N: " << N << ", K: " << K << ", B: " << B << endl;

    // Run the reference solution
    float* reference = initialize_grid(N, -100, 100);   //-> initializes a N*N grid to reference
    // print_grid(N, reference);
    // cout<<endl<<endl<<endl;
    auto begin = chrono::high_resolution_clock::now();
    basic_solution(N, K, reference);
    auto end = chrono::high_resolution_clock::now();
    cout << "Reference:  " << chrono::duration_cast<chrono::milliseconds>(end-begin).count() << "ms" << endl;
    // print_grid(N, reference);

    // Run the blocked version
    float* blocked = initialize_grid(N, -100, 100);    
    float* blocked_transposed = transpose(N, blocked);
    begin = chrono::high_resolution_clock::now();
    blocked_solution<B>(N, K, blocked, blocked_transposed);
    end = chrono::high_resolution_clock::now();
    cout << "Blocked:    " << chrono::duration_cast<chrono::milliseconds>(end-begin).count() << "ms" << endl;
    // print_grid(N, blocked);
    assert(test_grids(N, reference, blocked));

    delete[] blocked_transposed;
    delete[] blocked;

    /***** Run SIMD version *****/
    float* vec_blocked_vectorized = initialize_grid(N, -100, 100);
    float* vec_blocked_vectorized_transposed = transpose(N, vec_blocked_vectorized);
    begin = chrono::high_resolution_clock::now();
    blocked_simd<B>(N, K, vec_blocked_vectorized, vec_blocked_vectorized_transposed);
    end = chrono::high_resolution_clock::now();
    cout << "SIMD: " << chrono::duration_cast<chrono::milliseconds>(end-begin).count() << "ms" << endl;
    assert(test_grids(N, reference, vec_blocked_vectorized));
    // print_grid(N, vec_blocked_vectorized);
    delete[] vec_blocked_vectorized_transposed; 
    delete[] vec_blocked_vectorized;

    /***** Run OMP version *****/
    float* vec_blocked_vectorized_multithreaded = initialize_grid(N, -100, 100);
    float* vec_blocked_vectorized_transposed_multithreaded = transpose(N, vec_blocked_vectorized_multithreaded);
    begin = chrono::high_resolution_clock::now();
    blocked_simd_omp<B>(N, K, vec_blocked_vectorized_multithreaded, vec_blocked_vectorized_transposed_multithreaded);
    end = chrono::high_resolution_clock::now();
    cout << "SIMD+OMP: " << chrono::duration_cast<chrono::milliseconds>(end-begin).count() << "ms" << endl;
    assert(test_grids(N, reference, vec_blocked_vectorized_multithreaded));
    delete[] vec_blocked_vectorized_transposed_multithreaded; 
    delete[] vec_blocked_vectorized_multithreaded;
    
    // Free memory
    delete[] reference;

 
    return 0;
}
