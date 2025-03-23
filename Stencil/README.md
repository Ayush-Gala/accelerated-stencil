# Stencil Optimization

This project implements and optimizes a local mean stencil computation with various performance optimization techniques including blocking, SIMD instructions, and OpenMP parallelization.

## Overview

The stencil computation is a fundamental kernel used in physical simulations, scientific computing, and machine learning. This implementation focuses on a cross-shaped stencil where for each point (x,y) in a grid, we compute the mean of neighboring points in horizontal and vertical directions within distance K.

For example, with K=2, each point uses neighbors that are up to 2 positions away in all four directions (north, south, east, west).

## Problem Description

For each grid point (x,y), we compute the average of its neighbors in the "cross" pattern:
- Horizontal neighbors: (x, y-K), (x, y-K+1), ..., (x, y-1), (x, y+1), ..., (x, y+K)
- Vertical neighbors: (x-K, y), (x-K+1, y), ..., (x-1, y), (x+1, y), ..., (x+K, y)

The center point itself (x,y) is not included in the calculation of the mean.

## Implementation Versions

### 1. Basic Solution
- A straightforward implementation with nested loops
- Uses temporary storage to avoid in-place updates that would affect subsequent calculations
- Simple to understand but not optimized for performance

### 2. Blocked Solution
- Introduces blocking to improve cache utilization
- Processes the grid in blocks of size B×B
- Uses a transposed version of the input grid to improve memory access patterns for vertical neighbors

### 3. SIMD Implementation
- Leverages SSE instructions to process multiple elements at once
- Utilizes 128-bit vectors to compute 4 floating-point operations simultaneously
- Handles edge cases and alignment requirements carefully
- Significantly improves computation speed by exploiting data-level parallelism

### 4. OpenMP + SIMD Implementation
- Combines SIMD with multi-threading using OpenMP
- Parallelizes the outer loops to distribute work across multiple cores
- Uses dynamic scheduling for load balancing
- Each thread works with its own temporary storage to avoid synchronization overhead

## Optimizations

### Memory Access
- The algorithm uses a transposed version of the grid to ensure consecutive memory accesses when reading vertical neighbors
- Blocking improves temporal and spatial locality of memory accesses

### SIMD Vectorization
- SSE instructions process 4 floating-point values in parallel
- Handles both aligned and unaligned memory accesses
- Separate scalar processing for edge cases that don't fit into SIMD width

### Parallelization
- OpenMP distributes blocks across available threads
- Dynamic scheduling for better load balancing
- Thread-local temporary arrays avoid false sharing and synchronization overhead
- Separate parallelization strategies for computation and write-back phases

## Performance Comparison

The program automatically runs all four implementations and measures their performance:
1. Reference implementation (basic solution)
2. Blocked implementation
3. SIMD-optimized implementation
4. Combined SIMD + OpenMP implementation

You can compare the timing results to see the speedup achieved by each optimization.

## Building and Running

### Prerequisites
- C++ compiler with C++2a support
- SSE4.2 instruction set support
- OpenMP support

### Compilation
```bash
make
```

### Running
```bash
# Run with default parameters (N=1024, K=8)
./stencil

# Run with custom parameters
./stencil [N] [K]
```

Where:
- N is the size of the grid (N×N)
- K is the range of neighbors to include

### Cleaning
```bash
make clean
```

## Implementation Details

### SIMD Optimization
The SIMD implementation (`blocked_simd`) uses SSE intrinsics to vectorize the computation:

1. For each point (x,y), it vectorizes:
   - Horizontal neighbor summation using 4-wide SIMD operations
   - Vertical neighbor summation using the transposed grid for sequential access
   - Final accumulation and mean calculation

2. Special handling:
   - Non-aligned memory accesses at boundaries
   - Careful treatment of the center point (subtracted from the sum)
   - Vectorized write-back phase

### OpenMP Parallelization
The OpenMP implementation (`blocked_simd_omp`) extends the SIMD version with multi-threading:

1. Distributes block processing across threads:
   - Uses dynamic scheduling for better load balancing
   - Each thread maintains a private temporary array

2. Synchronization strategy:
   - Minimizes synchronization by using thread-local storage
   - Two-phase approach with separate parallelization for computation and write-back
   - Static scheduling for the write-back phase

## Performance Analysis

Typical performance improvements observed:

1. Blocked implementation: Improves cache behavior compared to the basic solution
2. SIMD implementation: Provides ~4x speedup over the blocked version (theoretical maximum with 4-wide vectors)
3. OpenMP implementation: Linear scaling with the number of cores/threads available

Note that actual performance gains depend on hardware capabilities, problem size, and compiler optimizations.

## Future Improvements

Possible enhancements to further optimize performance:
- AVX/AVX2/AVX-512 support for wider SIMD vectors (8/16 elements at once)
- More sophisticated blocking strategies (e.g., recursive blocking)
- Exploration of different OpenMP scheduling policies for specific problem sizes
- Auto-tuning of block size B based on cache characteristics