#include <cuda_runtime.h>
#include <iostream>

#define TILE_WIDTH 16  // Set the tile width appropriate for your GPU's architecture

// Naive matrix multiplication kernel
__global__ void naiveMatrixMul(float* d_A, float* d_B, float* d_C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;

    if (row < width && col < width) {
        for (int k = 0; k < width; k++) {
            sum += d_A[row * width + k] * d_B[k * width + col];
        }
        d_C[row * width + col] = sum;
    }
}

// Optimized matrix multiplication using tiling and shared memory
__global__ void tiledMatrixMul(float* d_A, float* d_B, float* d_C, int width) {
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;

    for (int m = 0; m < ceil(width / (float)TILE_WIDTH); ++m) {
        if (m*TILE_WIDTH + threadIdx.x < width && row < width)
            tile_A[threadIdx.y][threadIdx.x] = d_A[row * width + m*TILE_WIDTH + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0;

        if (m*TILE_WIDTH + threadIdx.y < width && col < width)
            tile_B[threadIdx.y][threadIdx.x] = d_B[(m*TILE_WIDTH + threadIdx.y) * width + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < width && col < width) {
        d_C[row * width + col] = sum;
    }
}

int main() {
    int width = 1024;  // Size of the matrix
    float *d_A, *d_B, *d_C;
    size_t size = width * width * sizeof(float);

    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Initialize matrices and copy them to device memory
    // Assume matrices A and B are initialized on the host and copied to device

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (width + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Measure performance of naive kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float millisecondsRegular = 0;
    float millisecondsTiled = 0;

    cudaEventRecord(start);
    naiveMatrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&millisecondsRegular, start, stop);
    std::cout << "Naive kernel time (ms): " << millisecondsRegular << std::endl;

    // Measure performance of tiled kernel
    cudaEventRecord(start);
    tiledMatrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&millisecondsTiled, start, stop);
    std::cout << "Tiled kernel time (ms): " << millisecondsTiled << std::endl;

    std::cout << "Performance Comparison: " << (millisecondsRegular - millisecondsTiled) / millisecondsRegular * 100 << "%" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
