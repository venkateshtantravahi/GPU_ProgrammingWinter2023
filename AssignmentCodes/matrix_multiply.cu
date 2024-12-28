#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define TILE_WIDTH 16

// CUDA kernel for matrix multiplication
__global__ void matrixMultiply(int *a, int *b, int *c, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int sum = 0;
    for (int k = 0; k < width; ++k) {
        sum += a[row * width + k] * b[k * width + col];
    }
    c[row * width + col] = sum;
}

// Timing functions
struct timeval start, end;

void starttimer() {
    gettimeofday(&start, NULL);
}

void endtimer() {
    gettimeofday(&end, NULL);
}

void printtime() {
    long seconds = end.tv_sec - start.tv_sec;
    long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);

    // Check if micros is negative, adjust seconds accordingly
    if (micros < 0) {
        micros += 1000000; // Adjusting microseconds
        seconds--; // Adjusting seconds
    }

    printf("Time elapsed is %ld seconds and %ld microseconds\n", seconds, micros);
}

int main() {
    int width = 1024; // Matrix width
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;

    // Allocate memory for matrices on host
    a = (int *)malloc(width * width * sizeof(int));
    b = (int *)malloc(width * width * sizeof(int));
    c = (int *)malloc(width * width * sizeof(int));

    // Initialize matrices on host
    for (int i = 0; i < width * width; ++i) {
        a[i] = rand() % 10;
        b[i] = rand() % 10;
    }

    // Allocate memory for matrices on device
    cudaMalloc((void **)&d_a, width * width * sizeof(int));
    cudaMalloc((void **)&d_b, width * width * sizeof(int));
    cudaMalloc((void **)&d_c, width * width * sizeof(int));

    // Copy matrices from host to device
    cudaMemcpy(d_a, a, width * width * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, width * width * sizeof(int), cudaMemcpyHostToDevice);

    // Loop over different configurations of blocks and threads
    for (int numBlocks = 1; numBlocks <= 512; numBlocks *= 2) {
        for (int threadsPerBlock = 32; threadsPerBlock <= 1024; threadsPerBlock *= 2) {
            // Start timer
            starttimer();

            // Launch kernel with current configuration
            dim3 dimGrid(width / TILE_WIDTH, width / TILE_WIDTH, 1);
            dim3 dimBlock(threadsPerBlock, threadsPerBlock, 1);
            matrixMultiply<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, width);

            // End timer
            endtimer();

            // Copy result matrix from device to host
            cudaMemcpy(c, d_c, width * width * sizeof(int), cudaMemcpyDeviceToHost);

            // Measure and print execution time
            printtime();

            // Output configuration and execution time
            printf("Configuration: Blocks=%d, ThreadsPerBlock=%d\n", numBlocks, threadsPerBlock);
        }
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(a);
    free(b);
    free(c);

    return 0;
}