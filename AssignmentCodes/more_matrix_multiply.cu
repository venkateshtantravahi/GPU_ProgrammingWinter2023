#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <sys/time.h>

#define MATRIX_DIM 1024  // Define a constant for the dimensions of the square matrix

/**
 * CUDA kernel to copy a matrix using pitched pointers.
 * @param source Pointer to the source matrix.
 * @param source_pitch Pitch of the source matrix.
 * @param destination Pointer to the destination matrix.
 * @param destination_pitch Pitch of the destination matrix.
 * @param width Width of the matrix.
 * @param height Height of the matrix.
 */
__global__ void CopyMatrix(int *source, size_t source_pitch, int *destination, size_t destination_pitch, int width, int height) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (column < width && row < height) {
        int* src_row = (int*)((char*)source + row * source_pitch);
        int* dest_row = (int*)((char*)destination + row * destination_pitch);
        dest_row[column] = src_row[column];
    }
}

/**
 * CUDA kernel for matrix multiplication using shared memory to enhance data reuse.
 * @param matrixA Device pointer to the first matrix.
 * @param matrixB Device pointer to the second matrix.
 * @param resultMatrix Device pointer for storing the result.
 * @param dim Dimension of the square matrices.
 */
__global__ void MultiplyMatrixShared(int *matrixA, int *matrixB, int *resultMatrix, int dim) {
    __shared__ int tileA[16][16];
    __shared__ int tileB[16][16];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * 16 + ty;
    int col = bx * 16 + tx;
    int temp = 0;
    int iterations = (dim + 15) / 16;

    for (int m = 0; m < iterations; ++m) {
        tileA[ty][tx] = matrixA[row * dim + (m * 16 + tx)];
        tileB[ty][tx] = matrixB[(m * 16 + ty) * dim + col];
        __syncthreads();

        for (int k = 0; k < 16; ++k) {
            temp += tileA[ty][k] * tileB[k][tx];
        }
        __syncthreads();
    }
    resultMatrix[row * dim + col] = temp;
}

// Timing functions using sys/time.h for high granularity time measurement
struct timeval start, end;

void startTimer() {
    gettimeofday(&start, NULL);
}

void endTimer(const char *label) {
    gettimeofday(&end, NULL);
    long seconds = end.tv_sec - start.tv_sec;
    long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    if (micros < 0) {
        micros += 1000000;
        seconds--;
    }
    printf("Time elapsed is %ld seconds and %ld microseconds\n", seconds, micros);
}

int main() {
    // Matrix dimensions
    const int width = MATRIX_DIM;
    const int height = MATRIX_DIM;

    // Host memory allocation
    int *h_matrix = (int *)malloc(width * height * sizeof(int));
    if (!h_matrix) {
        fprintf(stderr, "Failed to allocate host matrix\n");
        return EXIT_FAILURE;
    }

    // Initialize matrix
    for (int i = 0; i < width * height; ++i) {
        h_matrix[i] = i % 100;  // Simple initialization with values 0-99
    }

    // Device memory allocation
    int *d_matrixA, *d_matrixB, *d_matrixResult;
    size_t pitch;
    cudaMallocPitch(&d_matrixA, &pitch, width * sizeof(int), height);
    cudaMallocPitch(&d_matrixB, &pitch, width * sizeof(int), height);
    cudaMallocPitch(&d_matrixResult, &pitch, width * sizeof(int), height);

    // Copy matrices to the device
    cudaMemcpy2D(d_matrixA, pitch, h_matrix, width * sizeof(int), width * sizeof(int), height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_matrixB, pitch, h_matrix, width * sizeof(int), width * sizeof(int), height, cudaMemcpyHostToDevice);

    // Set up kernel execution configuration
    dim3 blocks(width / 16, height / 16);
    dim3 threads(16, 16);

    // Execute the kernel
    startTimer();
    CopyMatrix<<<blocks, threads>>>(d_matrixA, pitch, d_matrixB, pitch, width, height);
    cudaDeviceSynchronize();
    endTimer("Matrix Copy");

    // Copy back to verify
    cudaMemcpy2D(h_matrix, width * sizeof(int), d_matrixB, pitch, width * sizeof(int), height, cudaMemcpyDeviceToHost);
    printf("Sample elements of the copied matrix:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", h_matrix[i]);
    }
    printf("\n");

    startTimer();
    MultiplyMatrixShared<<<blocks, threads>>>(d_matrixA, d_matrixB, d_matrixResult, width);
    cudaDeviceSynchronize();
    endTimer("Matrix Multiplication");

    // Cleanup
    cudaFree(d_matrixA);
    cudaFree(d_matrixB);
    cudaFree(d_matrixResult);
    free(h_matrix);

    return 0;
}
