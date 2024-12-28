#include <iostream>
#include <cuda_runtime.h>

// Define a CUDA kernel function to print from the GPU, including thread and block information
__global__ void helloWorldKernel() {
    printf("Hello, World! from thread [%d,%d] in block [%d,%d]\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
}

int main() {
    // Launch the kernel with a single block of a single thread
    helloWorldKernel<<<1, 1>>>();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    // Wait for the GPU to finish before accessing on host
    cudaDeviceSynchronize();
    // Check for any errors during kernel execution or device synchronization
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        std::cerr << "CUDA device synchronization error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    // Print a message from the host
    std::cout << "Hello, World! from the host\n";

    return 0;
}
