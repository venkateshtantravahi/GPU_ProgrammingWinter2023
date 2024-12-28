#include <iostream>
#include <sys/time.h>
#include <cuda_runtime.h>

// Global variables for timing
struct timeval startTime, endTime;

// Start timer function
void startTimer() {
    gettimeofday(&startTime, NULL);
}

// End timer function
void endTimer() {
    gettimeofday(&endTime, NULL);
}

// Print time function
void printTime() {
    long seconds = endTime.tv_sec - startTime.tv_sec;
    long micros = (seconds * 1000000) + endTime.tv_usec - startTime.tv_usec;
    // Adjust for proper microsecond calculation
    if (endTime.tv_usec < startTime.tv_usec) {
        seconds -= 1;
        micros = (seconds * 1000000) + (1000000 + endTime.tv_usec - startTime.tv_usec);
    }
    printf("Time elapsed is %ld seconds and %ld micros\n", seconds, micros % 1000000);
}

// A simple CUDA kernel for demonstration
__global__ void dummyKernel() {
    int idx = threadIdx.x;
}

int main() {
    // Initialize CUDA device
    cudaSetDevice(0);

    // Start the timer
    startTimer();

    // Launch the kernel
    dummyKernel<<<1, 256>>>();

    // Check for errors in kernel launch
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        // Error handling with a descriptive message
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    // Wait for CUDA to finish
    cudaDeviceSynchronize();

    // End the timer
    endTimer();

    // Print the elapsed time
    printTime();

    return 0;
}
