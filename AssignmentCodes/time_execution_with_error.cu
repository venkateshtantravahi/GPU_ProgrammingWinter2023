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

// Print time function with error message handling for performance issue
void printTime() {
    long seconds = endTime.tv_sec - startTime.tv_sec;
    long micros = (endTime.tv_usec < startTime.tv_usec) ? 
                  (seconds * 1000000 + endTime.tv_usec + 1000000 - startTime.tv_usec) % 1000000 : 
                  (seconds * 1000000 + endTime.tv_usec - startTime.tv_usec) % 1000000;

    // Error condition: Elapsed time exceeds 5 seconds
    if (seconds > 5) {
        std::cerr << "Error occurred: Operation took unexpectedly long (" << seconds << " seconds), indicating a potential performance issue." << std::endl;
    } else {
        printf("Time elapsed is %ld seconds and %ld micros\n", seconds, micros);
    }
}

// A simple CUDA kernel for demonstration
__global__ void dummyKernel() {
    // Dummy computation
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
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    // Wait for CUDA to finish
    cudaDeviceSynchronize();

    // End the timer
    endTimer();

    // Print the elapsed time or error message
    printTime();

    return 0;
}
