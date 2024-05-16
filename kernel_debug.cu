#include <iostream>
#include <cuda_runtime.h>

__global__ void myKernel(int* d_array, int arraySize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < arraySize) {
        d_array[idx] += 1;
        //debugger does not flush the CUDA output buffer on every step 
        //but rather after the completion of the kernel or when the buffer is full
        printf("d_array[%d] = %d\n", idx, d_array[idx]);
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int arraySize = 10;  // Array size definition
    int h_array[arraySize] = {0};

    int* d_array;
    checkCudaError(cudaMalloc((void**)&d_array, arraySize * sizeof(int)), "Failed to allocate device memory");
    checkCudaError(cudaMemcpy(d_array, h_array, arraySize * sizeof(int), cudaMemcpyHostToDevice), "Failed to copy memory to device");

    int threadsPerBlock = 5; // Number of threads per block
    int numBlocks = (arraySize + threadsPerBlock - 1) / threadsPerBlock; // Number of blocks

    myKernel<<<numBlocks, threadsPerBlock>>>(d_array, arraySize); // Passing arraySize to kernel
    cudaDeviceSynchronize();        // Synchronize the device

    checkCudaError(cudaGetLastError(), "Kernel launch failed");
    checkCudaError(cudaDeviceSynchronize(), "Failed to synchronize device");
    checkCudaError(cudaMemcpy(h_array, d_array, arraySize * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy memory to host");
    checkCudaError(cudaFree(d_array), "Failed to free device memory");

    // you have to use std::endl let it print out immediately 
    for (int i = 0; i < arraySize; ++i) {
        std::cout << h_array[i] + i << std::endl;
    }
    // std::cout << std::endl;

    return 0;
}
