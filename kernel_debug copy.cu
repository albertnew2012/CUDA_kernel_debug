#include <iostream>
#include <cuda_runtime.h> 

__global__ void myKernel(int* d_array) {
    int idx = threadIdx.x;
    d_array[idx] += 1;
    printf("d_array[%d] = %d\n", idx, d_array[idx]);
}

int main() {
    const int arraySize = 10;
    int h_array[arraySize] = {0};

    int* d_array;
    cudaMalloc((void**)&d_array, arraySize * sizeof(int));
    cudaMemcpy(d_array, h_array, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    myKernel<<<1, arraySize>>>(d_array);

    // Add synchronization to wait for the kernel to finish
    cudaDeviceSynchronize();

    cudaMemcpy(h_array, d_array, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    

    for (int i = 0; i < arraySize; ++i) {
        std::cout << h_array[i] << " ";
    }
    std::cout << std::endl;
    cudaFree(d_array);
    return 0;
}
