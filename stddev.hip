// MIT License
//
// Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#define __HIP__ 1
//#include "example_utils.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
#include <cstdlib>

/// \brief Checks if the provided error code is \p hipSuccess and if not,
/// prints an error message to the standard error output and terminates the program
/// with an error code.
#define HIP_CHECK(condition)                                                                \
    {                                                                                       \
        const hipError_t error = condition;                                                 \
        if(error != hipSuccess)                                                             \
        {                                                                                   \
            std::cerr << "An error encountered: \"" << hipGetErrorString(error) << "\" at " \
                      << __FILE__ << ':' << __LINE__ << std::endl;                          \
            std::exit(1);                                                     \
        }                                                                                   \
    }

__device__ void Sum(float* sdata, float* result, unsigned arraySize) {
    unsigned int tid = threadIdx.x;
    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        if ((tid % (2*s) == 0) && (tid+s < arraySize)) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }    
    // write result for this block to global mem
    if (tid == 0) {
        //printf("writing block %d: %0.2f\n", blockIdx.x, sdata[0]);
        result[blockIdx.x] = sdata[0];
    }
}

const int blockNum = 10; // * 1000;
const int arraySize = 10;

__global__ void computeStdDev(const float* g_idata, float* g_odata) {
    __shared__ float sdata[arraySize];
    __shared__ float mean[blockNum];
    //__shared__ float sdata[1000];
    // float* sumBuf32 = localData;
    // float* mulBuf32 = localData+size;
    // float* sdata = mulBuf32+size;

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid < arraySize)
        sdata[tid] = g_idata[i];
    __syncthreads();
    Sum(sdata, mean, arraySize);
    if (tid == 0)
        mean[blockIdx.x] = mean[blockIdx.x] / arraySize;
    __syncthreads();

    // if (tid == 0){
    //     g_odata[blockIdx.x] = mean[blockIdx.x];
    // }

    // std::copy(data, data+size, sumBuf32);
    // SubNumber(sumBuf32, cpuMean, size);
    // std::copy(sumBuf32, sumBuf32+size, mulBuf32);
    // Mul(mulBuf32, mulBuf32, size);
    // variance = Sum(mulBuf32, size);
    sdata[tid] = g_idata[i] - mean[blockIdx.x];
    sdata[tid] = sdata[tid] * sdata[tid];
    __syncthreads();
    Sum(sdata, g_odata, arraySize);
}

int main() {
    float *h_input;
    float *h_output;

    h_input = new float[blockNum * arraySize];
    h_output = new float[blockNum];

    for(unsigned i=0; i<blockNum; i++){
        std::cout << "filling: " << i << std::endl;
        for(unsigned j=0; j<arraySize; j++){
            auto r = double(rand()) / (RAND_MAX + 1.) * double(i%100);
            h_input[i*arraySize+j] = r;
        }
        //std::fill(h_input + i*arraySize, h_input + i*arraySize  + arraySize, i%100);
    }
    for(unsigned i=0; i<blockNum; i++){
        printf("block %d: ", i);
        for(unsigned j=0; j<arraySize; j++){
            printf("%.2f,", h_input[i*arraySize+j]);
        }
        printf("\n");
    }
    float *d_input, *d_output;
    HIP_CHECK(hipMalloc(&d_input, blockNum * arraySize * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_output, blockNum * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_input, h_input, blockNum * arraySize * sizeof(float), hipMemcpyHostToDevice));
    computeStdDev<<<dim3(blockNum), dim3(arraySize), arraySize*3*sizeof(float)>>>(d_input, d_output);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(h_output, d_output, blockNum * sizeof(float), hipMemcpyDeviceToHost));

    for(unsigned i=0; i<blockNum; i++) {
        printf("stddev of %d: %0.2f\n", i, std::sqrt(h_output[i])/arraySize);
        //std::cout << "sum of " << i << ": " << h_output[i] << std::endl;
    }
}