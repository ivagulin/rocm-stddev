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


__device__ float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) 
        val += __shfl_down(val, offset);
    return val;
}


__device__ void blockReduceSum(const float* sdata, float* g_odata) {
    __shared__ float shared[32]; // Shared mem for 32 partial sums
    unsigned lane = threadIdx.x % warpSize;
    unsigned wid = threadIdx.x / warpSize;

    float val = warpReduceSum(sdata[threadIdx.x]);     // Each warp performs partial reduction

    if (lane==0) 
        shared[wid]=val; // Write reduced value to shared memory

    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if (wid==0) 
        val = warpReduceSum(val); //Final reduce within first warp

    if (threadIdx.x == 0)
        g_odata[blockIdx.x] = val;
}

const int blockNum = 1000*1000;
const int arraySize = 1024;

__global__ void computeStdDev(const float* g_idata, float* g_odata) {
    __shared__ float sdata[arraySize];
    //__shared__ float mean[blockNum];
    //__shared__ float sdata[1000];
    // float* sumBuf32 = localData;
    // float* mulBuf32 = localData+size;
    // float* sdata = mulBuf32+size;

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();
    blockReduceSum(sdata, g_odata);
    __syncthreads();
    sdata[tid] = g_idata[i] - (g_odata[blockIdx.x] / arraySize);
    sdata[tid] = sdata[tid] * sdata[tid];
    __syncthreads();
    blockReduceSum(sdata, g_odata);
}

int main() {
    float *h_input;
    float *h_output;

    h_input = new float[blockNum * arraySize];
    h_output = new float[blockNum];

    //prepare data
    for(unsigned i=0; i<blockNum; i++){
        // std::cout << "filling: " << i << std::endl;
        for(unsigned j=0; j<arraySize; j++){
            auto r = double(rand()) / (RAND_MAX + 1.) * double(i%100);
            h_input[i*arraySize+j] = r;
        }
        //std::fill(h_input + i*arraySize, h_input + i*arraySize  + arraySize, i%100);
    }

    // print input data
    // for(unsigned i=0; i<blockNum; i++){
    //     printf("block %d: ", i);
    //     for(unsigned j=0; j<arraySize; j++){
    //         printf("%.2f,", h_input[i*arraySize+j]);
    //     }
    //     printf("\n");
    // }

    float *d_input, *d_output;
    HIP_CHECK(hipMalloc(&d_input, blockNum * arraySize * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_output, blockNum * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_input, h_input, blockNum * arraySize * sizeof(float), hipMemcpyHostToDevice));
    for(unsigned i=0; i<10; i++){
        auto start = std::chrono::high_resolution_clock::now();
        computeStdDev<<<dim3(blockNum), dim3(arraySize), arraySize*3*sizeof(float)>>>(d_input, d_output);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipMemcpy(h_output, d_output, blockNum * sizeof(float), hipMemcpyDeviceToHost));
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "Время выполнения: " << duration.count() << " секунд" << std::endl;
    }

    // print results
    for(unsigned i=0; i<blockNum; i++) {
        // printf("stddev of %d: %0.2f\n", i, std::sqrt(h_output[i])/arraySize);
        // std::cout << "sum of " << i << ": " << h_output[i] << std::endl;
    }
}