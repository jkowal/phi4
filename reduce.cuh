#ifndef _REDUCE_H_
#define _REDUCE_H_

#pragma once

template <class T>
struct SharedMemory {
  __device__ inline operator T*() {
    extern __shared__ int __smem[];
    return (T*)__smem;
  }

  __device__ inline operator const T*() const {
    extern __shared__ int __smem[];
    return (T*)__smem;
  }
};

template <class T>
__global__ void reduce2(T* g_idata, float* g_odata, unsigned int n) {
  float* sdata = SharedMemory<float>();

  // load shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (i < n) ? (g_idata[i].vector[0] * g_idata[i].vector[0]) : 0;

  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void reduce6(T* g_idata, T* g_odata, unsigned int n) {
  T* sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
  unsigned int gridSize = blockSize * 2 * gridDim.x;

  T mySum = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n) {
    mySum += g_idata[i];
    // ensure we don't read out of bounds -- this is optimized away for powerOf2
    // sized arrays
    if (nIsPow2 || i + blockSize < n) mySum += g_idata[i + blockSize];
    i += gridSize;
  }

  // each thread puts its local sum into shared memory
  sdata[tid] = mySum;
  __syncthreads();

  // do reduction in shared mem
  if (blockSize >= 512) {
    if (tid < 256) {
      sdata[tid] = mySum = mySum + sdata[tid + 256];
    }
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (tid < 128) {
      sdata[tid] = mySum = mySum + sdata[tid + 128];
    }
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (tid < 64) {
      sdata[tid] = mySum = mySum + sdata[tid + 64];
    }
    __syncthreads();
  }

  if (tid < 32) {
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    volatile T* smem = sdata;
    if (blockSize >= 64) {
      smem[tid] = mySum = mySum + smem[tid + 32];
    }
    if (blockSize >= 32) {
      smem[tid] = mySum = mySum + smem[tid + 16];
    }
    if (blockSize >= 16) {
      smem[tid] = mySum = mySum + smem[tid + 8];
    }
    if (blockSize >= 8) {
      smem[tid] = mySum = mySum + smem[tid + 4];
    }
    if (blockSize >= 4) {
      smem[tid] = mySum = mySum + smem[tid + 2];
    }
    if (blockSize >= 2) {
      smem[tid] = mySum = mySum + smem[tid + 1];
    }
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

#endif  // #ifndef _PHI_KERNEL_H_
