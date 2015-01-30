
#ifndef _KERNEL_3D_H_
#define _KERNEL_3D_H_

#pragma once

#include "node.h"
#include "shmem_operations.cuh"
#include "prng.cuh"
#include "reduce.cuh"
#include "config.h"

template <typename F = float>
__device__ F calculate_corona(data<> *gmem, data<> *sdata, data<> node, F n,
                              F gamma, F lambda, unsigned int x, unsigned int y,
                              unsigned int z, int size) {
  F corona = F(0);

  // calculate mesh corona for given (x,y,z) memory location
  for (size_t i = 0; i < N; ++i) {
    corona += (sdata[index3D(x + 1, y, z)].vector[i] +
               sdata[index3D(x - 1, y, z)].vector[i] +
               sdata[index3D(x, y + 1, z)].vector[i] +
               sdata[index3D(x, y - 1, z)].vector[i] +
               sdata[index3D(x, y, z + 1)].vector[i] +
               sdata[index3D(x, y, z - 1)].vector[i]);
  }
  return corona;
}

// calculte extended mesh corona for given (x,y,z) memory location
template <typename F = float>
__device__ F
calculate_extended_corona(data<> *gmem, data<> *sdata, data<> node, F n,
                          F gamma, F lambda, unsigned int x, unsigned int y,
                          unsigned int z, F corona, int size) {
  F extended_corona = F(0);

  for (size_t i = 0; i < N; ++i) {
    extended_corona =
        -lambda * (F(2) * (sdata[index3D(x - 1, y - 1, z)].vector[i] +
                           sdata[index3D(x - 1, y + 1, z)].vector[i] +
                           sdata[index3D(x + 1, y + 1, z)].vector[i] +
                           sdata[index3D(x + 1, y - 1, z)].vector[i] +
                           sdata[index3D(x, y - 1, z - 1)].vector[i] +
                           sdata[index3D(x, y + 1, z - 1)].vector[i] +
                           sdata[index3D(x, y - 1, z + 1)].vector[i] +
                           sdata[index3D(x, y + 1, z + 1)].vector[i] +
                           sdata[index3D(x - 1, y, z - 1)].vector[i] +
                           sdata[index3D(x + 1, y, z - 1)].vector[i] +
                           sdata[index3D(x - 1, y, z + 1)].vector[i] +
                           sdata[index3D(x + 1, y, z + 1)].vector[i]) -
                   F(12) * corona + (sdata[index3D(x + 2, y, z)].vector[i] +
                                     sdata[index3D(x - 2, y, z)].vector[i] +
                                     sdata[index3D(x, y + 2, z)].vector[i] +
                                     sdata[index3D(x, y - 2, z)].vector[i] +
                                     sdata[index3D(x, y, z + 2)].vector[i] +
                                     sdata[index3D(x, y, z - 2)].vector[i]));
  }
  return extended_corona;
}

// calculate energy change for given (x,y,z) memory location
template <typename F = float>
__device__ F
calculate_delta_e(data<> *gmem, data<> *sdata, data<> node, F n, F gamma,
                  F lambda, size_t x, size_t y, size_t z, F corona, F newcorona,
                  F _equation_element1, F _equation_element2, size_t size) {
  F c = F(0);
  F d = F(0);
  F sum = F(0);

  for (size_t i = 0; i < N; ++i) {
    c +=
        (sdata[index3D(x, y, z)].vector[i] * sdata[index3D(x, y, z)].vector[i]);

    d += (node.vector[i] * node.vector[i]);

    sum = -node.vector[i] + sdata[index3D(x, y, z)].vector[i];
  }

  return (d - c) * (_equation_element1 + _equation_element2 * (d + c)) +
         (sum) * (corona + newcorona);
}

// execute local Monte Carlo Metropolis Algorithm for given (x,y,z) memory
// location
template <typename F = float>
__device__ void shmem_metropolis_3D(data<> *gmem, data<> *sdata,
                                    size_t location, size_t *seed, size_t size,
                                    F n, F gamma, F lambda, F eq_el1, F eq_el2,
                                    size_t x, size_t y, size_t z) {

  data<> node;

  F corona =
      calculate_corona<>(gmem, sdata, node, n, gamma, lambda, x, y, z, size);
  F extended_corona = calculate_extended_corona<>(
      gmem, sdata, node, n, gamma, lambda, x, y, z, corona, size);

  for (size_t i = 0; i < 4; i++) {

    for (size_t i = 0; i < N; i++) {

      node.vector[i] = sdata[index3D(x, y, z)].vector[i] +
                       F(0.4) * HybridTaus(seed[0], seed[1], seed[2], seed[3]) -
                       F(0.2);
    }

    F energy_diff =
        calculate_delta_e<>(gmem, sdata, node, n, gamma, lambda, x, y, z,
                            corona, extended_corona, eq_el1, eq_el2, size);

    if (energy_diff < 0) {

      sdata[index3D(x, y, z)] = node;
    }

    if (HybridTaus(seed[0], seed[1], seed[2], seed[3]) < __expf(-energy_diff)) {

      sdata[index3D(x, y, z)] = node;
    }
  }
}

template <typename F = float>
__device__ void Metropolis(data<> *gmem, data<> *sdata, size_t location,
                           size_t *register_seed, size_t size, F n, F gamma,
                           F lambda, F eq_el1, F eq_el2, size_t x, size_t y,
                           size_t z, size_t y1, size_t y2) {

  // loop limit - global sweep limiter
  for (size_t i = 0; i < 50; ++i) {

#define metropolis_3D(x, y, z)                                                 \
  shmem_metropolis_3D<float>(gmem, sdata, location, register_seed, size, n,    \
                             gamma, lambda, eq_el1, eq_el2, x, y, z);          \
  __syncthreads();

    size_t even_index = 2 * threadIdx.x;
    size_t odd_index = 2 * threadIdx.x + 1;

    // z space
    metropolis_3D(even_index, y1, z);
    metropolis_3D(odd_index, y1, z);
    metropolis_3D(even_index, y2, z);
    metropolis_3D(odd_index, y2, z);
    metropolis_3D(even_index, y1 + 1, z);
    metropolis_3D(odd_index, y1 + 1, z);
    metropolis_3D(even_index, y2 + 1, z);
    metropolis_3D(odd_index, y2 + 1, z);
    // z+1 space
    metropolis_3D(even_index, y1, z + 1);
    metropolis_3D(odd_index, y1, z + 1);
    metropolis_3D(even_index, y2, z + 1);
    metropolis_3D(odd_index, y2, z + 1);
    metropolis_3D(even_index, y1 + 1, z + 1);
    metropolis_3D(odd_index, y1 + 1, z + 1);
    metropolis_3D(even_index, y2 + 1, z + 1);
    metropolis_3D(odd_index, y2 + 1, z + 1);
  }
}

// main kernel
template<typename F = float>
__global__ void Phi_3D(data<F> *g_idata, size_t *seed, size_t size, F n,
                       F gamma, F lambda, size_t offsetx,
                       size_t offsety, size_t offsetz) {

  // parameters
  size_t row = __mul24(2 * blockIdx.x, SHMEM_CUBE_SIZE) +
               __mul24(2, threadIdx.x) + __mul24(SHMEM_CUBE_SIZE, offsetx);

  size_t col = 2 * (blockIdx.y) * SHMEM_CUBE_SIZE + 4 * threadIdx.y +
               SHMEM_CUBE_SIZE * offsety;

  size_t dim = 2 * (blockIdx.z) * SHMEM_CUBE_SIZE + 2 * threadIdx.z +
               SHMEM_CUBE_SIZE * offsetz;

  __shared__ data<> sdata[SHMEM_MESH_SIZE * SHMEM_MESH_SIZE * SHMEM_MESH_SIZE];

  size_t location = row * size + col + dim * size * size;

  size_t register_seed[4];

  register_seed[0] = seed[4 * location];
  register_seed[1] = seed[4 * location + 1];
  register_seed[2] = seed[4 * location + 2];
  register_seed[3] = seed[4 * location + 3];

  size_t x = (2 * threadIdx.x);
  size_t y = (4 * threadIdx.y);
  size_t z = (2 * threadIdx.z);

  F eq1 = F(3) + F(0.5) * n + F(21) * lambda;
  F eq2 = F(0.0416666) * gamma;

  // copy global memory mesh buffer to shared memory buffer
  globmem_to_shmem_3D<F>(sdata, g_idata, x, y, z, row, col, dim, location, size);
  __syncthreads();

  // calculate shared memory strides
  size_t y1 = ((threadIdx.z & 1) ? (((x + 2) & 3 ? y : y + 2))
                                 : ((x + 2) & 3 ? y + 2 : y));
  size_t y2 = ((threadIdx.z & 1) ? (((x + 2) & 3 ? y + 2 : y))
                                 : (((x + 2) & 3 ? y : y + 2)));

  // execute Multi-Hit Monte Carlo algorithm
  Metropolis<F>(g_idata, sdata, location, register_seed, size, n, gamma, lambda,
               eq1, eq2, x, y, z, y1, y2);

  // save the shared memory buffer data to global memory buffer
  size_t first_line_index =
      (x + 2) * SHMEM_MESH_SIZE + (y + 2) + (z + 2) * SHM_STRIDE_Z;
  size_t second_line_index =
      (x + 3) * SHMEM_MESH_SIZE + (y + 2) + (z + 2) * SHM_STRIDE_Z;

  g_idata[location] = sdata[first_line_index];
  g_idata[location + 1] = sdata[first_line_index + 1];
  g_idata[location + 2] = sdata[first_line_index + 2];
  g_idata[location + 3] = sdata[first_line_index + 3];
  g_idata[(row + 1) * size + col + dim * size * size] =
      sdata[second_line_index];
  g_idata[(row + 1) * size + col + dim * size * size + 1] =
      sdata[second_line_index + 1];
  g_idata[(row + 1) * size + col + dim * size * size + 2] =
      sdata[second_line_index + 2];
  g_idata[(row + 1) * size + col + dim * size * size + 3] =
      sdata[second_line_index + 3];

  g_idata[location + size * size] = sdata[first_line_index + SHM_STRIDE_Z];
  g_idata[location + size * size + 1] =
      sdata[first_line_index + 1 + SHM_STRIDE_Z];
  g_idata[location + size * size + 2] =
      sdata[first_line_index + 2 + SHM_STRIDE_Z];
  g_idata[location + size * size + 3] =
      sdata[first_line_index + 3 + SHM_STRIDE_Z];
  g_idata[(row + 1) * size + col + dim * size * size + size * size] =
      sdata[second_line_index + SHM_STRIDE_Z];
  g_idata[(row + 1) * size + col + dim * size * size + size * size + 1] =
      sdata[second_line_index + 1 + SHM_STRIDE_Z];
  g_idata[(row + 1) * size + col + dim * size * size + size * size + 2] =
      sdata[second_line_index + 2 + SHM_STRIDE_Z];
  g_idata[(row + 1) * size + col + dim * size * size + size * size + 3] =
      sdata[second_line_index + 3 + SHM_STRIDE_Z];

  // save the prng register state to global memory address
  seed[4 * location] = register_seed[0];
  seed[4 * location + 1] = register_seed[1];
  seed[4 * location + 2] = register_seed[2];
  seed[4 * location + 3] = register_seed[3];
}

template <class T>
__global__ void correlation(T *g_idata, float *g_odata, unsigned int n,
                            int size, int R) {
  float *sdata = SharedMemory<float>();

  // load global memory buffer to shared memory buffer
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int x = (i / size);
  unsigned int y = i - (x * size);

  unsigned int x_stride = ((x - R) & (size - 1) * size + y);
  unsigned int y_stride = (x * size + (y + R) & (size - 1));

  sdata[tid] = (i < n) ? (g_idata[i].vector[0] * g_idata[x_stride].vector[0] +
                          (g_idata[i].vector[0] * g_idata[y_stride].vector[0]))
                       : 0;

  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // write result for this block to global mem
  if (tid == 0)
    g_odata[blockIdx.x] = sdata[0];
}

#endif
