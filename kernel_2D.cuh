#ifndef _KERNEL_2D_H_
#define _KERNEL_2D_H_

#pragma once

#include "node.h"
#include "shmem_operations.cuh"
#include "prng.cuh"
#include "reduce.cuh"
#include "config.h"

__device__ float E(volatile data<> sdata[][NCOLUMNS], data<> node, float n,
                   float gamma, float lambda, unsigned int x, unsigned int y) {
  float c = 0.0f;
  float d = 0.0f;
  float corona = 0.0f;
  float newcorona = 0.0f;
  float sum = 0.0f;

  for (int i = 0; i < N; i++) {
    c += (sdata[x][y].vector[i] * sdata[x][y].vector[i]);

    d += (node.vector[i] * node.vector[i]);

    corona += (sdata[x + 1][y].vector[i] + sdata[x - 1][y].vector[i] +
               sdata[x][y + 1].vector[i] + sdata[x][y - 1].vector[i]);

    newcorona +=
        -lambda *
        ((sdata[x - 2][y].vector[i] + sdata[x + 2][y].vector[i] +
          sdata[x][y + 2].vector[i] + sdata[x][y - 2].vector[i]) -
         8.0f * corona +
         2.0f *
             (sdata[x - 1][y + 1].vector[i] + sdata[x + 1][y + 1].vector[i] +
              sdata[x - 1][y - 1].vector[i] + sdata[x + 1][y - 1].vector[i]));

    sum = -node.vector[i] + sdata[x][y].vector[i];
  }

  float e = (d - c) * (2.0f + 0.5f * n + 0.0416666f * gamma * (d + c) +
                       10.0f * lambda) +
            (sum) * (corona + newcorona);

  return e;
}

__device__ void shmem_metropolis_2D(data<> sdata[][SIZE_Y], unsigned int location,
                                    unsigned int* seed, unsigned int size,
                                    float n, float gamma, float lambda,
                                    unsigned int x, unsigned int y) {
	register data<> node;

  for (int i = 0; i < N; i++) {
    node.vector[i] = sdata[x][y].vector[i] +
                     0.4f * HybridTaus(seed[0], seed[1], seed[2], seed[3]) -
                     0.2f;
  }

  register float dE = E(sdata, node, n, gamma, lambda, x, y);

  if (HybridTaus(seed[0], seed[1], seed[2], seed[3]) < __expf(-dE)) {
    sdata[x][y] = node;
  }
}

__global__ void Phi_2D(data<>* g_idata, unsigned int* seed, int iteration,
                       unsigned int size, float n, float gamma, float lambda,
                       unsigned int offsetx, unsigned int offsety) {
  unsigned int location;
  int row, col;

  row = blockIdx.x * 4 * blockDim.x + 2 * threadIdx.x +
        offsetx * 16;  //+ 8*((blockIdx.x+offsetx)&1);// +
                       //((blockIdx.x+offsetx)&1)*size;

  col = 2 * blockIdx.y * 16 + 4 * threadIdx.y +
        2 * ((offsety)&1) * BLOCK;  // 2*((blockIdx.x+offsety)&1)*BLOCK;

  __shared__ data<> sdata[20][20];

  location = row * size + col;
  int second_location = (row + 1) * size + col;

  unsigned int register_seed[4];

  register_seed[0] = seed[4 * location];
  register_seed[1] = seed[4 * location + 1];
  register_seed[2] = seed[4 * location + 2];
  register_seed[3] = seed[4 * location + 3];

  globmem_to_shmem_2D(sdata, g_idata, row, col, location, second_location,
                      size);

  __syncthreads();

  int x = (2 * threadIdx.x) + 2;
  int z = (4 * threadIdx.y) + 2;

  int z1 = (((x)&3 ? z : z + 2));
  int z2 = (((x + 2) & 3 ? z + 2 : z));
  int z3 = (((x)&3 ? z + 2 : z));
  int z4 = (((x + 2) & 3 ? z : z + 2));

  for (int i = 0; i < LOCAL_SWEEPS; i++) {
    shmem_metropolis_2D(sdata, location, register_seed, size, n, gamma, lambda,
                        (2 * threadIdx.x) + 2, z);

    __syncthreads();

    shmem_metropolis_2D(sdata, location, register_seed, size, n, gamma, lambda,
                        (2 * threadIdx.x) + 3, z2);

    __syncthreads();

    shmem_metropolis_2D(sdata, location, register_seed, size, n, gamma, lambda,
                        (2 * threadIdx.x) + 2, z3);

    __syncthreads();

    shmem_metropolis_2D(sdata, location, register_seed, size, n, gamma, lambda,
                        (2 * threadIdx.x) + 3, z4);

    __syncthreads();

    shmem_metropolis_2D(sdata, location, register_seed, size, n, gamma, lambda,
                        (2 * threadIdx.x) + 2, z1 + 1);

    __syncthreads();

    shmem_metropolis_2D(sdata, location, register_seed, size, n, gamma, lambda,
                        (2 * threadIdx.x) + 3, z1 + 1);

    __syncthreads();

    shmem_metropolis_2D(sdata, location, register_seed, size, n, gamma, lambda,
                        (2 * threadIdx.x) + 2, z3 + 1);

    __syncthreads();

    shmem_metropolis_2D(sdata, location, register_seed, size, n, gamma, lambda,
                        (2 * threadIdx.x) + 3, z4 + 1);

    __syncthreads();
  }

  seed[4 * location] = register_seed[0];
  seed[4 * location + 1] = register_seed[1];
  seed[4 * location + 2] = register_seed[2];
  seed[4 * location + 3] = register_seed[3];

  g_idata[row * size + col] = sdata[2 * threadIdx.x + 2][z];
  g_idata[row * size + col + 1] = sdata[2 * threadIdx.x + 2][z + 1];
  g_idata[row * size + col + 2] = sdata[2 * threadIdx.x + 2][z + 2];
  g_idata[row * size + col + 3] = sdata[2 * threadIdx.x + 2][z + 3];
  g_idata[(row + 1) * size + col] = sdata[2 * threadIdx.x + 3][z];
  g_idata[(row + 1) * size + col + 1] = sdata[2 * threadIdx.x + 3][z + 1];
  g_idata[(row + 1) * size + col + 2] = sdata[2 * threadIdx.x + 3][z + 2];
  g_idata[(row + 1) * size + col + 3] = sdata[2 * threadIdx.x + 3][z + 3];
}

#endif
