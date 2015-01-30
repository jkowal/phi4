#ifndef _SHMEM_OPERATIONS_H_
#define _SHMEM_OPERATIONS_H_

#pragma once

#include "node.h"
#include "config.h"

__device__ void globmem_to_shmem_2D(data<> sdata[][NCOLUMNS], data<>* g_idata,
                                    int row, int col, int location,
                                    int second_location, int size) {
  int y = 4 * (threadIdx.y) + 2;

  sdata[2 * threadIdx.x + 2][y] = g_idata[location];
  sdata[2 * threadIdx.x + 2][y + 1] = g_idata[location + 1];
  sdata[2 * threadIdx.x + 2][y + 2] = g_idata[location + 2];
  sdata[2 * threadIdx.x + 2][y + 3] = g_idata[location + 3];

  sdata[2 * threadIdx.x + 3][y] = g_idata[second_location];
  sdata[2 * threadIdx.x + 3][y + 1] = g_idata[second_location + 1];
  sdata[2 * threadIdx.x + 3][y + 2] = g_idata[second_location + 2];
  sdata[2 * threadIdx.x + 3][y + 3] = g_idata[second_location + 3];

  if (threadIdx.y == 0) {
    sdata[(2 * threadIdx.x + 2)][0] =
        g_idata[(row * size + ((col - 2 + size) & size - 1))];
    sdata[(2 * threadIdx.x + 2)][1] =
        g_idata[(row * size + ((col - 1 + size) & size - 1))];

    sdata[(2 * threadIdx.x + 3)][0] =
        g_idata[((row + 1) * size + ((col - 2 + size) & size - 1))];
    sdata[(2 * threadIdx.x + 3)][1] =
        g_idata[((row + 1) * size) + ((col - 1 + size) & size - 1)];
  }

  // prawa krawedz 4 elementy
  if (threadIdx.y == (blockDim.y - 1)) {
    sdata[2 * threadIdx.x + 2][19] =
        g_idata[((row * size + ((col + 3 + 2 + size) & size - 1)))];
    sdata[2 * threadIdx.x + 2][18] =
        g_idata[((row * size + ((col + 3 + 1 + size) & size - 1)))];

    sdata[2 * threadIdx.x + 3][19] =
        g_idata[(((row + 1) * size + ((col + 3 + 2 + size) & size - 1)))];
    sdata[2 * threadIdx.x + 3][18] =
        g_idata[(((row + 1) * size + ((col + 3 + 1 + size) & size - 1)))];
  }

  // gorna krawedz 8 elementow
  if (threadIdx.x == 0) {
    int up2 = ((row - 2 + size) & size - 1) * size + col;
    int up1 = ((row - 1 + size) & size - 1) * size + col;

    sdata[threadIdx.x][y] = g_idata[up2];
    sdata[threadIdx.x][y + 1] = g_idata[up2 + 1];
    sdata[threadIdx.x][y + 2] = g_idata[up2 + 2];
    sdata[threadIdx.x][y + 3] = g_idata[up2 + 3];

    sdata[threadIdx.x + 1][y] = g_idata[up1];
    sdata[threadIdx.x + 1][y + 1] = g_idata[up1 + 1];
    sdata[threadIdx.x + 1][y + 2] = g_idata[up1 + 2];
    sdata[threadIdx.x + 1][y + 3] = g_idata[up1 + 3];
  }

  // dolna krawedz 8 elementow
  if (threadIdx.x == (blockDim.x - 1)) {
    int down1 = ((row + 2 + size) & size - 1) * size + col;
    int down2 = ((row + 3 + size) & size - 1) * size + col;

    sdata[2 * threadIdx.x + 4][y] = g_idata[down1];
    sdata[2 * threadIdx.x + 4][y + 1] = g_idata[down1 + 1];
    sdata[2 * threadIdx.x + 4][y + 2] = g_idata[down1 + 2];
    sdata[2 * threadIdx.x + 4][y + 3] = g_idata[down1 + 3];

    sdata[2 * threadIdx.x + 5][y] = g_idata[down2];
    sdata[2 * threadIdx.x + 5][y + 1] = g_idata[down2 + 1];
    sdata[2 * threadIdx.x + 5][y + 2] = g_idata[down2 + 2];
    sdata[2 * threadIdx.x + 5][y + 3] = g_idata[down2 + 3];
  }

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    sdata[(2 * threadIdx.x + 1)][threadIdx.y + 1] =
        g_idata[(__mul24((row - 1 + size & (size - 1)), size) +
                 ((col - 1 + size) & size - 1))];
  }
  if (threadIdx.x == 0 && threadIdx.y == (blockDim.y - 1)) {
    sdata[(2 * threadIdx.x + 1)][y + 4] =
        g_idata[(__mul24((row - 1 + size & (size - 1)), size) +
                 ((col + 4 + size) & size - 1))];
  }
  if (threadIdx.x == (blockDim.x - 1) && threadIdx.y == 0) {
    sdata[(2 * threadIdx.x + 4)][threadIdx.y + 1] =
        g_idata[(__mul24((row + 2 + size & (size - 1)), size) +
                 ((col - 1 + size) & size - 1))];
  }
  if (threadIdx.x == (blockDim.x - 1) && threadIdx.y == (blockDim.y - 1)) {
    sdata[(2 * threadIdx.x + 4)][y + 4] =
        g_idata[(__mul24((row + 2 + size & (size - 1)), size) +
                 ((col + 4 + size) & size - 1))];
  }
}

__device__ inline int index3D(int x, int y, int z) {
  int index =
      ((x + 2) * (4 + SHMEM_CUBE_SIZE)) + ((z + 2) * SHM_STRIDE_Z) + (y + 2);

  return index;
}

__device__ inline int gmem_index3D(int x, int y, int z, int size) {
  int index = x * size + y + z * size * size;

  return index;
}

template<typename F = float>
__device__ void globmem_to_shmem_3D(data<F>* sdata, data<F>* g_idata, int x, int y,
                                    int z, int row, int col, int dim,
                                    int location, int size) {
  sdata[index3D(x, y, z)] = g_idata[gmem_index3D(row, col, dim, size)];
  sdata[index3D(x, y + 1, z)] = g_idata[gmem_index3D(row, col + 1, dim, size)];
  sdata[index3D(x, y + 2, z)] = g_idata[gmem_index3D(row, col + 2, dim, size)];
  sdata[index3D(x, y + 3, z)] = g_idata[gmem_index3D(row, col + 3, dim, size)];
  sdata[index3D(x + 1, y, z)] = g_idata[gmem_index3D(row + 1, col, dim, size)];
  sdata[index3D(x + 1, y + 1, z)] =
      g_idata[gmem_index3D(row + 1, col + 1, dim, size)];
  sdata[index3D(x + 1, y + 2, z)] =
      g_idata[gmem_index3D(row + 1, col + 2, dim, size)];
  sdata[index3D(x + 1, y + 3, z)] =
      g_idata[gmem_index3D(row + 1, col + 3, dim, size)];

  sdata[index3D(x, y, z + 1)] = g_idata[gmem_index3D(row, col, dim + 1, size)];
  sdata[index3D(x, y + 1, z + 1)] =
      g_idata[gmem_index3D(row, col + 1, dim + 1, size)];
  sdata[index3D(x, y + 2, z + 1)] =
      g_idata[gmem_index3D(row, col + 2, dim + 1, size)];
  sdata[index3D(x, y + 3, z + 1)] =
      g_idata[gmem_index3D(row, col + 3, dim + 1, size)];
  sdata[index3D(x + 1, y, z + 1)] =
      g_idata[gmem_index3D(row + 1, col, dim + 1, size)];
  sdata[index3D(x + 1, y + 1, z + 1)] =
      g_idata[gmem_index3D(row + 1, col + 1, dim + 1, size)];
  sdata[index3D(x + 1, y + 2, z + 1)] =
      g_idata[gmem_index3D(row + 1, col + 2, dim + 1, size)];
  sdata[index3D(x + 1, y + 3, z + 1)] =
      g_idata[gmem_index3D(row + 1, col + 3, dim + 1, size)];

  if (threadIdx.y == 0) {
    sdata[index3D(x, -2, z)] =
        g_idata[gmem_index3D(row, (col - 2) & (size - 1), dim, size)];
    sdata[index3D(x, -1, z)] =
        g_idata[gmem_index3D(row, (col - 1) & (size - 1), dim, size)];
    sdata[index3D(x + 1, -2, z)] =
        g_idata[gmem_index3D(row + 1, (col - 2) & (size - 1), dim, size)];
    sdata[index3D(x + 1, -1, z)] =
        g_idata[gmem_index3D(row + 1, (col - 1) & (size - 1), dim, size)];

    sdata[index3D(x, -2, z + 1)] =
        g_idata[gmem_index3D(row, (col - 2) & (size - 1), dim + 1, size)];
    sdata[index3D(x, -1, z + 1)] =
        g_idata[gmem_index3D(row, (col - 1) & (size - 1), dim + 1, size)];
    sdata[index3D(x + 1, -2, z + 1)] =
        g_idata[gmem_index3D(row + 1, (col - 2) & (size - 1), dim + 1, size)];
    sdata[index3D(x + 1, -1, z + 1)] =
        g_idata[gmem_index3D(row + 1, (col - 1) & (size - 1), dim + 1, size)];
  }

  if (threadIdx.y == (blockDim.y - 1)) {
    sdata[index3D(x, SHMEM_CUBE_SIZE, z)] =
        g_idata[gmem_index3D(row, (col + 4) & (size - 1), dim, size)];
    sdata[index3D(x, SHMEM_CUBE_SIZE + 1, z)] =
        g_idata[gmem_index3D(row, (col + 5) & (size - 1), dim, size)];
    sdata[index3D(x + 1, SHMEM_CUBE_SIZE, z)] =
        g_idata[gmem_index3D(row + 1, (col + 4) & (size - 1), dim, size)];
    sdata[index3D(x + 1, SHMEM_CUBE_SIZE + 1, z)] =
        g_idata[gmem_index3D(row + 1, (col + 5) & (size - 1), dim, size)];
    sdata[index3D(x, SHMEM_CUBE_SIZE, z + 1)] =
        g_idata[gmem_index3D(row, (col + 4) & (size - 1), dim + 1, size)];
    sdata[index3D(x, SHMEM_CUBE_SIZE + 1, z + 1)] =
        g_idata[gmem_index3D(row, (col + 5) & (size - 1), dim + 1, size)];
    sdata[index3D(x + 1, SHMEM_CUBE_SIZE, z + 1)] =
        g_idata[gmem_index3D(row + 1, (col + 4) & (size - 1), dim + 1, size)];
    sdata[index3D(x + 1, SHMEM_CUBE_SIZE + 1, z + 1)] =
        g_idata[gmem_index3D(row + 1, (col + 5) & (size - 1), dim + 1, size)];
  }

  if (threadIdx.x == 0) {
    sdata[index3D(-2, y, z)] =
        g_idata[gmem_index3D((row - 2) & (size - 1), col, dim, size)];
    sdata[index3D(-2, y + 1, z)] =
        g_idata[gmem_index3D((row - 2) & (size - 1), col + 1, dim, size)];
    sdata[index3D(-2, y + 2, z)] =
        g_idata[gmem_index3D((row - 2) & (size - 1), col + 2, dim, size)];
    sdata[index3D(-2, y + 3, z)] =
        g_idata[gmem_index3D((row - 2) & (size - 1), col + 3, dim, size)];

    sdata[index3D(-1, y, z)] =
        g_idata[gmem_index3D((row - 1) & (size - 1), col, dim, size)];
    sdata[index3D(-1, y + 1, z)] =
        g_idata[gmem_index3D((row - 1) & (size - 1), col + 1, dim, size)];
    sdata[index3D(-1, y + 2, z)] =
        g_idata[gmem_index3D((row - 1) & (size - 1), col + 2, dim, size)];
    sdata[index3D(-1, y + 3, z)] =
        g_idata[gmem_index3D((row - 1) & (size - 1), col + 3, dim, size)];

    sdata[index3D(-2, y, z + 1)] =
        g_idata[gmem_index3D((row - 2) & (size - 1), col, dim + 1, size)];
    sdata[index3D(-2, y + 1, z + 1)] =
        g_idata[gmem_index3D((row - 2) & (size - 1), col + 1, dim + 1, size)];
    sdata[index3D(-2, y + 2, z + 1)] =
        g_idata[gmem_index3D((row - 2) & (size - 1), col + 2, dim + 1, size)];
    sdata[index3D(-2, y + 3, z + 1)] =
        g_idata[gmem_index3D((row - 2) & (size - 1), col + 3, dim + 1, size)];

    sdata[index3D(-1, y, z + 1)] =
        g_idata[gmem_index3D((row - 1) & (size - 1), col, dim + 1, size)];
    sdata[index3D(-1, y + 1, z + 1)] =
        g_idata[gmem_index3D((row - 1) & (size - 1), col + 1, dim + 1, size)];
    sdata[index3D(-1, y + 2, z + 1)] =
        g_idata[gmem_index3D((row - 1) & (size - 1), col + 2, dim + 1, size)];
    sdata[index3D(-1, y + 3, z + 1)] =
        g_idata[gmem_index3D((row - 1) & (size - 1), col + 3, dim + 1, size)];
  }

  if (threadIdx.x == (blockDim.x - 1)) {
    // even
    sdata[((2 * threadIdx.x + 4) * (4 + SHMEM_CUBE_SIZE) + y + 2 +
           (z + 2) * SHM_STRIDE_Z)] =
        g_idata[gmem_index3D((row + 2) & (size - 1), col, dim, size)];
    sdata[((2 * threadIdx.x + 4) * (4 + SHMEM_CUBE_SIZE) + y + 2 + 1 +
           (z + 2) * SHM_STRIDE_Z)] =
        g_idata[gmem_index3D((row + 2) & (size - 1), col + 1, dim, size)];
    sdata[((2 * threadIdx.x + 4) * (4 + SHMEM_CUBE_SIZE) + y + 2 + 2 +
           (z + 2) * SHM_STRIDE_Z)] =
        g_idata[gmem_index3D((row + 2) & (size - 1), col + 2, dim, size)];
    sdata[((2 * threadIdx.x + 4) * (4 + SHMEM_CUBE_SIZE) + y + 2 + 3 +
           (z + 2) * SHM_STRIDE_Z)] =
        g_idata[gmem_index3D((row + 2) & (size - 1), col + 3, dim, size)];

    sdata[((2 * threadIdx.x + 5) * (4 + SHMEM_CUBE_SIZE) + y + 2 +
           (z + 2) * SHM_STRIDE_Z)] =
        g_idata[gmem_index3D((row + 3) & (size - 1), col, dim, size)];
    sdata[((2 * threadIdx.x + 5) * (4 + SHMEM_CUBE_SIZE) + y + 2 + 1 +
           (z + 2) * SHM_STRIDE_Z)] =
        g_idata[gmem_index3D((row + 3) & (size - 1), col + 1, dim, size)];
    sdata[((2 * threadIdx.x + 5) * (4 + SHMEM_CUBE_SIZE) + y + 2 + 2 +
           (z + 2) * SHM_STRIDE_Z)] =
        g_idata[gmem_index3D((row + 3) & (size - 1), col + 2, dim, size)];
    sdata[((2 * threadIdx.x + 5) * (4 + SHMEM_CUBE_SIZE) + y + 2 + 3 +
           (z + 2) * SHM_STRIDE_Z)] =
        g_idata[gmem_index3D((row + 3) & (size - 1), col + 3, dim, size)];

    // odd
    sdata[((2 * threadIdx.x + 4) * (4 + SHMEM_CUBE_SIZE) + y + 2 +
           (z + 3) * SHM_STRIDE_Z)] =
        g_idata[gmem_index3D((row + 2) & (size - 1), col, dim + 1, size)];
    sdata[((2 * threadIdx.x + 4) * (4 + SHMEM_CUBE_SIZE) + y + 2 + 1 +
           (z + 3) * SHM_STRIDE_Z)] =
        g_idata[gmem_index3D((row + 2) & (size - 1), col + 1, dim + 1, size)];
    sdata[((2 * threadIdx.x + 4) * (4 + SHMEM_CUBE_SIZE) + y + 2 + 2 +
           (z + 3) * SHM_STRIDE_Z)] =
        g_idata[gmem_index3D((row + 2) & (size - 1), col + 2, dim + 1, size)];
    sdata[((2 * threadIdx.x + 4) * (4 + SHMEM_CUBE_SIZE) + y + 2 + 3 +
           (z + 3) * SHM_STRIDE_Z)] =
        g_idata[gmem_index3D((row + 2) & (size - 1), col + 3, dim + 1, size)];

    sdata[((2 * threadIdx.x + 5) * (4 + SHMEM_CUBE_SIZE) + y + 2 +
           (z + 3) * SHM_STRIDE_Z)] =
        g_idata[gmem_index3D((row + 3) & (size - 1), col, dim + 1, size)];
    sdata[((2 * threadIdx.x + 5) * (4 + SHMEM_CUBE_SIZE) + y + 2 + 1 +
           (z + 3) * SHM_STRIDE_Z)] =
        g_idata[gmem_index3D((row + 3) & (size - 1), col + 1, dim + 1, size)];
    sdata[((2 * threadIdx.x + 5) * (4 + SHMEM_CUBE_SIZE) + y + 2 + 2 +
           (z + 3) * SHM_STRIDE_Z)] =
        g_idata[gmem_index3D((row + 3) & (size - 1), col + 2, dim + 1, size)];
    sdata[((2 * threadIdx.x + 5) * (4 + SHMEM_CUBE_SIZE) + y + 2 + 3 +
           (z + 3) * SHM_STRIDE_Z)] =
        g_idata[gmem_index3D((row + 3) & (size - 1), col + 3, dim + 1, size)];
  }

  if (threadIdx.z == 0) {
    // even
    sdata[index3D(x, y, z - 1)] =
        g_idata[gmem_index3D(row, col, (dim - 1 + size & (size - 1)), size)];
    sdata[index3D(x, y + 1, z - 1)] = g_idata[gmem_index3D(
        row, col + 1, (dim - 1 + size & (size - 1)), size)];
    sdata[index3D(x, y + 2, z - 1)] = g_idata[gmem_index3D(
        row, col + 2, (dim - 1 + size & (size - 1)), size)];
    sdata[index3D(x, y + 3, z - 1)] = g_idata[gmem_index3D(
        row, col + 3, (dim - 1 + size & (size - 1)), size)];
    sdata[index3D(x + 1, y, z - 1)] = g_idata[gmem_index3D(
        row + 1, col, (dim - 1 + size & (size - 1)), size)];
    sdata[index3D(x + 1, y + 1, z - 1)] = g_idata[gmem_index3D(
        row + 1, col + 1, (dim - 1 + size & (size - 1)), size)];
    sdata[index3D(x + 1, y + 2, z - 1)] = g_idata[gmem_index3D(
        row + 1, col + 2, (dim - 1 + size & (size - 1)), size)];
    sdata[index3D(x + 1, y + 3, z - 1)] = g_idata[gmem_index3D(
        row + 1, col + 3, (dim - 1 + size & (size - 1)), size)];

    // odd
    sdata[index3D(x, y, z - 2)] =
        g_idata[gmem_index3D(row, col, (dim - 2 + size & (size - 1)), size)];
    sdata[index3D(x, y + 1, z - 2)] = g_idata[gmem_index3D(
        row, col + 1, (dim - 2 + size & (size - 1)), size)];
    sdata[index3D(x, y + 2, z - 2)] = g_idata[gmem_index3D(
        row, col + 2, (dim - 2 + size & (size - 1)), size)];
    sdata[index3D(x, y + 3, z - 2)] = g_idata[gmem_index3D(
        row, col + 3, (dim - 2 + size & (size - 1)), size)];
    sdata[index3D(x + 1, y, z - 2)] = g_idata[gmem_index3D(
        row + 1, col, (dim - 2 + size & (size - 1)), size)];
    sdata[index3D(x + 1, y + 1, z - 2)] = g_idata[gmem_index3D(
        row + 1, col + 1, (dim - 2 + size & (size - 1)), size)];
    sdata[index3D(x + 1, y + 2, z - 2)] = g_idata[gmem_index3D(
        row + 1, col + 2, (dim - 2 + size & (size - 1)), size)];
    sdata[index3D(x + 1, y + 3, z - 2)] = g_idata[gmem_index3D(
        row + 1, col + 3, (dim - 2 + size & (size - 1)), size)];
  }

  if (threadIdx.z == (blockDim.z - 1)) {
    // parzyste
    sdata[index3D(x, y, z + 2)] =
        g_idata[gmem_index3D(row, col, (dim + 2 + size & (size - 1)), size)];
    sdata[index3D(x, y + 1, z + 2)] = g_idata[gmem_index3D(
        row, col + 1, (dim + 2 + size & (size - 1)), size)];
    sdata[index3D(x, y + 2, z + 2)] = g_idata[gmem_index3D(
        row, col + 2, (dim + 2 + size & (size - 1)), size)];
    sdata[index3D(x, y + 3, z + 2)] = g_idata[gmem_index3D(
        row, col + 3, (dim + 2 + size & (size - 1)), size)];
    sdata[index3D(x + 1, y, z + 2)] = g_idata[gmem_index3D(
        row + 1, col, (dim + 2 + size & (size - 1)), size)];
    sdata[index3D(x + 1, y + 1, z + 2)] = g_idata[gmem_index3D(
        row + 1, col + 1, (dim + 2 + size & (size - 1)), size)];
    sdata[index3D(x + 1, y + 2, z + 2)] = g_idata[gmem_index3D(
        row + 1, col + 2, (dim + 2 + size & (size - 1)), size)];
    sdata[index3D(x + 1, y + 3, z + 2)] = g_idata[gmem_index3D(
        row + 1, col + 3, (dim + 2 + size & (size - 1)), size)];

    // nieparzyste
    sdata[index3D(x, y, z + 3)] =
        g_idata[gmem_index3D(row, col, (dim + 3 + size & (size - 1)), size)];
    sdata[index3D(x, y + 1, z + 3)] = g_idata[gmem_index3D(
        row, col + 1, (dim + 3 + size & (size - 1)), size)];
    sdata[index3D(x, y + 2, z + 3)] = g_idata[gmem_index3D(
        row, col + 2, (dim + 3 + size & (size - 1)), size)];
    sdata[index3D(x, y + 3, z + 3)] = g_idata[gmem_index3D(
        row, col + 3, (dim + 3 + size & (size - 1)), size)];
    sdata[index3D(x + 1, y, z + 3)] = g_idata[gmem_index3D(
        row + 1, col, (dim + 3 + size & (size - 1)), size)];
    sdata[index3D(x + 1, y + 1, z + 3)] = g_idata[gmem_index3D(
        row + 1, col + 1, (dim + 3 + size & (size - 1)), size)];
    sdata[index3D(x + 1, y + 2, z + 3)] = g_idata[gmem_index3D(
        row + 1, col + 2, (dim + 3 + size & (size - 1)), size)];
    sdata[index3D(x + 1, y + 3, z + 3)] = g_idata[gmem_index3D(
        row + 1, col + 3, (dim + 3 + size & (size - 1)), size)];
  }

  // lewa krawedz 4 elementy Y
  if (threadIdx.x == 0 && threadIdx.z == 0) {
    sdata[index3D(x - 1, y, z - 1)] = g_idata[gmem_index3D(
        row - 1 + size & (size - 1), (col + size) & (size - 1),
        (dim - 1 + size & (size - 1)), size)];
    sdata[index3D(x - 1, y + 1, z - 1)] = g_idata[gmem_index3D(
        row - 1 + size & (size - 1), (col + 1 + size) & (size - 1),
        (dim - 1 + size & (size - 1)), size)];
    sdata[index3D(x - 1, y + 2, z - 1)] = g_idata[gmem_index3D(
        row - 1 + size & (size - 1), (col + 2 + size) & (size - 1),
        (dim - 1 + size & (size - 1)), size)];
    sdata[index3D(x - 1, y + 3, z - 1)] = g_idata[gmem_index3D(
        row - 1 + size & (size - 1), (col + 3 + size) & (size - 1),
        (dim - 1 + size & (size - 1)), size)];
  }

  if (threadIdx.x == (blockDim.x - 1) && threadIdx.z == 0) {
    sdata[index3D(x + 2, y, z - 1)] = g_idata[gmem_index3D(
        row + 2 + size & (size - 1), (col + size) & (size - 1),
        (dim - 1 + size & (size - 1)), size)];
    sdata[index3D(x + 2, y + 1, z - 1)] = g_idata[gmem_index3D(
        row + 2 + size & (size - 1), (col + 1 + size) & (size - 1),
        (dim - 1 + size & (size - 1)), size)];
    sdata[index3D(x + 2, y + 2, z - 1)] = g_idata[gmem_index3D(
        row + 2 + size & (size - 1), (col + 2 + size) & (size - 1),
        (dim - 1 + size & (size - 1)), size)];
    sdata[index3D(x + 2, y + 3, z - 1)] = g_idata[gmem_index3D(
        row + 2 + size & (size - 1), (col + 3 + size) & (size - 1),
        (dim - 1 + size & (size - 1)), size)];
  }

  if (threadIdx.x == 0 && threadIdx.z == (blockDim.z - 1)) {
    sdata[index3D(x - 1, y, z + 2)] = g_idata[gmem_index3D(
        row - 1 + size & (size - 1), (col + size) & (size - 1),
        (dim + 2 + size & (size - 1)), size)];
    sdata[index3D(x - 1, y + 1, z + 2)] = g_idata[gmem_index3D(
        row - 1 + size & (size - 1), (col + 1 + size) & (size - 1),
        (dim + 2 + size & (size - 1)), size)];
    sdata[index3D(x - 1, y + 2, z + 2)] = g_idata[gmem_index3D(
        row - 1 + size & (size - 1), (col + 2 + size) & (size - 1),
        (dim + 2 + size & (size - 1)), size)];
    sdata[index3D(x - 1, y + 3, z + 2)] = g_idata[gmem_index3D(
        row - 1 + size & (size - 1), (col + 3 + size) & (size - 1),
        (dim + 2 + size & (size - 1)), size)];
  }

  if (threadIdx.x == (blockDim.x - 1) && threadIdx.z == (blockDim.z - 1)) {
    sdata[index3D(x + 2, y, z + 2)] = g_idata[gmem_index3D(
        row + 2 + size & (size - 1), (col + size) & (size - 1),
        (dim + 2 + size & (size - 1)), size)];
    sdata[index3D(x + 2, y + 1, z + 2)] = g_idata[gmem_index3D(
        row + 2 + size & (size - 1), (col + 1 + size) & (size - 1),
        (dim + 2 + size & (size - 1)), size)];
    sdata[index3D(x + 2, y + 2, z + 2)] = g_idata[gmem_index3D(
        row + 2 + size & (size - 1), (col + 2 + size) & (size - 1),
        (dim + 2 + size & (size - 1)), size)];
    sdata[index3D(x + 2, y + 3, z + 2)] = g_idata[gmem_index3D(
        row + 2 + size & (size - 1), (col + 3 + size) & (size - 1),
        (dim + 2 + size & (size - 1)), size)];
  }

  if (threadIdx.y == 0 && threadIdx.z == 0) {
    sdata[index3D(x, y - 1, z - 1)] =
        g_idata[gmem_index3D(row, (col - 1 + size) & (size - 1),
                             (dim - 1 + size) & (size - 1), size)];
    sdata[index3D(x + 1, y - 1, z - 1)] = g_idata[gmem_index3D(
        (row + 1 + size) & (size - 1), (col - 1 + size) & (size - 1),
        (dim - 1 + size) & (size - 1), size)];
  }

  if (threadIdx.y == (blockDim.y - 1) && threadIdx.z == 0) {
    sdata[index3D(x, y + 4, z - 1)] =
        g_idata[gmem_index3D(row, (col + 4 + size) & (size - 1),
                             (dim - 1 + size) & (size - 1), size)];
    sdata[index3D(x + 1, y + 4, z - 1)] = g_idata[gmem_index3D(
        (row + 1 + size) & (size - 1), (col + 4 + size) & (size - 1),
        (dim - 1 + size) & (size - 1), size)];
  }

  if (threadIdx.y == 0 && threadIdx.z == (blockDim.x - 1)) {
    sdata[index3D(x, y - 1, z + 2)] =
        g_idata[gmem_index3D(row, (col - 1 + size) & (size - 1),
                             (dim + 2 + size) & (size - 1), size)];
    sdata[index3D(x + 1, y - 1, z + 2)] = g_idata[gmem_index3D(
        (row + 1 + size) & (size - 1), (col - 1 + size) & (size - 1),
        (dim + 2 + size) & (size - 1), size)];
  }

  if (threadIdx.y == (blockDim.y - 1) && threadIdx.z == (blockDim.x - 1)) {
    sdata[index3D(x, y + 4, z + 2)] =
        g_idata[gmem_index3D(row, (col + 4 + size) & (size - 1),
                             (dim + 2 + size) & (size - 1), size)];
    sdata[index3D(x + 1, y + 4, z + 2)] = g_idata[gmem_index3D(
        (row + 1 + size) & (size - 1), (col + 4 + size) & (size - 1),
        (dim + 2 + size) & (size - 1), size)];
  }

  if (threadIdx.y == 0 && threadIdx.x == 0) {
    sdata[(2 * threadIdx.x + 1) * (4 + SHMEM_CUBE_SIZE) + 1 +
          (z + 2) * SHM_STRIDE_Z] =
        g_idata[gmem_index3D(row - 1 + size & (size - 1),
                             (col - 1 + size) & (size - 1),
                             (dim + size & (size - 1)), size)];
    sdata[(2 * threadIdx.x + 1) * (4 + SHMEM_CUBE_SIZE) + 1 +
          (z + 3) * SHM_STRIDE_Z] =
        g_idata[gmem_index3D(row - 1 + size & (size - 1),
                             (col - 1 + size) & (size - 1),
                             (dim + 1 + size & (size - 1)), size)];
  }

  if (threadIdx.y == (blockDim.y - 1) && threadIdx.x == 0) {
    sdata[(2 * threadIdx.x + 1) * (4 + SHMEM_CUBE_SIZE) +
          (SHMEM_CUBE_SIZE + 2) + (z + 2) * SHM_STRIDE_Z] =
        g_idata[gmem_index3D(row - 1 + size & (size - 1),
                             (col + 4 + size) & (size - 1),
                             (dim + size & (size - 1)), size)];
    sdata[(2 * threadIdx.x + 1) * (4 + SHMEM_CUBE_SIZE) +
          (SHMEM_CUBE_SIZE + 2) + (z + 3) * SHM_STRIDE_Z] =
        g_idata[gmem_index3D(row - 1 + size & (size - 1),
                             (col + 4 + size) & (size - 1),
                             (dim + 1 + size & (size - 1)), size)];
  }

  if (threadIdx.y == 0 && threadIdx.x == (blockDim.x - 1)) {
    sdata[(2 * threadIdx.x + 2 + 2) * (4 + SHMEM_CUBE_SIZE) + 1 +
          (z + 2) * SHM_STRIDE_Z] =
        g_idata[gmem_index3D(row + 2 + size & (size - 1),
                             (col - 1 + size) & (size - 1),
                             (dim + size & (size - 1)), size)];
    sdata[(2 * threadIdx.x + 2 + 2) * (4 + SHMEM_CUBE_SIZE) + 1 +
          (z + 3) * SHM_STRIDE_Z] =
        g_idata[gmem_index3D(row + 2 + size & (size - 1),
                             (col - 1 + size) & (size - 1),
                             (dim + 1 + size & (size - 1)), size)];
  }

  if (threadIdx.y == (blockDim.y - 1) && threadIdx.x == (blockDim.x - 1)) {
    sdata[(2 * threadIdx.x + 2 + 2) * (4 + SHMEM_CUBE_SIZE) +
          (SHMEM_CUBE_SIZE + 2) + (z + 2) * SHM_STRIDE_Z] =
        g_idata[gmem_index3D(row + 2 + size & (size - 1),
                             (col + 4 + size) & (size - 1),
                             (dim + size & (size - 1)), size)];
    sdata[(2 * threadIdx.x + 2 + 2) * (4 + SHMEM_CUBE_SIZE) +
          (SHMEM_CUBE_SIZE + 2) + (z + 3) * SHM_STRIDE_Z] =
        g_idata[gmem_index3D(row + 2 + size & (size - 1),
                             (col + 4 + size) & (size - 1),
                             (dim + 1 + size & (size - 1)), size)];
  }
}

__device__ inline int index3D_phi(int x, int y, int z) {
  int index = ((x + 1) * (2 + SHMEM_CUBE_SIZE)) +
              ((z + 1) * (2 + SHMEM_CUBE_SIZE) * (2 + SHMEM_CUBE_SIZE)) +
              (y + 1);

  return index;
}

__device__ void globmem_to_shmem_phi4_3D(data<>* sdata, data<>* g_idata, int x,
                                         int y, int z, int row, int col,
                                         int dim, int location, int size) {
  sdata[index3D_phi(x, y, z)] = g_idata[gmem_index3D(row, col, dim, size)];
  sdata[index3D_phi(x, y + 1, z)] =
      g_idata[gmem_index3D(row, col + 1, dim, size)];

  sdata[index3D_phi(x, y, z + 1)] =
      g_idata[gmem_index3D(row, col, dim + 1, size)];
  sdata[index3D_phi(x, y + 1, z + 1)] =
      g_idata[gmem_index3D(row, col + 1, dim + 1, size)];

  if (threadIdx.y == 0) {
    sdata[index3D_phi(x, -1, z)] =
        g_idata[gmem_index3D(row, (col - 1) & (size - 1), dim, size)];

    sdata[index3D_phi(x, -1, z + 1)] =
        g_idata[gmem_index3D(row, (col - 1) & (size - 1), dim + 1, size)];
  }

  if (threadIdx.y == (blockDim.y - 1)) {
    sdata[index3D_phi(x, SHMEM_CUBE_SIZE, z)] =
        g_idata[gmem_index3D(row, (col + 2) & (size - 1), dim, size)];

    sdata[index3D_phi(x, SHMEM_CUBE_SIZE, z + 1)] =
        g_idata[gmem_index3D(row, (col + 2) & (size - 1), dim + 1, size)];
  }

  if (threadIdx.x == 0) {
    sdata[index3D_phi(-1, y, z)] =
        g_idata[gmem_index3D((row - 1) & (size - 1), col, dim, size)];
    sdata[index3D_phi(-1, y + 1, z)] =
        g_idata[gmem_index3D((row - 1) & (size - 1), col + 1, dim, size)];

    sdata[index3D_phi(-1, y, z + 1)] =
        g_idata[gmem_index3D((row - 1) & (size - 1), col, dim + 1, size)];
    sdata[index3D_phi(-1, y + 1, z + 1)] =
        g_idata[gmem_index3D((row - 1) & (size - 1), col + 1, dim + 1, size)];
  }

  if (threadIdx.x == (blockDim.x - 1)) {
    sdata[index3D_phi(x + 1, y, z)] =
        g_idata[gmem_index3D((row + 1) & (size - 1), col, dim, size)];
    sdata[index3D_phi(x + 1, y + 1, z)] =
        g_idata[gmem_index3D((row + 1) & (size - 1), col + 1, dim, size)];

    sdata[index3D_phi(x + 1, y, z + 1)] =
        g_idata[gmem_index3D((row + 1) & (size - 1), col, dim + 1, size)];
    sdata[index3D_phi(x + 1, y + 1, z + 1)] =
        g_idata[gmem_index3D((row + 1) & (size - 1), col + 1, dim + 1, size)];
  }

  if (threadIdx.z == 0) {
    sdata[index3D_phi(x, y, z - 1)] =
        g_idata[gmem_index3D(row, col, (dim - 1 + size & (size - 1)), size)];
    sdata[index3D_phi(x, y + 1, z - 1)] = g_idata[gmem_index3D(
        row, col + 1, (dim - 1 + size & (size - 1)), size)];
  }

  if (threadIdx.z == (blockDim.z - 1)) {
    sdata[index3D_phi(x, y, z + 2)] =
        g_idata[gmem_index3D(row, col, (dim + 1 + size & (size - 1)), size)];
    sdata[index3D_phi(x, y + 1, z + 2)] = g_idata[gmem_index3D(
        row, col + 1, (dim + 1 + size & (size - 1)), size)];
  }
}

#endif
