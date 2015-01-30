// includes, system
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <fstream>
#include <vector>
#include <iterator>
#include <stdlib.h>

// includes, project
#include <cuda.h>
#include <curand_kernel.h>

// includes, kernels
#include "kernel_2D.cuh"
#include "kernel_3D.cuh"
#include "reduce.cuh"
#include "config.h"
#include "statistic.h"

// includes, cmdline
//#include "cmdline.h"
//#include "cmdline_types.h"

static cudaError err;

#define cuda(f, ...)                                                           \
  \
if((err = cuda##f(__VA_ARGS__)) != cudaSuccess) {                              \
    fprintf(stderr, #f "() %s\n", cudaGetErrorString(err));                    \
    exit(-1);                                                                  \
  \
}

#define cudaNoSync(...) cuda(__VA_ARGS__)

using namespace std;

template <typename F = float> struct obj { F a; };

class Simulation {
  Statistic *errors;
  obj<float> z;
  data<> *d_idata, *h_idata, *h_odata_test;
  vector<double> mean;
  vector<float> mean_time;
  std::size_t x, dimension, grid_size;
  std::size_t *seed;
  int _genseed;
  float n, gamma, lambda;

public:
  Simulation(int x, int dimension, float n, float gamma, float lambda,
             int _genseed)
      : x(x), dimension(dimension), n(n), gamma(gamma), lambda(lambda),
        _genseed(_genseed) {
    errors = new Statistic(x, n);
    grid_size = (dimension == 3) ? x * x * x : x * x;
    h_odata_test = (data<> *)malloc(sizeof(data<>) * grid_size);
  }

  void extended_phi4_metropolis_2D(int& iteration) {
    data<> *d_idata;
    unsigned int *d_seed;

    ofstream myfile("data");
    ofstream myfile2("data2");

    unsigned long mem_size = (sizeof(data<>) * grid_size);

    h_idata = (data<> *)malloc(mem_size);

    int seed_size = sizeof(unsigned int) * 4 * grid_size;
    seed = (unsigned int *)malloc(seed_size);

    init_gen(_genseed, h_idata, seed);

    cuda(Malloc, (void **)&d_idata, mem_size);
    cuda(Malloc, (void **)&d_seed, seed_size);
    cuda(Memcpy, d_idata, h_idata, mem_size, cudaMemcpyHostToDevice);
    cuda(Memcpy, d_seed, seed, seed_size, cudaMemcpyHostToDevice);

    data<> *h_odata = (data<> *)malloc(mem_size);

    dim3 grid(x / 32, x / 32);
    dim3 threads(8, 4);

    double timer = 0;

    double rt1 = 0;
    int limit = 0;

    dim3 seedgrid(x / 32, x / 32);
    dim3 seedthreads(8, 4);

    for (int i = 0; i < iteration; i++) {
      Phi_2D << <grid, threads>>>
          (d_idata, d_seed, iteration, x, n, gamma, lambda, 0, 0);

      cudaThreadSynchronize();

      Phi_2D << <grid, threads>>>
          (d_idata, d_seed, iteration, x, n, gamma, lambda, 1, 1);

      cudaThreadSynchronize();

      Phi_2D << <grid, threads>>>
          (d_idata, d_seed, iteration, x, n, gamma, lambda, 0, 1);

      cudaThreadSynchronize();

      Phi_2D << <grid, threads>>>
          (d_idata, d_seed, iteration, x, n, gamma, lambda, 1, 0);

      cudaThreadSynchronize();

      if (i % 1000 == 0) {
        printf("Iteration %d\n", i);
      }

      if (i > 1000) {
        rt1 += sqrt_mag(d_idata);
        limit++;

        if (i % 50 == 0 && i > 0) {
          // printf("    Iteration %d       Totality of M^2 :=
          // %4.10f\n",i*100,(rt/limit)/(x*x));
          mean.push_back((rt1 / limit) / grid_size);
          myfile << (rt1 / limit) / grid_size << endl;
          rt1 = 0.0f;
          limit = 0;
        }

        cudaThreadSynchronize();
      }
    }

    double analytical_value = errors->analytical_phi_2d(lambda);

    cout << "Monte Carlo output " << GPU_correlation(d_idata, 4) << endl;
    cout << "Analytica value: " << analytical_value << endl;
    errors->autocorr(analytical_value, mean);

    // stop kernel and timer

    double time = 0 - timer;
    cout << "Time: " << time << endl;

    cuda(Memcpy, h_odata, d_idata, mem_size, cudaMemcpyDeviceToHost);

    // create_vtkfile(h_odata,x);

    // cleanup memory
    free(h_idata);
    free(seed);
    cuda(Free, d_idata);
    cuda(Free, d_seed);

    cudaThreadExit();
  }

  void extended_phi4_metropolis_3D(int &GLOBAL_SWEEPS) {
    ofstream myfile("data");
    ofstream myfile2("data2");

    std::size_t mem_size = (sizeof(data<>) * grid_size);

    h_idata = (data<> *)malloc(mem_size);

    int seed_size = sizeof(std::size_t) * 4 * (grid_size);
    seed = (std::size_t *)malloc(seed_size);

    init_gen(_genseed, h_idata, seed);

    data<> *d_idata;
    std::size_t *d_seed;

    cuda(Malloc, (void **)&d_idata, mem_size);
    cuda(Malloc, (void **)&d_seed, seed_size);
    cuda(Memcpy, d_idata, h_idata, mem_size, cudaMemcpyHostToDevice);
    cuda(Memcpy, d_seed, seed, seed_size, cudaMemcpyHostToDevice);

    data<> *h_odata = (data<> *)malloc(mem_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float tempM;
    std::size_t limitM;
    float rt1 = 0;
    std::size_t limit = 0;

    dim3 grid(x / (2 * SHMEM_CUBE_SIZE), x / (2 * SHMEM_CUBE_SIZE),
              x / (2 * SHMEM_CUBE_SIZE)); //( x 4,y 4)
    dim3 threads(SHMEM_CUBE_SIZE / 2, SHMEM_CUBE_SIZE / 4, SHMEM_CUBE_SIZE / 2);

    cudaEventRecord(start);

    float last_mean = 0;

#define Phi_3D_Kernel(stride_x, stride_y, stride_z)                            \
  Phi_3D<float> << <grid, threads>>>                                           \
      (d_idata, d_seed, x, n, gamma, lambda, stride_x, stride_y, stride_z);    \
  cudaThreadSynchronize();

    for (int i = 0; i < GLOBAL_SWEEPS; ++i) {

      Phi_3D_Kernel(0, 0, 0);
      Phi_3D_Kernel(1, 1, 0);
      Phi_3D_Kernel(0, 1, 0);
      Phi_3D_Kernel(1, 0, 0);
      Phi_3D_Kernel(0, 0, 1);
      Phi_3D_Kernel(1, 1, 1);
      Phi_3D_Kernel(0, 1, 1);
      Phi_3D_Kernel(1, 0, 1);

      /*
Phi_3D<float> << <grid, threads>>>
(d_idata, d_seed, x, n, gamma, lambda, 0, 0, 0);

cudaThreadSynchronize();

Phi_3D << <grid, threads>>>
(d_idata, d_seed, x, n, gamma, lambda, 1, 1, 0);

cudaThreadSynchronize();

Phi_3D << <grid, threads>>>
(d_idata, d_seed, x, n, gamma, lambda, 0, 1, 0);

cudaThreadSynchronize();

Phi_3D << <grid, threads>>>
(d_idata, d_seed, x, n, gamma, lambda, 1, 0, 0);

cudaThreadSynchronize();

Phi_3D << <grid, threads>>>
(d_idata, d_seed, x, n, gamma, lambda, 0, 0, 1);

cudaThreadSynchronize();

Phi_3D << <grid, threads>>>
(d_idata, d_seed, x, n, gamma, lambda, 1, 1, 1);

cudaThreadSynchronize();

Phi_3D << <grid, threads>>>
(d_idata, d_seed, x, n, gamma, lambda, 0, 1, 1);

cudaThreadSynchronize();

Phi_3D << <grid, threads>>>
(d_idata, d_seed, x, n, gamma, lambda, 1, 0, 1);

cudaThreadSynchronize();

*/
      if (i > 400) {
        rt1 += sqrt_mag(d_idata);
        limit++;

        if (i > 400) {
          mean.push_back((rt1 / limit) / grid_size);
          myfile << (rt1 / limit) / (x * x * x) << endl;
          last_mean = (rt1 / limit) / grid_size;
          printf("    Iteration %d       Totality of M^2 := %4.10f\n", i,
                 last_mean);
          rt1 = 0.0f;
          limit = 0;
        }
        cudaThreadSynchronize();
      }
    }

    cudaEventRecord(stop);
    cudaEventRecord(stop);

    float timer;
    cudaEventElapsedTime(&timer, start, stop);

    tempM = rt1;
    limitM = limit;

    cuda(Memcpy, h_odata, d_idata, mem_size, cudaMemcpyDeviceToHost);

    double analytical_value = errors->analytical_phi_3d(lambda);

    std::cout << endl << "Monte Carlo Mean: " << last_mean

              << std::endl;
    std::cout << "Analytical Mean: " << analytical_value << endl;
    errors->autocorr(analytical_value, mean);

    free(h_odata);
    free(h_idata);
    free(seed);
    cuda(Free, d_idata);
    cuda(Free, d_seed);

    cudaThreadExit();
  }

  void getGPU() {
    int devId = -1;
    cudaDeviceProp pdev;

    cudaGetDevice(&devId);
    cudaGetDeviceProperties(&pdev, devId);

    cout << "\t"
         << "GPU properties: " << endl;
    cout << "\t"
         << "name:         " << pdev.name << endl;
    cout << "\t"
         << "capability:   " << pdev.major << "." << pdev.minor << endl;
    cout << "\t"
         << "clock:        " << pdev.clockRate / 1000000.0 << " GHz" << endl;
    cout << "\t"
         << "processors:   " << pdev.multiProcessorCount << endl;
    cout << "\t"
         << "cores:        " << 32 * pdev.multiProcessorCount << endl;
    cout << "\t"
         << "warp:         " << pdev.warpSize << endl;
    cout << "\t"
         << "max thr/blk:  " << pdev.maxThreadsPerBlock << endl;
    cout << "\t"
         << "max blk size: " << pdev.maxThreadsDim[0] << "x"
         << pdev.maxThreadsDim[1] << "x" << pdev.maxThreadsDim[2] << endl;
    cout << "\t"
         << "max grd size: " << pdev.maxGridSize[0] << "x"
         << pdev.maxGridSize[1] << endl;
  }

  void create_vtkfile(data<> *V, int &size) {
    ofstream myfile("phi.vtk");
    myfile << "# vtk DataFile Version 2.0 " << endl;
    myfile << "Cuda simulation of Phi4 Model" << endl;
    myfile << "ASCII" << endl;
    myfile << "DATASET STRUCTURED_GRID" << endl;
    myfile << "DIMENSIONS " << size << " " << size << " " << 1 << endl;
    myfile << "POINTS " << (int)(x * x) << " float" << endl;

    for (std::size_t i = 0; i < x; i++) {
      for (std::size_t j = 0; j < x; j++) {
        myfile << (float)i << " " << (float)j << " 0.0" << endl;
      }
    }

    // data points
    myfile << "POINT_DATA " << (int)(x * x) << endl;
    myfile << "SCALARS data float" << endl;
    myfile << "LOOKUP_TABLE default" << endl;

    for (std::size_t z = 0; z < (x * x); ++z) {
      myfile << V[z].vector[0] << endl;
    }
  }

  float sqrt_mag(data<> *d_idata) {
    int threads = 64;

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(grid_size / threads, 1, 1);

    float *d_odata = NULL;

    cuda(Malloc, (void **)&d_odata, grid_size / threads * sizeof(float));

    int smemSize =
        (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

    reduce2<data<>> << <dimGrid, dimBlock, smemSize>>>
        (d_idata, d_odata, grid_size);
    float *out = NULL;
    out = (float *)malloc(grid_size / threads * sizeof(float));

    cuda(Memcpy, out, d_odata, grid_size / threads * sizeof(float),
         cudaMemcpyDeviceToHost);

    float gpu_result = 0;

    for (std::size_t i = 0; i < (grid_size / threads); ++i) {
      gpu_result += out[i];
    }

    free(out);
    cuda(Free, d_odata);
    return (gpu_result);
  }

  float sqrt_mag2(float *d_idata) {
    int threads = 256;

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(grid_size / threads, 1, 1);

    float *d_odata = NULL;

    cuda(Malloc, (void **)&d_odata, grid_size / threads * sizeof(float));

    int smemSize =
        (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

    reduce6<float, 256, true> << <dimGrid, dimBlock, smemSize>>>
        (d_idata, d_odata, x * x * x);

    float *out = NULL;
    out = (float *)malloc(grid_size / threads * sizeof(float));

    cuda(Memcpy, out, d_odata, grid_size / threads * sizeof(float),
         cudaMemcpyDeviceToHost);

    float gpu_result = 0;

    for (std::size_t i = 0; i < (grid_size / threads); i++) {
      gpu_result += out[i];
    }

    free(out);
    cuda(Free, d_odata);
    return (gpu_result);
  }

  void init_gen(int &genseed, data<> *h_idata, unsigned int *seed) {
    srand(*seed);

    for (std::size_t i = 0; i < grid_size; ++i) {
      h_idata[i].vector[0] = static_cast<float>(rand() / RAND_MAX);
    }

    for (std::size_t i = 0; i < 4 * grid_size; ++i) {
      seed[i] = rand();
    }
  }

  float GPU_correlation(data<> *d_idata, int R) {
    int threads = 1024;

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(grid_size / threads, 1, 1);

    float *d_odata = NULL;

    cuda(Malloc, (void **)&d_odata, grid_size / threads * sizeof(float));

    int smemSize =
        (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

    correlation<data<>> << <dimGrid, dimBlock, smemSize>>>
        (d_idata, d_odata, grid_size, x, R);

    float *out = NULL;
    out = (float *)malloc(grid_size / threads * sizeof(float));

    cuda(Memcpy, out, d_odata, grid_size / threads * sizeof(float),
         cudaMemcpyDeviceToHost);

    float gpu_result = 0;

    for (std::size_t i = 0; i < (grid_size / threads); i++) {
      gpu_result += out[i];
    }

    free(out);
    cuda(Free, d_odata);
    return (gpu_result) / (2 * grid_size);
  }

  ~Simulation() {
    delete errors;
    free(h_odata_test);
  }
};

int main(int argc, char **argv) {
  int dimension = 3;
  int x = 64;
  float mi = 0.25f;
  float gamma = 0.0f;
  float lambda = 2.0f;
  int iteration = 1000;
  int seed = 12345;

  Simulation *Phi = new Simulation(x, dimension, mi, gamma, lambda, seed);

  if (dimension == 2) {
    Phi->extended_phi4_metropolis_2D(iteration);

  } else if (dimension == 3) {
    Phi->extended_phi4_metropolis_3D(iteration);
  }

  delete Phi;

  return 0;
}
