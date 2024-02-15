/*
 * RTX 3060: nvcc -O2 -gencode=arch=compute_86,code=sm_86 -ccbin /usr/bin/g++-12 ./test.cu
 */

#include <vector>
#include <iostream>
#include <chrono>
#include <random>
#include "helper_cuda.h"
#include "helper_string.h"
#include "cuda_runtime.h"

const size_t N = 50000000;

template <typename T>
__global__ void addData(const T* d_array, T* d_sum, unsigned int sz) {
  unsigned int tid = threadIdx.x + (blockDim.x * blockIdx.x);
  if (tid < sz) {
    atomicAdd(d_sum, d_array[tid]);
  }
}

long sumFloat32() {
  std::vector<float> data(N);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> distrib(-10, 10);
  for (size_t i = 0; i < N; ++i) {
    data[i] = distrib(gen);
  }
  float* d_data;
  checkCudaErrors(cudaMalloc(&d_data, sizeof(float) * N));
  checkCudaErrors(cudaMemcpy(d_data, data.data(), sizeof(float) * N, cudaMemcpyHostToDevice));
  float* d_sum;
  checkCudaErrors(cudaMalloc(&d_sum, sizeof(float)));
  const int block = 256;
  const int grid = (N + block - 1) / block;
  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));
  auto t1 = std::chrono::high_resolution_clock::now();
  addData<float><<<grid, block, 0, stream>>>(d_data, d_sum, N);
  cudaStreamSynchronize(stream);
  auto t2 = std::chrono::high_resolution_clock::now();
  float* h_sum;
  checkCudaErrors(cudaMallocHost(&h_sum, sizeof(float)));
  checkCudaErrors(cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
  // std::cout << "sumFloat32: " << (*h_sum) << std::endl;
  auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
  std::cout << "sumFloat32 time " << int_ms.count() << " ms, size = " << sizeof(float) << " bytes." << std::endl;
  checkCudaErrors(cudaFreeHost(h_sum));
  checkCudaErrors(cudaStreamDestroy(stream));
  checkCudaErrors(cudaFree(d_sum));
  checkCudaErrors(cudaFree(d_data));
  return int_ms.count();
}

long sumFloat64() {
  std::vector<double> data(N);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> distrib(-10, 10);
  for (size_t i = 0; i < N; ++i) {
    data[i] = distrib(gen);
  }
  double* d_data;
  checkCudaErrors(cudaMalloc(&d_data, sizeof(double) * N));
  checkCudaErrors(cudaMemcpy(d_data, data.data(), sizeof(double) * N, cudaMemcpyHostToDevice));
  double* d_sum;
  checkCudaErrors(cudaMalloc(&d_sum, sizeof(double)));
  const int block = 256;
  const int grid = (N + block - 1) / block;
  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));
  auto t1 = std::chrono::high_resolution_clock::now();
  addData<double><<<grid, block, 0, stream>>>(d_data, d_sum, N);
  cudaStreamSynchronize(stream);
  auto t2 = std::chrono::high_resolution_clock::now();
  double* h_sum;
  checkCudaErrors(cudaMallocHost(&h_sum, sizeof(double)));
  checkCudaErrors(cudaMemcpy(h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
  // std::cout << "sumFloat64: " << (*h_sum) << std::endl;
  auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
  std::cout << "sumFloat64 time " << int_ms.count() << " ms, size = " << sizeof(double) << " bytes." << std::endl;
  checkCudaErrors(cudaFreeHost(h_sum));
  checkCudaErrors(cudaStreamDestroy(stream));
  checkCudaErrors(cudaFree(d_sum));
  checkCudaErrors(cudaFree(d_data));
  return int_ms.count();
}

long sumUInt64() {
  using UInt64 = unsigned long long;
  std::vector<UInt64> data(N);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<UInt64> distrib(0, 1000);
  for (size_t i = 0; i < N; ++i) {
    data[i] = distrib(gen);
  }
  UInt64* d_data;
  checkCudaErrors(cudaMalloc(&d_data, sizeof(UInt64) * N));
  checkCudaErrors(cudaMemcpy(d_data, data.data(), sizeof(UInt64) * N, cudaMemcpyHostToDevice));
  UInt64* d_sum;
  checkCudaErrors(cudaMalloc(&d_sum, sizeof(UInt64)));
  const int block = 256;
  const int grid = (N + block - 1) / block;
  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));
  auto t1 = std::chrono::high_resolution_clock::now();
  addData<UInt64><<<grid, block, 0, stream>>>(d_data, d_sum, N);
  cudaStreamSynchronize(stream);
  auto t2 = std::chrono::high_resolution_clock::now();
  UInt64* h_sum;
  checkCudaErrors(cudaMallocHost(&h_sum, sizeof(UInt64)));
  checkCudaErrors(cudaMemcpy(h_sum, d_sum, sizeof(UInt64), cudaMemcpyDeviceToHost));
  // std::cout << "sumUInt64: " << (*h_sum) << std::endl;
  auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
  std::cout << "sumUInt64 time " << int_ms.count() << " ms, size = " << sizeof(UInt64) << " bytes." << std::endl;
  checkCudaErrors(cudaFreeHost(h_sum));
  checkCudaErrors(cudaStreamDestroy(stream));
  checkCudaErrors(cudaFree(d_sum));
  checkCudaErrors(cudaFree(d_data));
  return int_ms.count();
}

long sumUInt32() {
  using UInt32 = unsigned int;
  std::vector<UInt32> data(N);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<UInt32> distrib(0, 1000);
  for (size_t i = 0; i < N; ++i) {
    data[i] = distrib(gen);
  }
  UInt32* d_data;
  checkCudaErrors(cudaMalloc(&d_data, sizeof(UInt32) * N));
  checkCudaErrors(cudaMemcpy(d_data, data.data(), sizeof(UInt32) * N, cudaMemcpyHostToDevice));
  UInt32* d_sum;
  checkCudaErrors(cudaMalloc(&d_sum, sizeof(UInt32)));
  const int block = 256;
  const int grid = (N + block - 1) / block;
  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));
  auto t1 = std::chrono::high_resolution_clock::now();
  addData<UInt32><<<grid, block, 0, stream>>>(d_data, d_sum, N);
  cudaStreamSynchronize(stream);
  auto t2 = std::chrono::high_resolution_clock::now();
  UInt32* h_sum;
  checkCudaErrors(cudaMallocHost(&h_sum, sizeof(UInt32)));
  checkCudaErrors(cudaMemcpy(h_sum, d_sum, sizeof(UInt32), cudaMemcpyDeviceToHost));
  // std::cout << "sumUInt32: " << (*h_sum) << std::endl;
  auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
  std::cout << "sumUInt32 time " << int_ms.count() << " ms, size = " << sizeof(UInt32) << " bytes." << std::endl;
  checkCudaErrors(cudaFreeHost(h_sum));
  checkCudaErrors(cudaStreamDestroy(stream));
  checkCudaErrors(cudaFree(d_sum));
  checkCudaErrors(cudaFree(d_data));
  return int_ms.count();
}

int main() {
  sumUInt64();
  sumUInt32();
  sumFloat32();
  sumFloat64();
  return 0;
}
