// Copyright 2024 The Elastic AI Search Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#pragma once

#include <collie/taskflow/cuda/cudaflow.h>

namespace collie::tf {

// ----------------------------------------------------------------------------
// row-major matrix multiplication
// ----------------------------------------------------------------------------

template <typename T>
__global__ void cuda_matmul(
  const T* A,
  const T* B,
  T* C,
  size_t M,
  size_t K,
  size_t N
) {
  __shared__ T A_tile[32][32];
  __shared__ T B_tile[32][32];

  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  T res = 0;

  for(size_t k = 0; k < K; k += 32) {
    if((threadIdx.x + k) < K && y < M) {
      A_tile[threadIdx.y][threadIdx.x] = A[y * K + threadIdx.x + k];
    }
    else{
      A_tile[threadIdx.y][threadIdx.x] = 0;
    }

    if((threadIdx.y + k) < K && x < N) {
      B_tile[threadIdx.y][threadIdx.x] = B[(threadIdx.y + k) * N + x];
    }
    else{
      B_tile[threadIdx.y][threadIdx.x] = 0;
    }

    __syncthreads();

    for(size_t i = 0; i < 32; ++i) {
      res += A_tile[threadIdx.y][i] * B_tile[i][threadIdx.x];
    }
    __syncthreads();
  }

  if(x < N && y < M) {
    C[y * N + x] = res;
  }

}

} // end of namespace collie::tf ---------------------------------------------------------
