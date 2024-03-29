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

#include <cuda.h>
#include <iostream>
#include <sstream>
#include <exception>

#include <collie/taskflow//utility/stream.h>

#define TF_CUDA_EXPAND( x ) x
#define TF_CUDA_REMOVE_FIRST_HELPER(N, ...) __VA_ARGS__
#define TF_CUDA_REMOVE_FIRST(...) TF_CUDA_EXPAND(TF_CUDA_REMOVE_FIRST_HELPER(__VA_ARGS__))
#define TF_CUDA_GET_FIRST_HELPER(N, ...) N
#define TF_CUDA_GET_FIRST(...) TF_CUDA_EXPAND(TF_CUDA_GET_FIRST_HELPER(__VA_ARGS__))

#define TF_CHECK_CUDA(...)                                       \
if(TF_CUDA_GET_FIRST(__VA_ARGS__) != cudaSuccess) {              \
  std::ostringstream oss;                                        \
  auto __ev__ = TF_CUDA_GET_FIRST(__VA_ARGS__);                  \
  oss << "[" << __FILE__ << ":" << __LINE__ << "] "              \
      << (cudaGetErrorString(__ev__)) << " ("                    \
      << (cudaGetErrorName(__ev__)) << ") - ";                   \
  collie::tf::ostreamize(oss, TF_CUDA_REMOVE_FIRST(__VA_ARGS__));        \
  throw std::runtime_error(oss.str());                           \
}

