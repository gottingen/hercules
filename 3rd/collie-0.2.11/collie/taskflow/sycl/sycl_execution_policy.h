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

/**
@file sycl_execution_policy.hpp
@brief SYCL execution policy include file
*/

namespace collie::tf {

/**
@class syclExecutionPolicy

@brief class to define execution policy for SYCL standard algorithms

@tparam NT number of threads per block
@tparam VT number of work units per thread

Execution policy configures the kernel execution parameters in SYCL algorithms.
The first template argument, @c NT, the number of threads per block should
always be a power-of-two number.
The second template argument, @c VT, the number of work units per thread
is recommended to be an odd number to avoid bank conflict.

Details can be referred to @ref SYCLSTDExecutionPolicy.
*/
template<unsigned NT, unsigned VT>
class syclExecutionPolicy {

  static_assert(is_pow2(NT), "max # threads per block must be a power of two");

  public:

  /** @brief static constant for getting the number of threads per block */
  const static unsigned nt = NT;

  /** @brief static constant for getting the number of work units per thread */
  const static unsigned vt = VT;

  /** @brief static constant for getting the number of elements to process per block */
  const static unsigned nv = NT*VT;

  /**
  @brief constructs an execution policy object with the given queue
   */
  syclExecutionPolicy(sycl::queue& queue) : _queue{queue} {}

  /**
  @brief returns an mutable reference to the associated queue
   */
  sycl::queue& queue() noexcept { return _queue; };

  /**
  @brief returns an immutable reference to the associated queue
   */
  const sycl::queue& queue() const noexcept { return _queue; }

  private:

  sycl::queue& _queue;
};

/**
@brief default execution policy
 */
using syclDefaultExecutionPolicy = syclExecutionPolicy<512, 9>;

}  // end of namespace collie::tf -----------------------------------------------------



