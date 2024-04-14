// Copyright 2024 The Elastic-AI Authors.
// part of Elastic AI Search
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

#ifdef __STSE2__

#define COLLIE_SIMD_DEFAULT_ARCH collie::simd::sse2
#include <collie/simd/simd.h>

#include "test_utils.hpp"

// Could be different than sse2 if we compile for other architecture avx
static_assert(std::is_same<collie::simd::default_arch, collie::simd::sse2>::value, "default arch correctly hooked");

#else

#define COLLIE_SIMD_DEFAULT_ARCH collie::simd::unsupported
#include <collie/simd/simd.h>

#endif
