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

#ifndef COLLIE_SIMD_ARCH_FMA3_AVX2_H_
#define COLLIE_SIMD_ARCH_FMA3_AVX2_H_

#include <collie/simd/types/fma3_avx2_register.h>

// Allow inclusion of fma3_avx.hpp
#ifdef COLLIE_SIMD_ARCH_FMA3_AVX_H_
#undef COLLIE_SIMD_ARCH_FMA3_AVX_H_
#define COLLIE_SIMD_FORCE_FMA3_AVX_H_
#endif

// Disallow inclusion of ./fma3_avx_register.hpp
#ifndef COLLIE_SIMD_TYPES_FMA3_AVX_REGISTER_H_
#define COLLIE_SIMD_TYPES_FMA3_AVX_REGISTER_H_
#define COLLIE_SIMD_FORCE_FMA3_AVX_REGISTER_H_
#endif

// Include ./fma3_avx.hpp but s/avx/avx2
#define avx avx2
#include <collie/simd/arch/fma3_avx.h>
#undef avx
#undef COLLIE_SIMD_ARCH_FMA3_AVX_H_

// Carefully restore guards
#ifdef COLLIE_SIMD_FORCE_FMA3_AVX_H_
#define COLLIE_SIMD_ARCH_FMA3_AVX_H_
#undef COLLIE_SIMD_FORCE_FMA3_AVX_H_
#endif

#ifdef COLLIE_SIMD_FORCE_FMA3_AVX_REGISTER_H_
#undef COLLIE_SIMD_TYPES_FMA3_AVX_REGISTER_H_
#undef COLLIE_SIMD_FORCE_FMA3_AVX_REGISTER_H_
#endif

#endif  // COLLIE_SIMD_ARCH_FMA3_AVX2_H_
