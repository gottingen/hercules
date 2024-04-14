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

#pragma once

#include <collie/simd/types/fma3_sse_register.h>
#include <collie/simd/types/fma4_register.h>
#include <collie/simd/types/sse2_register.h>
#include <collie/simd/types/sse3_register.h>
#include <collie/simd/types/sse4_1_register.h>
#include <collie/simd/types/sse4_2_register.h>

#include <collie/simd/types/avx2_register.h>
#include <collie/simd/types/avx_register.h>
#include <collie/simd/types/avxvnni_register.h>
#include <collie/simd/types/fma3_avx2_register.h>
#include <collie/simd/types/fma3_avx_register.h>

#include <collie/simd/types/avx512vnni_avx512bw_register.h>
#include <collie/simd/types/avx512vnni_avx512vbmi_register.h>

#include <collie/simd/types/avx512ifma_register.h>
#include <collie/simd/types/avx512vbmi_register.h>

#include <collie/simd/types/avx512er_register.h>
#include <collie/simd/types/avx512pf_register.h>

#include <collie/simd/types/avx512bw_register.h>
#include <collie/simd/types/avx512cd_register.h>
#include <collie/simd/types/avx512dq_register.h>
#include <collie/simd/types/avx512f_register.h>

#include <collie/simd/types/neon64_register.h>
#include <collie/simd/types/neon_register.h>

#include <collie/simd/types/sve_register.h>

#include <collie/simd/types/rvv_register.h>

#include <collie/simd/types/wasm_register.h>
