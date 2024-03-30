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

#ifndef COLLIE_SIMD_SIMD_H_
#define COLLIE_SIMD_SIMD_H_

#include <collie/port/simd_config.h>
#include <collie/simd/arch/scalar.h>
#include <collie//simd/memory/allocator.h>

#if defined(COLLIE_SIMD_NO_SUPPORTED_ARCHITECTURE)
// to type definition or anything appart from scalar definition and aligned allocator
#else
#include <collie/simd/types/batch.h>
#include <collie/simd/types/batch_constant.h>
#include <collie/simd/types/simd_traits.h>
// This include must come last
#include <collie/simd/types/api.h>
#endif
#endif  // COLLIE_SIMD_SIMD_H_
