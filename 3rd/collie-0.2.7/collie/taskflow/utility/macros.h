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

#if defined(_MSC_VER)
  #define TF_FORCE_INLINE __forceinline
#elif defined(__GNUC__) && __GNUC__ > 3
  #define TF_FORCE_INLINE __attribute__((__always_inline__)) inline
#else
  #define TF_FORCE_INLINE inline
#endif

#if defined(_MSC_VER)
  #define TF_NO_INLINE __declspec(noinline)
#elif defined(__GNUC__) && __GNUC__ > 3
  #define TF_NO_INLINE __attribute__((__noinline__))
#else
  #define TF_NO_INLINE
#endif

// ----------------------------------------------------------------------------

#ifdef TF_DISABLE_EXCEPTION_HANDLING
  #define TF_EXECUTOR_EXCEPTION_HANDLER(worker, node, code_block) \
    code_block;
#else
  #define TF_EXECUTOR_EXCEPTION_HANDLER(worker, node, code_block)  \
    try {                                          \
      code_block;                                  \
    } catch(...) {                                 \
      _process_exception(worker, node);            \
    }
#endif

// ----------------------------------------------------------------------------    
