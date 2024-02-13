// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from incubator-tvm.
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file cpu_device_api.cc
 */
#include <hercules/runtime/device_api.h>

#include <cstdlib>
#include <cstring>

#include "hercules/core//framework/allocator.h"
#include "hercules/core//framework/arena.h"
#include "hercules/core//framework/bfc_arena.h"

#include <hercules/runtime/c_runtime_api.h>
#include <hercules/runtime/dlpack.h>
#include <hercules/runtime/logging.h>
#include <hercules/runtime/registry.h>

#ifdef __ANDROID__
#include <android/api-level.h>
#endif

namespace hercules {
namespace runtime {

class CPUDeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(HerculesDevice ctx) final {
  }
  void GetAttr(HerculesDevice ctx, DeviceAttrKind kind, RTValue* rv) final {
    if (kind == kExist) {
      *rv = 1;
    }
  }

  void* Alloc(HerculesDevice ctx, size_t nbytes) final {
    HSCHECK(cpuBFCAllocator != nullptr);
    void* ptr = cpuBFCAllocator->Alloc(nbytes);
    return ptr;
  }

  void* Alloc(HerculesDevice ctx, size_t nbytes, size_t alignment, DLDataType type_hint) final {
    return Alloc(ctx, nbytes);
  }

  void* AllocRaw(HerculesDevice ctx,
                 size_t nbytes,
                 size_t alignment,
                 DLDataType type_hint) final {
    void* ptr;
#if _MSC_VER
    ptr = _aligned_malloc(nbytes, alignment);
    if (ptr == nullptr)
      throw std::bad_alloc();
#elif defined(__ANDROID__) && __ANDROID_API__ < 17
    ptr = memalign(alignment, nbytes);
    if (ptr == nullptr)
      throw std::bad_alloc();
#else
    // posix_memalign is available in android ndk since __ANDROID_API__ >= 17
    int ret = posix_memalign(&ptr, alignment, nbytes);
    if (ret != 0)
      throw std::bad_alloc();
#endif
    return ptr;
  }

  void Free(HerculesDevice ctx, void* ptr) final {
    HSCHECK(cpuBFCAllocator != nullptr);
    cpuBFCAllocator->Free(ptr);
  }

  void FreeRaw(HerculesDevice ctx, void* ptr) final {
#if _MSC_VER
    _aligned_free(ptr);
#else
    free(ptr);
#endif
  }

  void CopyDataFromTo(const void* from,
                      size_t from_offset,
                      void* to,
                      size_t to_offset,
                      size_t size,
                      HerculesDevice ctx_from,
                      HerculesDevice ctx_to,
                      DLDataType type_hint,
                      HerculesStreamHandle stream) final {
    std::memcpy(
        static_cast<char*>(to) + to_offset, static_cast<const char*>(from) + from_offset, size);
  }

  HerculesStreamHandle CreateStream(HerculesDevice ctx) final {
    return nullptr;
  }

  void FreeStream(HerculesDevice ctx, HerculesStreamHandle stream) final {
  }

  HerculesStreamHandle GetDefaultStream(HerculesDevice ctx) final {
    return nullptr;
  }

  HerculesStreamHandle GetCurrentThreadStream(HerculesDevice ctx) final {
    return nullptr;
  }

  std::shared_ptr<void> GetSharedCurrentThreadStream(HerculesDevice ctx) final {
    return nullptr;
  }

  void SetCurrentThreadStream(HerculesDevice ctx, std::shared_ptr<void> stream) final {
  }

  void StreamSync(HerculesDevice ctx, HerculesStreamHandle stream) final {
  }

  void CreateEventSync(HerculesStreamHandle stream) final {
  }

  void SyncStreamFromTo(HerculesDevice ctx,
                        HerculesStreamHandle event_src,
                        HerculesStreamHandle event_dst) final {
  }

  static CPUDeviceAPI* Global() {
    // NOTE: explicitly use new to avoid exit-time destruction of global state
    // Global state will be recycled by OS as the process exits.
    static auto* inst = new CPUDeviceAPI();
    return inst;
  }

 private:
  brt::BFCArena* cpuBFCAllocator =
      new brt::BFCArena(std::unique_ptr<brt::IAllocator>(new brt::CPUAllocator()), 1ULL << 32);
  ;
};

struct CPUGlobalEntry {
  CPUGlobalEntry() {
    CPUDeviceAPI::Global();
  }
};

HERCULES_REGISTER_GLOBAL("device_api.cpu").set_body([](PyArgs args) -> RTValue {
  DeviceAPI* ptr = CPUDeviceAPI::Global();
  return static_cast<void*>(ptr);
});
}  // namespace runtime
}  // namespace hercules
