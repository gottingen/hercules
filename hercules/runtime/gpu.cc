// Copyright 2024 The EA Authors.
// Copyright(c) 2015-present, Gabi Melman & spdlog contributors.
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

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <hercules/runtime/lib.h>

#ifdef HERCULES_GPU

#include <cuda.h>

#define fail(err)                                                                      \
  do {                                                                                 \
    const char *msg;                                                                   \
    cuGetErrorString((err), &msg);                                                     \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, msg);             \
    abort();                                                                           \
  } while (0)

#define check(call)                                                                    \
  do {                                                                                 \
    auto err = (call);                                                                 \
    if (err != CUDA_SUCCESS) {                                                         \
      fail(err);                                                                       \
    }                                                                                  \
  } while (0)

static std::vector<CUmodule> modules;
static CUcontext context;

void hs_nvptx_init() {
  CUdevice device;
  check(cuInit(0));
  check(cuDeviceGet(&device, 0));
  check(cuCtxCreate(&context, 0, device));
}

HS_FUNC void seq_nvptx_load_module(const char *filename) {
  CUmodule module;
  check(cuModuleLoad(&module, filename));
  modules.push_back(module);
}

HS_FUNC hs_int_t hs_nvptx_device_count() {
  int devCount;
  check(cuDeviceGetCount(&devCount));
  return devCount;
}

HS_FUNC hs_str_t hs_nvptx_device_name(CUdevice device) {
  char name[128];
  check(cuDeviceGetName(name, sizeof(name) - 1, device));
  auto sz = static_cast<hs_int_t>(strlen(name));
  auto *p = (char *)hs_alloc_atomic(sz);
  memcpy(p, name, sz);
  return {sz, p};
}

HS_FUNC hs_int_t hs_nvptx_device_capability(CUdevice device) {
  int devMajor, devMinor;
  check(cuDeviceComputeCapability(&devMajor, &devMinor, device));
  return ((hs_int_t)devMajor << 32) | (hs_int_t)devMinor;
}

HS_FUNC CUdevice hs_nvptx_device(hs_int_t idx) {
  CUdevice device;
  check(cuDeviceGet(&device, idx));
  return device;
}

static bool name_char_valid(char c, bool first) {
  bool ok = ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || (c == '_');
  if (!first)
    ok = ok || ('0' <= c && c <= '9');
  return ok;
}

HS_FUNC CUfunction seq_nvptx_function(hs_str_t name) {
  CUfunction function;
  CUresult result;

  std::vector<char> clean(name.len + 1);
  for (unsigned i = 0; i < name.len; i++) {
    char c = name.str[i];
    clean[i] = (name_char_valid(c, i == 0) ? c : '_');
  }
  clean[name.len] = '\0';

  for (auto it = modules.rbegin(); it != modules.rend(); ++it) {
    result = cuModuleGetFunction(&function, *it, clean.data());
    if (result == CUDA_SUCCESS) {
      return function;
    } else if (result == CUDA_ERROR_NOT_FOUND) {
      continue;
    } else {
      break;
    }
  }

  fail(result);
  return {};
}

HS_FUNC void seq_nvptx_invoke(CUfunction f, unsigned int gridDimX,
                               unsigned int gridDimY, unsigned int gridDimZ,
                               unsigned int blockDimX, unsigned int blockDimY,
                               unsigned int blockDimZ, unsigned int sharedMemBytes,
                               void **kernelParams) {
  check(cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                       sharedMemBytes, nullptr, kernelParams, nullptr));
}

HS_FUNC CUdeviceptr hs_nvptx_device_alloc(hs_int_t size) {
  if (size == 0)
    return {};

  CUdeviceptr devp;
  check(cuMemAlloc(&devp, size));
  return devp;
}

HS_FUNC void hs_nvptx_memcpy_h2d(CUdeviceptr devp, char *hostp, hs_int_t size) {
  if (size)
    check(cuMemcpyHtoD(devp, hostp, size));
}

HS_FUNC void hs_nvptx_memcpy_d2h(char *hostp, CUdeviceptr devp, hs_int_t size) {
  if (size)
    check(cuMemcpyDtoH(hostp, devp, size));
}

HS_FUNC void hs_nvptx_device_free(CUdeviceptr devp) {
  if (devp)
    check(cuMemFree(devp));
}

#endif /* HERCULES_GPU */
