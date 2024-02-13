// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
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

#include <hercules/runtime/c_runtime_api.h>
#include <hercules/runtime/device_api.h>
#include <hercules/runtime/dlpack.h>
#include <hercules/runtime/generic/generic_funcs.h>
#include <hercules/runtime/registry.h>

namespace hercules {
namespace runtime {

// set device api
HERCULES_REGISTER_GLOBAL("runtime.SetDevice").set_body([](PyArgs args) -> RTValue {
  HerculesDevice device;
  device.device_type = static_cast<DLDeviceType>(args[0].As<int64_t>());
  device.device_id = args[1].As<int64_t>();
  DeviceAPI::Get(device)->SetDevice(device);
  return None;
});

// set device api
HERCULES_REGISTER_GLOBAL("runtime.GetDeviceAttr").set_body([](PyArgs args) -> RTValue {
  HerculesDevice ctx;
  ctx.device_type = static_cast<DLDeviceType>(args[0].As<int64_t>());
  ctx.device_id = args[1].As<int64_t>();

  RTValue ret;
  DeviceAttrKind kind = static_cast<DeviceAttrKind>(args[2].As<int64_t>());
  if (kind == kExist) {
    DeviceAPI* api = DeviceAPI::Get(ctx, true);
    if (api != nullptr) {
      api->GetAttr(ctx, kind, &ret);
    } else {
      ret = 0;
    }
  } else {
    DeviceAPI::Get(ctx)->GetAttr(ctx, kind, &ret);
  }
  return ret;
});

HERCULES_REGISTER_GLOBAL("runtime.HerculesSetCurrentThreadStream")
    .set_body_typed(HerculesSetCurrentThreadStream);

// create stream
HERCULES_REGISTER_GLOBAL("runtime.DefaultStream").set_body([](PyArgs args) -> RTValue {
  HSCHECK(args.size() == 1) << "cuda_module_default_stream expect 1 args, bug get " << args.size();
  HSCHECK(args[0].type_code() == TypeIndex::kRuntimeInteger)
      << "Create Stream first arg must be integer. ";
  int device_id = args[0].As<int64_t>();
  HSCHECK(device_id >= 0) << "Device Id must be equal or greater than zeros .";

  return kernel_cuda_module_create_stream(device_id);
});

// create stream
HERCULES_REGISTER_GLOBAL("runtime.CreateStream").set_body([](PyArgs args) -> RTValue {
  HSCHECK(args.size() == 1) << "cuda_module_create_stream expect 1 args, bug get " << args.size();
  HSCHECK(args[0].type_code() == TypeIndex::kRuntimeInteger)
      << "Create Stream first arg must be integer. ";
  int device_id = args[0].As<int64_t>();
  HSCHECK(device_id >= 0) << "Device Id must be equal or greater than zeros .";

  return kernel_cuda_module_create_stream(device_id);
});

// StreamSync
HERCULES_REGISTER_GLOBAL("runtime.StreamSync").set_body([](PyArgs args) -> RTValue {
  HSCHECK(args.size() == 2) << "StreamSync expect 2 args, bug get " << args.size();
  RTView opaq = args[0].As<RTView>();
  int device_id = args[1].As<int64_t>();
  HSCHECK(device_id >= 0) << "Device Id must be equal or greater than zeros .";

  kernel_cuda_module_stream_sync(opaq, device_id);
  return None;
});

// StreamSync
HERCULES_REGISTER_GLOBAL("runtime.CurrentThreadStreamSync").set_body([](PyArgs args) -> RTValue {
  HSCHECK(args.size() == 2) << "CurrentThreadStreamSync expect 2 args, bug get " << args.size();
  HSCHECK(args[0].type_code() == TypeIndex::kRuntimeInteger)
      << "CurrentThreadStreamSync first arg must be integer. ";
  HSCHECK(args[1].type_code() == TypeIndex::kRuntimeInteger)
      << "CurrentThreadStreamSync first arg must be integer. ";
  int64_t device_type = args[0].AsNoCheck<int64_t>();
  int64_t device_id = args[1].AsNoCheck<int64_t>();
  HerculesDevice device{DLDeviceType(device_type), DLDeviceType(device_id)};
  DeviceAPI::Get(device)->CurrentThreadStreamSync(device);
  return None;
});

}  // namespace runtime
}  // namespace hercules
