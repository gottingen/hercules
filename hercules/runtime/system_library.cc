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
 * \file system_library.cc
 * \brief Create library module that directly get symbol from the system lib.
 */
#include "library_module.h"

#include <mutex>

#include <hercules/runtime/c_backend_api.h>
#include <hercules/runtime/memory.h>
#include <hercules/runtime/registry.h>

namespace matxscript {
namespace runtime {

class SystemLibrary : public Library {
 public:
  SystemLibrary() = default;

  void* GetSymbol(const char* name) final {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = tbl_.find(name);
    if (it != tbl_.end()) {
      return it->second;
    } else {
      return nullptr;
    }
  }

  void RegisterSymbol(const std::string& name, void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = tbl_.find(name);
    if (it != tbl_.end() && ptr != it->second) {
      MXLOG(WARNING) << "SystemLib symbol " << name << " get overridden to a different address "
                     << ptr << "->" << it->second;
    }
    tbl_[name] = ptr;
  }

  static const ObjectPtr<SystemLibrary>& Global() {
    static auto inst = make_object<SystemLibrary>();
    return inst;
  }

 private:
  // Internal mutex
  std::mutex mutex_;
  // Internal symbol table
  std::unordered_map<std::string, void*> tbl_;
};

MATXSCRIPT_REGISTER_GLOBAL("runtime.SystemLib").set_body_typed([]() {
  static auto mod = CreateModuleFromLibrary(SystemLibrary::Global());
  return mod;
});
}  // namespace runtime
}  // namespace matxscript

int MATXScriptBackendRegisterSystemLibSymbol(const char* name, void* ptr) {
  ::matxscript::runtime::SystemLibrary::Global()->RegisterSymbol(name, ptr);
  return 0;
}