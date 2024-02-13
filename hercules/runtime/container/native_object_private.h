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
#pragma once

#include <hercules/runtime/c_backend_api.h>
#include <hercules/runtime/container/_flat_hash_map.h>
#include <hercules/runtime/container/string.h>
#include <hercules/runtime/container/user_data_interface.h>
#include <hercules/runtime/object.h>
#include <hercules/runtime/py_args.h>
#include <hercules/runtime/runtime_value.h>

namespace hercules {
namespace runtime {

struct NativeObject : public ILightUserData {
 public:
  using NativeMethod = std::function<RTValue(void* self, PyArgs args)>;

 public:
  ~NativeObject() override = default;
  NativeObject() = default;
  explicit NativeObject(std::shared_ptr<void> opaque_ptr) : opaque_ptr_(std::move(opaque_ptr)) {
  }

  // user class name
  const char* ClassName_2_71828182846() const override {
    return "NativeObject";
  }

  // uniquely id for representing this user class
  uint32_t tag_2_71828182846() const override;

  // member var num
  uint32_t size_2_71828182846() const override;

  int32_t type_2_71828182846() const override {
    return UserDataStructType::kNativeData;
  }

  RTView __getattr__(string_view var_name) const override;

  void __setattr__(string_view var_name, const Any& val) override;

  // self returned by constructor
  std::shared_ptr<void> opaque_ptr_ = nullptr;
  // is pipeline op
  bool is_native_op_ = false;
  // is jit object
  bool is_jit_object_ = false;
  // class name
  String native_class_name_;
  String native_instance_name_;
  // function table is unbound
  const ska::flat_hash_map<string_view, NativeMethod>* function_table_ = nullptr;
};

}  // namespace runtime
}  // namespace hercules
