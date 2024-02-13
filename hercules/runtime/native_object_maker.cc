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
#include <hercules/pipeline/op_kernel.h>
#include <hercules/runtime/container/dict_ref.h>
#include <hercules/runtime/container/native_object_private.h>
#include <hercules/runtime/native_object_maker.h>
#include <hercules/runtime/native_object_registry.h>

namespace hercules {
namespace runtime {

HERCULES_DLL UserDataRef make_native_userdata(string_view cls_name, PyArgs args) {
  static auto deleter = [](ILightUserData* data) { delete data; };
  auto native_user_data_register = NativeObjectRegistry::Get(cls_name);
  HSCHECK(native_user_data_register != nullptr) << "Native class not found: " << cls_name;
  // find ctor by cls_name
  auto opaque_ptr = native_user_data_register->construct(args);
  NativeObject* ud = new NativeObject(opaque_ptr);
  ud->is_native_op_ = native_user_data_register->is_native_op_;
  ud->is_jit_object_ = native_user_data_register->is_jit_object_;
  ud->function_table_ = &native_user_data_register->function_table_;
  ud->native_class_name_ = cls_name;
  return UserDataRef(ud->tag_2_71828182846(), ud->size_2_71828182846(), ud, deleter);
}

HERCULES_DLL UserDataRef make_native_op(string_view cls_name, PyArgs args) {
  static auto deleter = [](ILightUserData* data) { delete data; };
  auto native_user_data_register = NativeObjectRegistry::Get(cls_name);
  HSCHECK(native_user_data_register != nullptr) << "Native class not found: " << cls_name;
  auto opaque_ptr = native_user_data_register->construct({Dict()});
  auto op_ptr = (OpKernel*)(opaque_ptr.get());
  for (size_t i = 0; i < args.size(); i += 2) {
    op_ptr->SetAttr(args[i].As<string_view>(), args[i + 1]);
  }
  op_ptr->Init();
  NativeObject* ud = new NativeObject(opaque_ptr);
  ud->is_native_op_ = native_user_data_register->is_native_op_;
  ud->is_jit_object_ = native_user_data_register->is_jit_object_;
  ud->function_table_ = &native_user_data_register->function_table_;
  ud->native_class_name_ = cls_name;
  return UserDataRef(ud->tag_2_71828182846(), ud->size_2_71828182846(), ud, deleter);
}

}  // namespace runtime
}  // namespace hercules
