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

#include <hercules/pipeline/global_unique_index.h>
#include <hercules/pipeline/op_kernel.h>
#include <hercules/runtime/container_private.h>
#include <hercules/runtime/file_util.h>
#include <hercules/runtime/native_object_registry.h>

namespace hercules {
namespace runtime {

class Future {
 public:
  Future() : body_(nullptr) {
  }
  ~Future() = default;

  void set_body(std::function<RTValue()> body);

  RTValue get() const;

  static UserDataRef make_future_udref(std::function<RTValue()> body);

 private:
  std::function<RTValue()> body_;
};

}  // namespace runtime
}  // namespace hercules
