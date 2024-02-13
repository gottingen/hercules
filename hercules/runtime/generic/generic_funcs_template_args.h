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

#include "./generic_funcs.h"

/******************************************************************************
 * This file is only for cc_test
 *****************************************************************************/

namespace hercules {
namespace runtime {

/******************************************************************************
 * builtin object's member function
 *
 * Function schema :
 *    RTValue unbound_function(self, *args);
 *
 *****************************************************************************/

#define HERCULES_KERNEL_OBJECT_UNBOUND_FUNC(RetType, FuncName)                                 \
  template <typename... Args>                                                                    \
  static inline RetType FuncName(RTView self, Args&&... args) {                                  \
    return FuncName(static_cast<const Any&>(self),                                               \
                    PyArgs{std::initializer_list<RTView>{RTView(std::forward<Args>(args))...}}); \
  }

HERCULES_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_append);
HERCULES_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_add);
HERCULES_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_extend);
HERCULES_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_clear);
HERCULES_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_reserve);
HERCULES_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_capacity);
HERCULES_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_bucket_count);
HERCULES_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_find);
HERCULES_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_update);

// str/bytes/regex
HERCULES_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_lower);
HERCULES_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_upper);
HERCULES_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_isdigit);
HERCULES_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_isalpha);
HERCULES_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_encode);
HERCULES_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_decode);
HERCULES_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_split);
HERCULES_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_join);
HERCULES_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_replace);
HERCULES_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_match);
HERCULES_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_startswith);
HERCULES_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_endswith);
HERCULES_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_lstrip);
HERCULES_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_rstrip);
HERCULES_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_strip);
HERCULES_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_count);

// dict
HERCULES_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_keys);
HERCULES_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_values);
HERCULES_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_items);
HERCULES_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_get);

// NDArray
HERCULES_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_to_list);

// trie tree
HERCULES_KERNEL_OBJECT_UNBOUND_FUNC(RTValue, kernel_object_prefix_search);

#undef HERCULES_KERNEL_OBJECT_UNBOUND_FUNC

/******************************************************************************
 * python simple builtin modules and functions
 *
 * Function schema:
 *     RTValue module_method(*args);
 *
 *****************************************************************************/

#define HERCULES_KERNEL_GLOBAL_FUNC(RetType, FuncName)                                         \
  template <typename... Args>                                                                    \
  static inline RetType FuncName(Args&&... args) {                                               \
    return FuncName(PyArgs{std::initializer_list<RTView>{RTView(std::forward<Args>(args))...}}); \
  }

// json
HERCULES_KERNEL_GLOBAL_FUNC(RTValue, kernel_json_load);
HERCULES_KERNEL_GLOBAL_FUNC(RTValue, kernel_json_loads);
HERCULES_KERNEL_GLOBAL_FUNC(Unicode, kernel_json_dumps);

// file
HERCULES_KERNEL_GLOBAL_FUNC(File, kernel_file_open);

// builtin math func
HERCULES_KERNEL_GLOBAL_FUNC(RTValue, kernel_math_min);
HERCULES_KERNEL_GLOBAL_FUNC(RTValue, kernel_math_max);
HERCULES_KERNEL_GLOBAL_FUNC(RTValue, kernel_math_and);
HERCULES_KERNEL_GLOBAL_FUNC(RTValue, kernel_math_or);

}  // namespace runtime
}  // namespace hercules
