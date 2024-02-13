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

#include <hercules/runtime/msgpack/msgpack.h>
#include <hercules/runtime/registry.h>

namespace hercules {
namespace runtime {

HERCULES_REGISTER_GLOBAL("runtime.msgpack_dumps").set_body([](PyArgs args) -> RTValue {
  HSCHECK(args.size() == 1) << "[runtime.msgpack.dumps] Expect 1 arguments but get " << args.size();
  return serialization::msgpack_dumps(args[0]);
});

HERCULES_REGISTER_GLOBAL("runtime.msgpack_loads").set_body([](PyArgs args) -> RTValue {
  HSCHECK(args.size() == 1) << "[runtime.msgpack.loads] Expect 1 arguments but get " << args.size();
  if (args[0].Is<string_view>()) {
    return serialization::msgpack_loads(args[0].AsNoCheck<string_view>());
  } else if (args[0].Is<unicode_view>()) {
    return serialization::msgpack_loads(UTF8Encode(args[0].AsNoCheck<unicode_view>()));
  } else {
    HSTHROW << "[runtime.msgpack.loads] Expect bytes or str but get " << args[0].type_name();
  }
  return None;
});

}  // namespace runtime
}  // namespace hercules
