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
#include <hercules/runtime/cxxabi_helper.h>

extern "C" {
#if defined(__clang__) || (__GNUC__ <= 4) || \
    (defined(_GLIBCXX_USE_CXX11_ABI) && _GLIBCXX_USE_CXX11_ABI == 0)
MATX_DLL int MATXSCRIPT_FLAGS_GLIBCXX_USE_CXX11_ABI = 0;
#else
MATX_DLL int MATXSCRIPT_FLAGS_GLIBCXX_USE_CXX11_ABI = 1;
#endif

int MATXScriptAPI_USE_CXX11_ABI() {
  return MATXSCRIPT_FLAGS_GLIBCXX_USE_CXX11_ABI;
}
}