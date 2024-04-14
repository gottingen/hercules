// Copyright 2023 The titan-search Authors.
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

#include "turbo/log/details/null_mutex.h"
#include <turbo/log/sinks/basic_file_sink-inl.h>
#include <turbo/log/sinks/base_sink-inl.h>
#include <turbo/log/sinks/rotating_file_sink-inl.h>
#include <mutex>

template
class TURBO_DLL turbo::tlog::sinks::basic_file_sink<std::mutex>;

template
class TURBO_DLL turbo::tlog::sinks::basic_file_sink<turbo::tlog::details::null_mutex>;


template
class TURBO_DLL turbo::tlog::sinks::rotating_file_sink<std::mutex>;

template
class TURBO_DLL turbo::tlog::sinks::rotating_file_sink<turbo::tlog::details::null_mutex>;
