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


#include <turbo/log/tlog_inl.h>
#include <turbo/log/common-inl.h>
#include <turbo/log/details/backtracer-inl.h>
#include <turbo/log/details/registry-inl.h>
#include <turbo/log/details/os-inl.h>
#include <turbo/log/pattern_formatter-inl.h>
#include <turbo/log/details/log_msg-inl.h>
#include <turbo/log/details/log_msg_buffer-inl.h>
#include <turbo/log/logger-inl.h>
#include <turbo/log/sinks/sink-inl.h>
#include <turbo/log/sinks/base_sink-inl.h>
#include "turbo/log/details/null_mutex.h"

#include <mutex>

// template instantiate logger constructor with sinks init list
template TURBO_DLL turbo::tlog::logger::logger(std::string name, sinks_init_list::iterator begin, sinks_init_list::iterator end);
template class TURBO_DLL turbo::tlog::sinks::base_sink<std::mutex>;
template class TURBO_DLL turbo::tlog::sinks::base_sink<turbo::tlog::details::null_mutex>;
