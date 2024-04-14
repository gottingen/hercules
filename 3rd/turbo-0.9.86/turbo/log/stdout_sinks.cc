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


#include <mutex>

#include "turbo/log/details/null_mutex.h"
#include <turbo/log/async.h>
#include <turbo/log/sinks/stdout_sinks-inl.h>

template
class TURBO_DLL turbo::tlog::sinks::stdout_sink_base<turbo::tlog::details::console_mutex>;

template
class TURBO_DLL turbo::tlog::sinks::stdout_sink_base<turbo::tlog::details::console_nullmutex>;

template
class TURBO_DLL turbo::tlog::sinks::stdout_sink<turbo::tlog::details::console_mutex>;

template
class TURBO_DLL turbo::tlog::sinks::stdout_sink<turbo::tlog::details::console_nullmutex>;

template
class TURBO_DLL turbo::tlog::sinks::stderr_sink<turbo::tlog::details::console_mutex>;

template
class TURBO_DLL turbo::tlog::sinks::stderr_sink<turbo::tlog::details::console_nullmutex>;

template TURBO_DLL std::shared_ptr<turbo::tlog::logger>
turbo::tlog::stdout_logger_mt<turbo::tlog::synchronous_factory>(const std::string &logger_name);

template TURBO_DLL std::shared_ptr<turbo::tlog::logger>
turbo::tlog::stdout_logger_st<turbo::tlog::synchronous_factory>(const std::string &logger_name);

template TURBO_DLL std::shared_ptr<turbo::tlog::logger>
turbo::tlog::stderr_logger_mt<turbo::tlog::synchronous_factory>(const std::string &logger_name);

template TURBO_DLL std::shared_ptr<turbo::tlog::logger>
turbo::tlog::stderr_logger_st<turbo::tlog::synchronous_factory>(const std::string &logger_name);

template TURBO_DLL std::shared_ptr<turbo::tlog::logger>
turbo::tlog::stdout_logger_mt<turbo::tlog::async_factory>(const std::string &logger_name);

template TURBO_DLL std::shared_ptr<turbo::tlog::logger>
turbo::tlog::stdout_logger_st<turbo::tlog::async_factory>(const std::string &logger_name);

template TURBO_DLL std::shared_ptr<turbo::tlog::logger>
turbo::tlog::stderr_logger_mt<turbo::tlog::async_factory>(const std::string &logger_name);

template TURBO_DLL std::shared_ptr<turbo::tlog::logger>
turbo::tlog::stderr_logger_st<turbo::tlog::async_factory>(const std::string &logger_name);
