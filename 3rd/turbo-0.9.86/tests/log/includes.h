
// Copyright 2023 The Turbo Authors.
//
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
#pragma once

#include "utils.h"
#include <chrono>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <iomanip>
#include <stdlib.h>

#define TLOG_ACTIVE_LEVEL TLOG_LEVEL_DEBUG

#include "turbo/log/logging.h"
#include "turbo/log/async.h"
#include "turbo/log/details/fmt_helper.h"
#include "turbo/log/sinks/basic_file_sink.h"
#include "turbo/log/sinks/daily_file_sink.h"
#include "turbo/log/sinks/null_sink.h"
#include "turbo/log/sinks/ostream_sink.h"
#include "turbo/log/sinks/rotating_file_sink.h"
#include "turbo/log/sinks/stdout_color_sinks.h"
#include "turbo/log/pattern_formatter.h"
