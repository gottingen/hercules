
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

#include "collie/log/logging.h"
#include "collie/log/async.h"
#include "collie/log/details/fmt_helper.h"
#include "collie/log/sinks/basic_file_sink.h"
#include "collie/log/sinks/daily_file_sink.h"
#include "collie/log/sinks/null_sink.h"
#include "collie/log/sinks/ostream_sink.h"
#include "collie/log/sinks/rotating_file_sink.h"
#include "collie/log/sinks/stdout_color_sinks.h"
#include "collie/log/pattern_formatter.h"
