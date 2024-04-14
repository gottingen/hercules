// Copyright 2023 The Elastic-AI Authors.
// part of Elastic AI Search
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
//
// Created by jeff on 23-12-10.
//

#ifndef TURBO_TIMES_VERSION_H_
#define TURBO_TIMES_VERSION_H_

#include "turbo/module/module_version.h"

/**
 * @defgroup turbo_times_clock Clock utilities
 * @defgroup turbo_times_stop_watcher StopWatcher utilities
 * @defgroup turbo_times_duration Duration utilities
 * @defgroup turbo_times_time_point_create TimePoint utilities
 * @defgroup turbo_times_time_point TimePoint utilities
 * @defgroup turbo_times_time_zone TimeZone utilities
 * @defgroup turbo_times_civil_time CivilTime utilities
 */
namespace turbo {

    static constexpr turbo::ModuleVersion times_version = turbo::ModuleVersion{0, 9, 36};

}  // namespace turbo
#endif  // TURBO_TIMES_VERSION_H_
