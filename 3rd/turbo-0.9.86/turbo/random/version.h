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

#ifndef TURBO_RANDOM_VERSION_H_
#define TURBO_RANDOM_VERSION_H_

#include "turbo/module/module_version.h"

/**
 * @defgroup turbo_random_module Turbo Random Module
 * @defgroup turbo_random_engine Turbo Random Module
 * @defgroup turbo_random_uniform Turbo Random Module
 * @defgroup turbo_random_bernoulli Turbo Random Module
 * @defgroup turbo_random_beta Turbo Random Module
 * @defgroup turbo_random_exponential Turbo Random Module
 * @defgroup turbo_random_gaussian Turbo Random Module
 * @defgroup turbo_random_log_uniform Turbo Random Module
 * @defgroup turbo_random_poisson Turbo Random Module
 * @defgroup turbo_random_zipf Turbo Random Module
 * @defgroup turbo_random_bytes Turbo Random Module
 * @defgroup turbo_random_unicode Turbo Random Module
 * @brief Turbo Random Module
 */
namespace turbo {

    static constexpr turbo::ModuleVersion random_version = turbo::ModuleVersion{0, 9, 36};

}  // namespace turbo
#endif  // TURBO_MEMORY_VERSION_H_
