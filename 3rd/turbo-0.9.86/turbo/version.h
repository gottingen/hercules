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

#ifndef TURBO_VERSION_H_
#define TURBO_VERSION_H_

#include "turbo/base/version.h"
#include "turbo/concurrent/version.h"
#include "turbo/container/version.h"
#include "turbo/crypto/version.h"
#include "turbo/files/version.h"
#include "turbo/flags/version.h"
#include "turbo/format/version.h"
#include "turbo/hash/version.h"
#include "turbo/log/version.h"
#include "turbo/memory/version.h"
#include "turbo/meta/version.h"
#include "turbo/module/version.h"
#include "turbo/platform/version.h"
#include "turbo/random/version.h"
#include "turbo/strings/version.h"
#include "turbo/times/version.h"
#include "turbo/unicode/version.h"
#include "turbo/system/version.h"

namespace turbo {

    // Check if the version is compatible with the submodules
    static constexpr turbo::ModuleVersion kMinCompatibleVersion = turbo::ModuleVersion{0, 9, 36};
    /**
     * @brief Turbo version this is the version of the whole library
     *        it should be updated when any of the submodules are updated.
     *        The version is in the format of major.minor.patch
     *        major: major changes, API changes, breaking changes
     *        minor: minor changes, new features, new modules
     *        patch: bug fixes, performance improvements
     *        The version is used to check if the library is compatible with the
     *        application.
     */
    static constexpr turbo::ModuleVersion version = turbo::ModuleVersion{0, 9, 54};

    // Check if the version is compatible with the submodules
    static_assert(kMinCompatibleVersion <= base_version, "Turbo version is lower than base version");
    static_assert(kMinCompatibleVersion <= concurrent_version, "Turbo version is lower than concurrent version");
    static_assert(kMinCompatibleVersion <= container_version, "Turbo version is lower than container version");
    static_assert(kMinCompatibleVersion <= crypto_version, "Turbo version is lower than crypto version");
    static_assert(kMinCompatibleVersion <= files_version, "Turbo version is lower than files version");
    static_assert(kMinCompatibleVersion <= flags_version, "Turbo version is lower than flags version");
    static_assert(kMinCompatibleVersion <= format_version, "Turbo version is lower than format version");
    static_assert(kMinCompatibleVersion <= hash_version, "Turbo version is lower than hash version");
    static_assert(kMinCompatibleVersion <= log_version, "Turbo version is lower than log version");
    static_assert(kMinCompatibleVersion <= memory_version, "Turbo version is lower than memory version");
    static_assert(kMinCompatibleVersion <= meta_version, "Turbo version is lower than meta version");
    static_assert(kMinCompatibleVersion <= platform_version, "Turbo version is lower than platform version");
    static_assert(kMinCompatibleVersion <= module_version, "Turbo version is lower than module version");
    static_assert(kMinCompatibleVersion <= random_version, "Turbo version is lower than random version");
    static_assert(kMinCompatibleVersion <= strings_version, "Turbo version is lower than strings version");
    static_assert(kMinCompatibleVersion <= times_version, "Turbo version is lower than times version");
    static_assert(kMinCompatibleVersion <= unicode_version, "Turbo version is lower than unicode version");
    static_assert(kMinCompatibleVersion <= sys_version, "Turbo version is lower than tf version");

    // turbo version is the most recent version of the library
    static_assert(version >= base_version, "Turbo version is lower than base version");
    static_assert(version >= concurrent_version, "Turbo version is lower than concurrent version");
    static_assert(version >= container_version, "Turbo version is lower than container version");
    static_assert(version >= crypto_version, "Turbo version is lower than crypto version");
    static_assert(version >= files_version, "Turbo version is lower than files version");
    static_assert(version >= flags_version, "Turbo version is lower than flags version");
    static_assert(version >= format_version, "Turbo version is lower than format version");
    static_assert(version >= hash_version, "Turbo version is lower than hash version");
    static_assert(version >= log_version, "Turbo version is lower than log version");
    static_assert(version >= memory_version, "Turbo version is lower than memory version");
    static_assert(version >= meta_version, "Turbo version is lower than meta version");
    static_assert(version >= module_version, "Turbo version is lower than module version");
    static_assert(version >= platform_version, "Turbo version is lower than platform version");
    static_assert(version >= random_version, "Turbo version is lower than random version");
    static_assert(version >= strings_version, "Turbo version is lower than strings version");
    static_assert(version >= times_version, "Turbo version is lower than times version");
    static_assert(version >= unicode_version, "Turbo version is lower than unicode version");
    static_assert(version >= files_version, "Turbo version is lower than fiber version");
    static_assert(version >= sys_version, "Turbo version is lower than taskflow version");


} // namespace turbo

#endif // TURBO_VERSION_H_
