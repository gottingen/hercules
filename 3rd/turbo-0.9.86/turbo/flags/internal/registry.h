// Copyright 2023 The titan-search Authors.
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

#ifndef TURBO_FLAGS_INTERNAL_REGISTRY_H_
#define TURBO_FLAGS_INTERNAL_REGISTRY_H_

#include <functional>

#include "turbo/platform/port.h"
#include "turbo/flags/commandlineflag.h"
#include "turbo/flags/internal/commandlineflag.h"
#include "turbo/strings/string_view.h"

// --------------------------------------------------------------------
// Global flags registry API.

namespace turbo::flags_internal {
    // Executes specified visitor for each non-retired flag in the registry. While
    // callback are executed, the registry is locked and can't be changed.
    void ForEachFlag(std::function<void(CommandLineFlag &)> visitor);

    //-----------------------------------------------------------------------------

    bool RegisterCommandLineFlag(CommandLineFlag &, const char *filename);

    void FinalizeRegistry();

    //-----------------------------------------------------------------------------
    // Retired registrations:
    //
    // Retired flag registrations are treated specially. A 'retired' flag is
    // provided only for compatibility with automated invocations that still
    // name it.  A 'retired' flag:
    //   - is not bound to a C++ FLAGS_ reference.
    //   - has a type and a value, but that value is intentionally inaccessible.
    //   - does not appear in --help messages.
    //   - is fully supported by _all_ flag parsing routines.
    //   - consumes args normally, and complains about type mismatches in its
    //     argument.
    //   - emits a complaint but does not die (e.g. LOG(ERROR)) if it is
    //     accessed by name through the flags API for parsing or otherwise.
    //
    // The registrations for a flag happen in an unspecified order as the
    // initializers for the namespace-scope objects of a program are run.
    // Any number of weak registrations for a flag can weakly define the flag.
    // One non-weak registration will upgrade the flag from weak to non-weak.
    // Further weak registrations of a non-weak flag are ignored.
    //
    // This mechanism is designed to support moving dead flags into a
    // 'graveyard' library.  An example migration:
    //
    //   0: Remove references to this FLAGS_flagname in the C++ codebase.
    //   1: Register as 'retired' in old_lib.
    //   2: Make old_lib depend on graveyard.
    //   3: Add a redundant 'retired' registration to graveyard.
    //   4: Remove the old_lib 'retired' registration.
    //   5: Eventually delete the graveyard registration entirely.
    //

    // Retire flag with name "name" and type indicated by ops.
    void Retire(const char *name, FlagFastTypeId type_id, char *buf);

    constexpr size_t kRetiredFlagObjSize = 3 * sizeof(void *);
    constexpr size_t kRetiredFlagObjAlignment = alignof(void *);

    // Registered a retired flag with name 'flag_name' and type 'T'.
    template<typename T>
    class RetiredFlag {
    public:
        void Retire(const char *flag_name) {
            flags_internal::Retire(flag_name, base_internal::FastTypeId<T>(), buf_);
        }

    private:
        alignas(kRetiredFlagObjAlignment) char buf_[kRetiredFlagObjSize];
    };

}  // namespace turbo::flags_internal

#endif  // TURBO_FLAGS_INTERNAL_REGISTRY_H_
