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

#include "turbo/flags/usage_config.h"

#include <functional>
#include <iostream>
#include <string>
#include <mutex>
#include "turbo/platform/port.h"
#include "turbo/base/const_init.h"
#include "turbo/platform/thread_annotations.h"
#include "turbo/flags/internal/path_util.h"
#include "turbo/flags/internal/program_name.h"
#include "turbo/strings/match.h"
#include "turbo/strings/string_view.h"
#include "turbo/strings/str_strip.h"

extern "C" {

// Additional report of fatal usage error message before we std::exit. Error is
// fatal if is_fatal argument to ReportUsageError is true.
TURBO_WEAK void turbo_internal_report_fatal_usage_error(std::string_view) {}

}  // extern "C"

namespace turbo {

    namespace flags_internal {

        namespace {

// --------------------------------------------------------------------
// Returns true if flags defined in the filename should be reported with
// -helpshort flag.

            bool ContainsHelpshortFlags(std::string_view filename) {
                // By default we only want flags in binary's main. We expect the main
                // routine to reside in <program>.cc or <program>-main.cc or
                // <program>_main.cc, where the <program> is the name of the binary
                // (without .exe on Windows).
                auto suffix = flags_internal::Basename(filename);
                auto program_name = flags_internal::short_program_invocation_name();
                std::string_view program_name_ref = program_name;
#if defined(_WIN32)
                turbo::ConsumeSuffix(&program_name_ref, ".exe");
#endif
                if (!turbo::consume_prefix(&suffix, program_name_ref))
                    return false;
                return turbo::starts_with(suffix, ".") || turbo::starts_with(suffix, "-main.") ||
                       turbo::starts_with(suffix, "_main.");
            }

// --------------------------------------------------------------------
// Returns true if flags defined in the filename should be reported with
// -helppackage flag.

            bool ContainsHelppackageFlags(std::string_view filename) {
                // TODO(rogeeff): implement properly when registry is available.
                return ContainsHelpshortFlags(filename);
            }

// --------------------------------------------------------------------
// Generates program version information into supplied output.

            std::string VersionString() {
                std::string version_str(flags_internal::short_program_invocation_name());

                version_str += "\n";

#if !defined(NDEBUG)
                version_str += "Debug build (NDEBUG not #defined)\n";
#endif

                return version_str;
            }

// --------------------------------------------------------------------
// Normalizes the filename specific to the build system/filesystem used.

            std::string NormalizeFilename(std::string_view filename) {
                // Skip any leading slashes
                auto pos = filename.find_first_not_of("\\/");
                if (pos == std::string_view::npos) return "";

                filename.remove_prefix(pos);
                return std::string(filename);
            }

// --------------------------------------------------------------------

            TURBO_CONST_INIT std::mutex custom_usage_config_guard;
            TURBO_CONST_INIT FlagsUsageConfig *custom_usage_config
                    TURBO_GUARDED_BY(custom_usage_config_guard) = nullptr;

        }  // namespace

        FlagsUsageConfig GetUsageConfig() {
            std::unique_lock l(custom_usage_config_guard);

            if (custom_usage_config) return *custom_usage_config;

            FlagsUsageConfig default_config;
            default_config.contains_helpshort_flags = &ContainsHelpshortFlags;
            default_config.contains_help_flags = &ContainsHelppackageFlags;
            default_config.contains_helppackage_flags = &ContainsHelppackageFlags;
            default_config.version_string = &VersionString;
            default_config.normalize_filename = &NormalizeFilename;

            return default_config;
        }

        void ReportUsageError(std::string_view msg, bool is_fatal) {
            std::cerr << "ERROR: " << msg << std::endl;

            if (is_fatal) {
                turbo_internal_report_fatal_usage_error(msg);
            }
        }

    }  // namespace flags_internal

    void set_flags_usage_config(FlagsUsageConfig usage_config) {
        std::unique_lock l(flags_internal::custom_usage_config_guard);

        if (!usage_config.contains_helpshort_flags)
            usage_config.contains_helpshort_flags =
                    flags_internal::ContainsHelpshortFlags;

        if (!usage_config.contains_help_flags)
            usage_config.contains_help_flags = flags_internal::ContainsHelppackageFlags;

        if (!usage_config.contains_helppackage_flags)
            usage_config.contains_helppackage_flags =
                    flags_internal::ContainsHelppackageFlags;

        if (!usage_config.version_string)
            usage_config.version_string = flags_internal::VersionString;

        if (!usage_config.normalize_filename)
            usage_config.normalize_filename = flags_internal::NormalizeFilename;

        if (flags_internal::custom_usage_config)
            *flags_internal::custom_usage_config = usage_config;
        else
            flags_internal::custom_usage_config = new FlagsUsageConfig(usage_config);
    }


}  // namespace turbo
