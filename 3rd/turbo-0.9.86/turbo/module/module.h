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
#ifndef TURBO_MODULE_MODULE_H_
#define TURBO_MODULE_MODULE_H_


#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <functional>
#include "turbo/module/module_loader.h"
#include "turbo/platform/port.h"

namespace turbo {
    /// Allows you to create classes which dynamically load dependencies at runtime
    ///
    /// Creates a dependency module which will dynamically load a library
    /// at runtime instead of at link time. This class will be a component of a
    /// module class which will have member functions for each of the functions
    /// we use in turbo
    class Module {
    public:
        /// Loads the library \p plugin_file_name from the \p paths locations
        /// \param plugin_file_name  The name of the library without any prefix or
        ///                          extensions
        /// \param paths             The locations to search for the libraries if
        ///                          not found in standard locations
        Module(const std::string &plugin_file_name,
               const std::vector<std::string> &paths = {});

        Module(
                const std::string &plugin_base_file_name,
                const std::vector<std::string> &suffixes,
                const std::vector<std::string> &paths,
                const std::vector<ModuleVersion> &versions = {},
                std::function<ModuleVersion(const ModuleHandle&)> f = {});
        Module() = default;
        ~Module() noexcept;

        Module(Module &&m) noexcept;

        Module &operator=(Module &&m) noexcept;

        /// Returns a function pointer to the function with the name symbol_name
        template<typename T>
        T get_symbol(const char *symbol_name) {
            _functions.push_back(ModuleLoader::get_function_pointer(_handle, symbol_name));
            return (T) _functions.back();
        }

        /// Returns true if the module was successfully loaded
        [[nodiscard]] bool is_loaded() const noexcept;

        /// Returns true if all of the symbols for the module were loaded
        [[nodiscard]] bool symbols_loaded() const noexcept;

        /// Returns the version of the module
        [[nodiscard]] ModuleVersion get_version() const noexcept { return _version; }

        [[nodiscard]] std::string get_prefix() const noexcept { return _prefix_path; }

        /// Returns the last error message that occurred because of loading the
        /// library
        static std::string get_error_message() noexcept;

        static Module load_module(const std::string &plugin_file_name);

        static Module load_module(const std::string &plugin_file_name, const ModuleVersion & version);

        static Module load_module(const std::string &plugin_file_name, const ModuleVersion & version, const std::vector<std::string> &paths);

    protected:
        // nolint
        TURBO_NON_COPYABLE(Module);
        ModuleHandle _handle{nullptr};
        std::vector<void *> _functions;
        ModuleVersion _version{kNullVersion};
        std::string _prefix_path;

    };

}  // namespace turbo

/// Creates a function pointer
#define MODULE_MEMBER(NAME) decltype(&::NAME) NAME

/// Dynamically loads the function pointer at runtime
#define MODULE_FUNCTION_INIT(m, NAME) \
    NAME = m.get_symbol<decltype(&::NAME)>(#NAME);

#endif  // TURBO_MODULE_MODULE_H_
