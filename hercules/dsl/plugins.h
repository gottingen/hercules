// Copyright 2024 The EA Authors.
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

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <hercules/hir/util/iterators.h>
#include <hercules/compiler/error.h>
#include <hercules/dsl/dsl.h>
#include <llvm/Support/DynamicLibrary.h>

namespace hercules {

    /// Plugin metadata
    struct Plugin {
        /// the associated DSL
        std::unique_ptr<DSL> dsl;
        /// plugin information
        DSL::Info info;
        /// library handle
        llvm::sys::DynamicLibrary handle;

        Plugin(std::unique_ptr<DSL> dsl, DSL::Info info, llvm::sys::DynamicLibrary handle)
                : dsl(std::move(dsl)), info(std::move(info)), handle(std::move(handle)) {}
    };

    /// Manager for loading, applying and unloading plugins.
    class PluginManager {
    private:
        /// Hercules executable location
        std::string argv0;
        /// vector of loaded plugins
        std::vector<std::unique_ptr<Plugin>> plugins;

    public:
        /// Constructs a plugin manager
        PluginManager(const std::string &argv0) : argv0(argv0), plugins() {}

        /// @return iterator to the first plugin
        auto begin() { return ir::util::raw_ptr_adaptor(plugins.begin()); }

        /// @return iterator beyond the last plugin
        auto end() { return ir::util::raw_ptr_adaptor(plugins.end()); }

        /// @return const iterator to the first plugin
        auto begin() const { return ir::util::const_raw_ptr_adaptor(plugins.begin()); }

        /// @return const iterator beyond the last plugin
        auto end() const { return ir::util::const_raw_ptr_adaptor(plugins.end()); }

        /// Loads the plugin at the given load path.
        /// @param path path to plugin directory containing "plugin.toml" file
        /// @return plugin pointer if successful, plugin error otherwise
        llvm::Expected<Plugin *> load(const std::string &path);
    };

} // namespace hercules
