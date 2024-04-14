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
#include "turbo/module/module.h"
#include "turbo/log/logging.h"
#include "turbo/files/filesystem.h"
#include "turbo/module/constants.h"

namespace turbo {


    Module::Module(const std::string &plugin_file_name,
                   const std::vector<std::string> &paths) {
        if (plugin_file_name.empty()) {
            return;
        }

        // try rpath
        auto rpath_m = load_module(plugin_file_name);
        if (rpath_m.is_loaded()) {
            *this = std::move(rpath_m);
            return;
        }

        auto path_m = load_module(plugin_file_name, kNullVersion, paths);
        if (path_m.is_loaded()) {
            *this = std::move(path_m);
            return;
        }

        TLOG_ERROR("Unable to open {}", plugin_file_name);

    }

    Module::Module(
            const std::string &plugin_file_name, const std::vector<std::string> &suffixes,
            const std::vector<std::string> &paths,
            const std::vector<ModuleVersion> &versions, std::function<ModuleVersion(const ModuleHandle&)> f) {
        TURBO_UNUSED(suffixes);
        if (plugin_file_name.empty()) {
            return;
        }

        // try rpath
        for (auto &v: versions) {
            auto rpath_m = load_module(plugin_file_name, v);
            if (rpath_m.is_loaded()) {
                *this = std::move(rpath_m);
                return;
            }
        }
        // try rpath
        for (auto &v: versions) {
            auto path_m = load_module(plugin_file_name, v, paths);
            if (path_m.is_loaded()) {
                *this = std::move(path_m);
                return;
            }
        }
        // version empty path empty
        auto m = load_module(plugin_file_name, kNullVersion);
        if (m.is_loaded() && f) {
            m._version = f(m._handle);
        }
        if (m.is_loaded()) {
            *this = std::move(m);
            return;
        }

        m = load_module(plugin_file_name, kNullVersion, paths);
        if (m.is_loaded() && f) {
            m._version = f(m._handle);
        }
        if (m.is_loaded()) {
            *this = std::move(m);
            return;
        }
        TLOG_ERROR("Unable to open {}", plugin_file_name);
    }

    Module::~Module() noexcept {
        if (_handle) { ModuleLoader::unload_library(_handle); }
    }

    Module::Module(Module &&m) noexcept {
        std::swap(_handle, m._handle);
        std::swap(_functions, m._functions);
        std::swap(_version, m._version);
        std::swap(_prefix_path, m._prefix_path);
    }

    Module &Module::operator=(Module &&m) noexcept {
        std::swap(_handle, m._handle);
        std::swap(_functions, m._functions);
        std::swap(_version, m._version);
        std::swap(_prefix_path, m._prefix_path);
        return *this;
    }

    bool Module::is_loaded() const noexcept {
        return static_cast<bool>(_handle);
    }

    bool Module::symbols_loaded() const noexcept {
        return std::all_of(begin(_functions), end(_functions),
                           [](void *ptr) { return ptr != nullptr; });
    }

    std::string Module::get_error_message() noexcept {
        return ModuleLoader::get_error_message();
    }

    Module Module::load_module(const std::string &plugin_file_name) {
        auto fileNames = ModuleLoader::module_name(plugin_file_name);
        auto handle = ModuleLoader::load_library(fileNames[0].c_str());
        Module m;
        m._handle = handle;
        return m;
    }

    Module Module::load_module(const std::string &plugin_file_name, const ModuleVersion &version) {
        auto fileNames = ModuleLoader::module_name(plugin_file_name, "", version);
        auto handle = ModuleLoader::load_library(fileNames[0].c_str());
        Module m;
        m._handle = handle;
        return m;
    }

    Module Module::load_module(const std::string &plugin_file_name, const ModuleVersion &version,
                               const std::vector<std::string> &paths) {
        auto fileNames = ModuleLoader::module_name(plugin_file_name, "", version);
        Module m;
        m._version = version;
        for (auto &path: paths) {
            turbo::filesystem::path sopath(path);
            sopath /= fileNames[0];
            auto handle = ModuleLoader::load_library(fileNames[0].c_str());
            if (!handle) {
                continue;
            }
            m._handle = handle;
            m._prefix_path = sopath.string();
            break;
        }
        return m;
    }

}  // namespace turbo
