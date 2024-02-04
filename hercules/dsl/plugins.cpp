// Copyright 2023 The titan-search Authors.
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

#include "plugins.h"

#include <cstdlib>
#include <toml++/toml.h>
#include "hercules/dsl/semver.hpp"
#include "hercules/parser/common.h"
#include "hercules/util/common.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace hercules {
namespace {
llvm::Expected<Plugin *> pluginError(const std::string &msg) {
  return llvm::make_error<error::PluginErrorInfo>(msg);
}

typedef std::unique_ptr<DSL> LoadFunc();
} // namespace

llvm::Expected<Plugin *> PluginManager::load(const std::string &path) {
#if __APPLE__
  const std::string libExt = "dylib";
#else
  const std::string libExt = "so";
#endif

  const std::string config = "plugin.toml";

  llvm::SmallString<128> tomlPath(path);
  llvm::sys::path::append(tomlPath, config);
  if (!llvm::sys::fs::exists(tomlPath)) {
    // try default install path
    tomlPath = llvm::SmallString<128>(
        llvm::sys::path::parent_path(ast::executable_path(argv0.c_str())));
    llvm::sys::path::append(tomlPath, "../lib/hercules/plugins", path, config);
  }

  toml::parse_result tml;
  try {
    tml = toml::parse_file(tomlPath.str());
  } catch (const toml::parse_error &e) {
    return pluginError(
        fmt::format("[toml::parse_file(\"{}\")] {}", tomlPath.str(), e.what()));
  }
  auto about = tml["about"];
  auto library = tml["library"];

  std::string cppLib = library["cpp"].value_or("");
  std::string dylibPath;
  if (!cppLib.empty()) {
    llvm::SmallString<128> p = llvm::sys::path::parent_path(tomlPath);
    llvm::sys::path::append(p, cppLib + "." + libExt);
    dylibPath = p.str();
  }

  auto link = library["link"];
  std::vector<std::string> linkArgs;
  if (auto arr = link.as_array()) {
    arr->for_each([&linkArgs](auto &&el) {
      std::string l = el.value_or("");
      if (!l.empty())
        linkArgs.push_back(l);
    });
  } else {
    std::string l = link.value_or("");
    if (!l.empty())
      linkArgs.push_back(l);
  }
  for (auto &l : linkArgs)
    l = fmt::format(l, fmt::arg("root", llvm::sys::path::parent_path(tomlPath)));

  std::string herculesLib = library["hercules"].value_or("");
  std::string stdlibPath;
  if (!herculesLib.empty()) {
    llvm::SmallString<128> p = llvm::sys::path::parent_path(tomlPath);
    llvm::sys::path::append(p, herculesLib);
    stdlibPath = p.str();
  }

  DSL::Info info = {about["name"].value_or(""),
                    about["description"].value_or(""),
                    about["version"].value_or(""),
                    about["url"].value_or(""),
                    about["supported"].value_or(""),
                    stdlibPath,
                    dylibPath,
                    linkArgs};

  bool versionOk = false;
  try {
    versionOk = semver::range::satisfies(
        semver::version(HERCULES_VERSION_MAJOR, HERCULES_VERSION_MINOR, HERCULES_VERSION_PATCH),
        info.supported);
  } catch (const std::invalid_argument &e) {
    return pluginError(fmt::format("[semver::range::satisfies(..., \"{}\")] {}",
                                   info.supported, e.what()));
  }
  if (!versionOk)
    return pluginError(fmt::format("unsupported version {} (supported: {})",
                                   HERCULES_VERSION, info.supported));

  if (!dylibPath.empty()) {
    std::string libLoadErrorMsg;
    auto handle = llvm::sys::DynamicLibrary::getPermanentLibrary(dylibPath.c_str(),
                                                                 &libLoadErrorMsg);
    if (!handle.isValid())
      return pluginError(fmt::format(
          "[llvm::sys::DynamicLibrary::getPermanentLibrary(\"{}\", ...)] {}", dylibPath,
          libLoadErrorMsg));

    auto *entry = (LoadFunc *)handle.getAddressOfSymbol("load");
    if (!entry)
      return pluginError(
          fmt::format("could not find 'load' in plugin shared library: {}", dylibPath));

    auto dsl = (*entry)();
    plugins.push_back(std::make_unique<Plugin>(std::move(dsl), info, handle));
  } else {
    plugins.push_back(std::make_unique<Plugin>(std::make_unique<DSL>(), info,
                                               llvm::sys::DynamicLibrary()));
  }
  return plugins.back().get();
}

} // namespace hercules
