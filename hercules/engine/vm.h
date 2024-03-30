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

#ifndef HERCULES_ENGINE_VM_H_
#define HERCULES_ENGINE_VM_H_

#include <vector>
#include <string>
#include <llvm/Support/CommandLine.h>
#include <hercules/compiler/compiler.h>
#include <hercules/compiler/jit.h>

namespace hercules {

    enum BuildKind {
        LLVM, Bitcode, Object, Executable, Library, PyExtension, Detect
    };
    enum OptMode {
        Debug, Release
    };
    enum Numerics {
        C, Python
    };

    constexpr bool is_mac_os() {
#ifdef __APPLE__
        return true;
#else
        return false;
#endif
    }

    constexpr bool is_linux_os() {
#ifdef __linux__
        return true;
#else
        return false;
#endif
    }

    void version_dump(llvm::raw_ostream &out);

    inline std::string get_os_lib_extension() {
        if (is_mac_os()) {
            return ".dylib";
        } else if(is_linux_os()) {
            return ".so";
        }
    }

    const std::vector<std::string> &supported_extensions();

    bool has_extension(const std::string &filename, const std::string &extension);

    std::string trim_extension(const std::string &filename, const std::string &extension);

    std::string make_output_filename(const std::string &filename, const std::string &extension);

    void init_log_flags(const llvm::cl::opt<std::string> &log);

    class EngineVM {
    public:
        EngineVM() = default;
        ~EngineVM() = default;

        int prepare_run(std::vector<const char *> &args);

        int run();

        int document(const std::vector<const char *> &args, const std::string &argv0);

        int build(const std::vector<const char *> &args, const std::string &argv0);

        int jit(const std::vector<const char *> &args);

    private:
        bool process_source(
                const std::vector<const char *> &args, bool standalone,
                std::function<bool()> pyExtension = [] { return false; });

        std::string jit_exec(hercules::jit::JIT *jit, const std::string &code);

        void jit_loop(hercules::jit::JIT *jit, std::istream &fp);

    private:
        std::unique_ptr<hercules::Compiler> _compiler;
        std::vector<std::string>            _libs;
        std::vector<std::string>            _prog_args;
        bool                                _standalone{false};
        bool                                _valid{true};
        bool                                _pyExtension{false};
    };
}  // namespace hercules

#endif // HERCULES_ENGINE_VM_H_
