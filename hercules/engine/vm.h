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
#include <iostream>
#include <hercules/compiler/compiler.h>
#include <hercules/compiler/jit.h>
#include <llvm/Support/CodeGen.h>
#include <collie/cli/cli.h>

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

    void version_dump(std::ostream &out);

    inline std::string get_os_lib_extension() {
        if (is_mac_os()) {
            return ".dylib";
        } else if(is_linux_os()) {
            return ".so";
        }
    }

    struct VmContext{
        static VmContext & instance(){
            static VmContext ctx;
            return ctx;
        }
        std::vector<const char *> args;
        std::vector<const char *> llvm_args;
        std::string orig_argv0;
        std::string mode;
        std::string argv0;
        OptMode opt_mode{OptMode::Debug};
        std::vector<std::string> defines;
        std::vector<std::string> disabled_opts;
        std::vector<std::string> plugins;
        std::string log;
        Numerics numeric{Numerics::C};
        std::vector<std::string> libs;
        std::string flags;
        std::vector<std::string> prog_args;
        std::string output;
        BuildKind build_kind{BuildKind::Detect};
        std::string py_module;
        llvm::Reloc::Model reloc_model{llvm::Reloc::Model::Static};
        std::vector<std::string> llvm_flags;
        std::string input = "-";
    };

    const std::vector<std::string> &supported_extensions();

    bool has_extension(const std::string &filename, const std::string &extension);

    std::string trim_extension(const std::string &filename, const std::string &extension);

    std::string make_output_filename(const std::string &filename, const std::string &extension);

    void init_log_flags(const std::string &log);

    void set_up_run_command(collie::App* app);

    void set_up_build_command(collie::App* app);

    void set_up_doc_command(collie::App* app);
    void set_up_jit_command(collie::App* app);
    bool tidy_program_args();

    class EngineVM {
    public:
        EngineVM() = default;
        ~EngineVM() = default;

        int prepare_run();

        int run();

        int document();

        int build(const std::string &argv0);

        int jit();

    private:
        bool process_source(bool standalone, std::function<bool()> pyExtension = [] { return false; });

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
