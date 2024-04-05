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

#include <hercules/engine/vm.h>
#include <hercules/util/common.h>
#include <llvm/Support/CommandLine.h>
#include <iostream>

namespace hercules {

    const std::vector<std::string> &supported_extensions() {
        static const std::vector<std::string> extensions = {".hs", ".py", ".seq"};
        return extensions;
    }

    bool has_extension(const std::string &filename, const std::string &extension) {
        return filename.size() >= extension.size() &&
               filename.compare(filename.size() - extension.size(), extension.size(),
                                extension) == 0;
    }

    std::string trim_extension(const std::string &filename, const std::string &extension) {
        if (has_extension(filename, extension)) {
            return filename.substr(0, filename.size() - extension.size());
        } else {
            return filename;
        }
    }

    std::string make_output_filename(const std::string &filename,
                                     const std::string &extension) {
        for (const auto &ext: hercules::supported_extensions()) {
            if (has_extension(filename, ext))
                return trim_extension(filename, ext) + extension;
        }
        return filename + extension;
    }

    void init_log_flags(const std::string &log) {
        hercules::getLogger().parse(log);
        if (auto *d = getenv("HERCULES_DEBUG"))
            hercules::getLogger().parse(std::string(d));
    }

    void display(const hercules::error::ParserErrorInfo &e) {
        using hercules::MessageGroupPos;
        for (auto &group: e) {
            for (auto &msg: group) {
                MessageGroupPos pos = MessageGroupPos::NONE;
                if (&msg == &group.front()) {
                    pos = MessageGroupPos::HEAD;
                } else if (&msg == &group.back()) {
                    pos = MessageGroupPos::LAST;
                } else {
                    pos = MessageGroupPos::MID;
                }
                hercules::compilationError(msg.getMessage(), msg.getFile(), msg.getLine(),
                                           msg.getColumn(), msg.getLength(), msg.getErrorCode(),
                        /*terminate=*/false, pos);
            }
        }
    }

    bool tidy_program_args() {
        auto &ins = hercules::VmContext::instance();
        if (ins.prog_args.empty()) {
            return false;
        }
        ins.input = ins.prog_args[0];
        ins.prog_args.erase(ins.prog_args.begin());
        std::map <llvm::Reloc::Model, std::string> reloc_map{
                {llvm::Reloc::Model::Static, "static"},
                {llvm::Reloc::Model::PIC_, "pic"},
                {llvm::Reloc::Model::DynamicNoPIC, "dpic"},
                {llvm::Reloc::Model::ROPI, "ropi"},
                {llvm::Reloc::Model::RWPI, "rwpi"},
                {llvm::Reloc::Model::ROPI_RWPI, "ropi-rwpi"}
        };
        auto it = reloc_map.find(ins.reloc_model);
        if (it != reloc_map.end()) {
             auto reloc_model_str = "--relocation-model=" + it->second;
            ins.llvm_flags.push_back(reloc_model_str);
        }

        ins.llvm_args.push_back(ins.args[0]);
        if(!ins.llvm_flags.empty()) {
            for (const auto &arg: ins.llvm_flags) {
                ins.llvm_args.push_back(arg.c_str());
            }
        }
        // must run it, for the llvm::cl::opt to be initialized
        llvm::cl::ParseCommandLineOptions(ins.llvm_args.size(), ins.llvm_args.data());
        return true;
    }

    static void set_up_process_command(collie::App *app) {
        auto &ins = hercules::VmContext::instance();
        std::map<std::string, OptMode> opt_map{{"debug",   OptMode::Debug},
                                               {"release", OptMode::Release}};
        std::map<std::string, Numerics> numeric_map{{"c",  Numerics::C},
                                                    {"py", Numerics::Python}};
        app->add_option("-m, --mode", ins.opt_mode, "optimization mode")->transform(
                collie::CheckedTransformer(opt_map, collie::ignore_case));
        app->add_option("-D, --define", ins.defines, "Add static variable definitions. The syntax is <name>=<value>");
        app->add_option("-d, --disable-opt", ins.disabled_opts, "Disable the specified IR optimization");
        app->add_option("-p, --plugin", ins.plugins, "Load specified plugin");
        app->add_option("--log", ins.log, "enable log \'t\' for time, \'r\' for realize, \'T\' for typecheck, \'i\' for IR, \'l\' for user log\n"
                                          "for example --log Ti will enable typecheck and IR log");
        app->add_option("-n, --numeric", ins.numeric, "numerical semantics")->transform(
                collie::CheckedTransformer(numeric_map, collie::ignore_case));
        app->add_option("-l, --lib", ins.libs, "Link the specified library");
    }

    static void run_run_command() {
        auto &ins = hercules::VmContext::instance();
        hercules::EngineVM vm;
        ins.ret_code = vm.prepare_run();
        if (ins.ret_code != EXIT_SUCCESS)
            return ;
        ins.ret_code =  vm.run();
    }

    void set_up_run_command(collie::App *app) {
        set_up_process_command(app);
        app->add_option("prog_args", hercules::VmContext::instance().prog_args, "program arguments");
        app->callback([]() {
            hercules::VmContext::instance().mode = "run";
            hercules::VmContext::instance().argv0 = hercules::VmContext::instance().orig_argv0 + " run";
            run_run_command();
        });
    }
    static void run_build_command() {
        auto &ins = hercules::VmContext::instance();
        hercules::EngineVM vm;
        ins.ret_code = vm.build(ins.orig_argv0);
    }

    void set_up_build_command(collie::App *app) {
        set_up_process_command(app);
        app->add_option("prog_args", hercules::VmContext::instance().prog_args, "program arguments");
        app->add_option("-F, --flags", hercules::VmContext::instance().flags, "compiler flags");
        app->add_option("-o, --output", hercules::VmContext::instance().output, "output file");
        std::map<std::string, BuildKind> build_map{
                {"llvm", BuildKind::LLVM},
                {"bc",   BuildKind::Bitcode},
                {"obj",  BuildKind::Object},
                {"exe",  BuildKind::Executable},
                {"lib",  BuildKind::Library},
                {"pyext", BuildKind::PyExtension},
                {"detect", BuildKind::Detect}};
        app->add_option("-k, --kind", hercules::VmContext::instance().build_kind, "output type")->transform(
                collie::CheckedTransformer(build_map, collie::ignore_case));
        app->add_option("-y, --py_module", hercules::VmContext::instance().py_module, "Python extension module name");
        std::map<std::string, llvm::Reloc::Model> reloc_map{
                {"static", llvm::Reloc::Model::Static},
                {"pic",    llvm::Reloc::Model::PIC_},
                {"dpic", llvm::Reloc::Model::DynamicNoPIC},
                {"ropi", llvm::Reloc::Model::ROPI},
                {"rwpi", llvm::Reloc::Model::RWPI},
                {"ropi-rwpi", llvm::Reloc::Model::ROPI_RWPI}
        };
        app->add_option("-r, --relocation", hercules::VmContext::instance().reloc_model, "relocation model")->transform(
                collie::CheckedTransformer(reloc_map, collie::ignore_case));
        app->callback([]() {
            hercules::VmContext::instance().mode = "build";
            hercules::VmContext::instance().argv0 = hercules::VmContext::instance().orig_argv0 + " build";
            run_build_command();
        });
    }

    void set_up_doc_command(collie::App *app) {

    }

    static void run_jit_command() {
        auto &ins = hercules::VmContext::instance();
        hercules::EngineVM vm;
        ins.ret_code = vm.jit();
    }

    void set_up_jit_command(collie::App *app) {
        set_up_process_command(app);
        app->add_option("prog_args", hercules::VmContext::instance().prog_args, "program arguments");
        app->callback([](){
            hercules::VmContext::instance().mode = "jit";
            hercules::VmContext::instance().argv0 = hercules::VmContext::instance().orig_argv0 + " jit";
            run_jit_command();
        });
    }

    int EngineVM::prepare_run() {
        _valid = process_source(false);
        if (!_valid)
            return EXIT_FAILURE;
        auto &ins = hercules::VmContext::instance();
        _libs = ins.libs;
        _prog_args = ins.prog_args;
        _prog_args.insert(_prog_args.begin(), _compiler->getInput());
        return EXIT_SUCCESS;
    }

    int EngineVM::run() {
        if (!_valid)
            return EXIT_FAILURE;
        _compiler->getLLVMVisitor()->run(_prog_args, _libs);
        return EXIT_SUCCESS;
    }


    bool EngineVM::process_source(bool standalone, std::function<bool()> pyExtension) {
        auto &ins = hercules::VmContext::instance();
        init_log_flags(ins.log);

        std::unordered_map<std::string, std::string> defmap;
        for (const auto &define: ins.defines) {
            auto eq = define.find('=');
            if (eq == std::string::npos || !eq) {
                hercules::compilationWarning("ignoring malformed definition: " + define);
                continue;
            }

            auto name = define.substr(0, eq);
            auto value = define.substr(eq + 1);

            if (defmap.find(name) != defmap.end()) {
                hercules::compilationWarning("ignoring duplicate definition: " + define);
                continue;
            }

            defmap.emplace(name, value);
        }

        const bool isDebug = (ins.opt_mode == OptMode::Debug);
        std::vector<std::string> disabledOptsVec(ins.disabled_opts);
        _compiler = std::make_unique<hercules::Compiler>(ins.argv0, isDebug, disabledOptsVec,
                /*isTest=*/false, (ins.numeric == Numerics::Python), pyExtension());
        _compiler->getLLVMVisitor()->setStandalone(standalone);

        // load plugins
        bool failed = false;
        llvm::handleAllErrors(
                _compiler->load_builtin(), [&failed](const hercules::error::PluginErrorInfo &e) {
                    hercules::compilationError(e.getMessage(), /*file=*/"",
                            /*line=*/0, /*col=*/0, /*len*/ 0, /*errorCode*/ -1,
                            /*terminate=*/false);
                    failed = true;
                });
        if (failed) {
            return false;
        }
        for (const auto &plugin: ins.plugins) {
            failed = false;
            llvm::handleAllErrors(
                    _compiler->load(plugin), [&failed](const hercules::error::PluginErrorInfo &e) {
                        hercules::compilationError(e.getMessage(), /*file=*/"",
                                /*line=*/0, /*col=*/0, /*len*/ 0, /*errorCode*/ -1,
                                /*terminate=*/false);
                        failed = true;
                    });
            if (failed) {
                return false;
            }
        }

        failed = false;
        int testFlags = 0;
        if (auto *tf = getenv("HERCULES_TEST_FLAGS"))
            testFlags = std::atoi(tf);
        llvm::handleAllErrors(_compiler->parseFile(ins.input, /*testFlags=*/testFlags, defmap),
                              [&failed](const hercules::error::ParserErrorInfo &e) {
                                  display(e);
                                  failed = true;
                              });
        if (failed) {
            return false;
        }

        {
            TIME("compile");
            llvm::cantFail(_compiler->compile());
        }
        return true;
    }

    int EngineVM::document() {
        auto &ins = hercules::VmContext::instance();
        std::vector<std::string> files;
        for (std::string line; std::getline(std::cin, line);)
            files.push_back(line);

        _compiler = std::make_unique<hercules::Compiler>(ins.argv0);
        bool failed = false;
        auto result = _compiler->docgen(files);
        llvm::handleAllErrors(result.takeError(),
                              [&failed](const hercules::error::ParserErrorInfo &e) {
                                  display(e);
                                  failed = true;
                              });
        if (failed)
            return EXIT_FAILURE;

        collie::print("{}\n", *result);
        return EXIT_SUCCESS;
    }


    int EngineVM::build(const std::string &argv0) {
        auto &ins = hercules::VmContext::instance();

        _valid = process_source(/*standalone=*/true,
                                               [&] { return ins.build_kind == BuildKind::PyExtension; });
        if (!_valid)
            return EXIT_FAILURE;
        std::vector<std::string> libsVec(ins.libs);

        if (ins.output.empty() && _compiler->getInput() == "-")
            hercules::compilationError("output file must be specified when reading from stdin");
        std::string extension;
        switch (ins.build_kind) {
            case BuildKind::LLVM:
                extension = ".ll";
                break;
            case BuildKind::Bitcode:
                extension = ".bc";
                break;
            case BuildKind::Object:
            case BuildKind::PyExtension:
                extension = ".o";
                break;
            case BuildKind::Library:
                extension = hercules::get_os_lib_extension();
                break;
            case BuildKind::Executable:
            case BuildKind::Detect:
                extension = "";
                break;
            default:
                seqassertn(0, "unknown build kind");
        }
        const std::string filename =
                ins.output.empty() ? hercules::make_output_filename(_compiler->getInput(), extension) : ins.output;
        switch (ins.build_kind) {
            case BuildKind::LLVM:
                _compiler->getLLVMVisitor()->writeToLLFile(filename);
                break;
            case BuildKind::Bitcode:
                _compiler->getLLVMVisitor()->writeToBitcodeFile(filename);
                break;
            case BuildKind::Object:
                _compiler->getLLVMVisitor()->writeToObjectFile(filename);
                break;
            case BuildKind::Executable:
                _compiler->getLLVMVisitor()->writeToExecutable(filename, argv0, false, libsVec,
                                                               ins.flags);
                break;
            case BuildKind::Library:
                _compiler->getLLVMVisitor()->writeToExecutable(filename, argv0, true, libsVec,
                                                               ins.flags);
                break;
            case BuildKind::PyExtension:
                _compiler->getCache()->pyModule->name =
                        ins.py_module.empty() ? llvm::sys::path::stem(_compiler->getInput()).str() : ins.py_module;
                _compiler->getLLVMVisitor()->writeToPythonExtension(*_compiler->getCache()->pyModule,
                                                                    filename);
                break;
            case BuildKind::Detect:
                _compiler->getLLVMVisitor()->compile(filename, argv0, libsVec,ins.flags);
                break;
            default:
                seqassertn(0, "unknown build kind");
        }

        return EXIT_SUCCESS;
    }


    int EngineVM::jit() {
        auto &ins = hercules::VmContext::instance();
        init_log_flags(ins.log);
        hercules::jit::JIT jit(ins.argv0);

        // load plugins
        bool failed = false;
        llvm::handleAllErrors(jit.getCompiler()->load_builtin(),
                              [&failed](const hercules::error::PluginErrorInfo &e) {
                                  hercules::compilationError(e.getMessage(), /*file=*/"",
                                          /*line=*/0, /*col=*/0, /*len=*/0,
                                          /*errorCode*/ -1,
                                          /*terminate=*/false);
                                  failed = true;
                              });
        if (failed)
            return EXIT_FAILURE;
        for (const auto &plugin: ins.plugins) {
            failed = false;
            llvm::handleAllErrors(jit.getCompiler()->load(plugin),
                                  [&failed](const hercules::error::PluginErrorInfo &e) {
                                      hercules::compilationError(e.getMessage(), /*file=*/"",
                                              /*line=*/0, /*col=*/0, /*len=*/0,
                                              /*errorCode*/ -1,
                                              /*terminate=*/false);
                                      failed = true;
                                  });
            if (failed)
                return EXIT_FAILURE;
        }

        llvm::cantFail(jit.init());
        collie::print(">>> Hercules JIT v{} <<<\n", HERCULES_VERSION);
        if (ins.input == "-") {
            jit_loop(&jit, std::cin);
        } else {
            std::ifstream fileInput(ins.input);
            jit_loop(&jit, fileInput);
        }
        return EXIT_SUCCESS;
    }

    std::string EngineVM::jit_exec(hercules::jit::JIT *jit, const std::string &code) {
        auto result = jit->execute(code);
        if (auto err = result.takeError()) {
            std::string output;
            llvm::handleAllErrors(
                    std::move(err), [](const hercules::error::ParserErrorInfo &e) { display(e); },
                    [&output](const hercules::error::RuntimeErrorInfo &e) {
                        std::stringstream buf;
                        buf << e.getOutput();
                        buf << "\n\033[1mBacktrace:\033[0m\n";
                        for (const auto &line: e.getBacktrace()) {
                            buf << "  " << line << "\n";
                        }
                        output = buf.str();
                    });
            return output;
        }
        return *result;
    }

    void EngineVM::jit_loop(hercules::jit::JIT *jit, std::istream &fp) {
        std::string code;
        for (std::string line; std::getline(fp, line);) {
            if (line != "#%%") {
                code += line + "\n";
            } else {
                collie::print("{}[done]\n", jit_exec(jit, code));
                code = "";
                fflush(stdout);
            }
        }
        if (!code.empty())
            collie::print("{}[done]\n", jit_exec(jit, code));
    }

}  // namespace hercules
