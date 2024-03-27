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

#include "hercules/engine/vm.h"
#include "hercules/util/common.h"

namespace hercules {

    void version_dump(llvm::raw_ostream &out) {
        out << HERCULES_VERSION_MAJOR << "." << HERCULES_VERSION_MINOR << "." << HERCULES_VERSION_PATCH
            << "\n";
    }

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

    void init_log_flags(const llvm::cl::opt<std::string> &log) {
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

    int EngineVM::prepare_run(std::vector<const char *> &args) {
        llvm::cl::list<std::string> libs(
                "l", llvm::cl::desc("Load and link the specified library"));
        llvm::cl::list<std::string> progArgs(llvm::cl::ConsumeAfter,
                                             llvm::cl::desc("<program arguments>..."));
        _valid = process_source(args, /*standalone=*/false);
        if (!_valid)
            return EXIT_FAILURE;
        std::vector<std::string> libsVec(libs);
        std::vector<std::string> argsVec(progArgs);
        _libs = std::move(libsVec);
        _prog_args = std::move(argsVec);
        _prog_args.insert(_prog_args.begin(), _compiler->getInput());
        return EXIT_SUCCESS;
    }

    int EngineVM::run() {
        if (!_valid)
            return EXIT_FAILURE;
        _compiler->getLLVMVisitor()->run(_prog_args, _libs);
        return EXIT_SUCCESS;
    }


    bool EngineVM::process_source(
            const std::vector<const char *> &args, bool standalone,
            std::function<bool()> pyExtension) {
        llvm::cl::opt<std::string> input(llvm::cl::Positional, llvm::cl::desc("<input file>"),
                                         llvm::cl::init("-"));
        auto regs = llvm::cl::getRegisteredOptions();
        llvm::cl::opt<OptMode> optMode(
                llvm::cl::desc("optimization mode"),
                llvm::cl::values(
                        clEnumValN(Debug, regs.find("debug") != regs.end() ? "default" : "debug",
                                   "Turn off compiler optimizations and show backtraces"),
                        clEnumValN(Release, "release",
                                   "Turn on compiler optimizations and disable debug info")),
                llvm::cl::init(Debug));
        llvm::cl::list<std::string> defines(
                "D", llvm::cl::Prefix,
                llvm::cl::desc("Add static variable definitions. The syntax is <name>=<value>"));
        llvm::cl::list<std::string> disabledOpts(
                "disable-opt", llvm::cl::desc("Disable the specified IR optimization"));
        llvm::cl::list<std::string> plugins("plugin",
                                            llvm::cl::desc("Load specified plugin"));
        llvm::cl::opt<std::string> log("log", llvm::cl::desc("Enable given log streams"));
        llvm::cl::opt<Numerics> numerics(
                "numerics", llvm::cl::desc("numerical semantics"),
                llvm::cl::values(
                        clEnumValN(C, "c", "C semantics: best performance but deviates from Python"),
                        clEnumValN(Python, "py",
                                   "Python semantics: mirrors Python but might disable optimizations "
                                   "like vectorization")),
                llvm::cl::init(C));

        llvm::cl::ParseCommandLineOptions(args.size(), args.data());
        init_log_flags(log);

        std::unordered_map<std::string, std::string> defmap;
        for (const auto &define: defines) {
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

        const bool isDebug = (optMode == OptMode::Debug);
        std::vector<std::string> disabledOptsVec(disabledOpts);
        _compiler = std::make_unique<hercules::Compiler>(
                args[0], isDebug, disabledOptsVec,
                /*isTest=*/false, (numerics == Numerics::Python), pyExtension());
        _compiler->getLLVMVisitor()->setStandalone(standalone);

        // load plugins
        for (const auto &plugin: plugins) {
            bool failed = false;
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

        bool failed = false;
        int testFlags = 0;
        if (auto *tf = getenv("HERCULES_TEST_FLAGS"))
            testFlags = std::atoi(tf);
        llvm::handleAllErrors(_compiler->parseFile(input, /*testFlags=*/testFlags, defmap),
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

    int EngineVM::document(const std::vector<const char *> &args, const std::string &argv0) {
        llvm::cl::ParseCommandLineOptions(args.size(), args.data());
        std::vector<std::string> files;
        for (std::string line; std::getline(std::cin, line);)
            files.push_back(line);

        _compiler = std::make_unique<hercules::Compiler>(args[0]);
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


    int EngineVM::build(const std::vector<const char *> &args, const std::string &argv0) {
        llvm::cl::list<std::string> libs(
                "l", llvm::cl::desc("Link the specified library (only for executables)"));
        llvm::cl::opt<std::string> lflags("linker-flags",
                                          llvm::cl::desc("Pass given flags to linker"));
        llvm::cl::opt<BuildKind> buildKind(
                llvm::cl::desc("output type"),
                llvm::cl::values(
                        clEnumValN(LLVM, "llvm", "Generate LLVM IR"),
                        clEnumValN(Bitcode, "bc", "Generate LLVM bitcode"),
                        clEnumValN(Object, "obj", "Generate native object file"),
                        clEnumValN(Executable, "exe", "Generate executable"),
                        clEnumValN(Library, "lib", "Generate shared library"),
                        clEnumValN(PyExtension, "pyext", "Generate Python extension module"),
                        clEnumValN(Detect, "detect",
                                   "Detect output type based on output file extension")),
                llvm::cl::init(Detect));
        llvm::cl::opt<std::string> output(
                "o",
                llvm::cl::desc(
                        "Write compiled output to specified file. Supported extensions: "
                        "none (executable), .o (object file), .ll (LLVM IR), .bc (LLVM bitcode)"));
        llvm::cl::opt<std::string> pyModule(
                "module", llvm::cl::desc("Python extension module name (only applicable when "
                                         "building Python extension module)"));

        _valid = process_source(args, /*standalone=*/true,
                                      [&] { return buildKind == BuildKind::PyExtension; });
        if (!_valid)
            return EXIT_FAILURE;
        std::vector<std::string> libsVec(libs);

        if (output.empty() && _compiler->getInput() == "-")
            hercules::compilationError("output file must be specified when reading from stdin");
        std::string extension;
        switch (buildKind) {
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
                output.empty() ? hercules::make_output_filename(_compiler->getInput(), extension) : output;
        switch (buildKind) {
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
                                                              lflags);
                break;
            case BuildKind::Library:
                _compiler->getLLVMVisitor()->writeToExecutable(filename, argv0, true, libsVec,
                                                              lflags);
                break;
            case BuildKind::PyExtension:
                _compiler->getCache()->pyModule->name =
                        pyModule.empty() ? llvm::sys::path::stem(_compiler->getInput()).str() : pyModule;
                _compiler->getLLVMVisitor()->writeToPythonExtension(*_compiler->getCache()->pyModule,
                                                                   filename);
                break;
            case BuildKind::Detect:
                _compiler->getLLVMVisitor()->compile(filename, argv0, libsVec, lflags);
                break;
            default:
                seqassertn(0, "unknown build kind");
        }

        return EXIT_SUCCESS;
    }


    int EngineVM::jit(const std::vector<const char *> &args) {
        llvm::cl::opt<std::string> input(llvm::cl::Positional, llvm::cl::desc("<input file>"),
                                         llvm::cl::init("-"));
        llvm::cl::list<std::string> plugins("plugin",
                                            llvm::cl::desc("Load specified plugin"));
        llvm::cl::opt<std::string> log("log", llvm::cl::desc("Enable given log streams"));
        llvm::cl::ParseCommandLineOptions(args.size(), args.data());
        init_log_flags(log);
        hercules::jit::JIT jit(args[0]);

        // load plugins
        for (const auto &plugin: plugins) {
            bool failed = false;
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
        if (input == "-") {
            jit_loop(&jit, std::cin);
        } else {
            std::ifstream fileInput(input);
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
