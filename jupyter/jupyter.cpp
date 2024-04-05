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

#include "jupyter.h"

#include <dirent.h>
#include <fcntl.h>
#include <iostream>
#include <locale>
#include <nlohmann/json.hpp>
#include <unistd.h>
#include <xeus-zmq/xserver_zmq.hpp>
#include <xeus/xhelper.hpp>
#include <xeus/xkernel.hpp>
#include <xeus/xkernel_configuration.hpp>

#include <hercules/compiler/compiler.h>
#include "hercules/compiler/error.h"
#include <hercules/compiler/jit.h>
#include "hercules/config/config.h"
#include <hercules/parser/common.h>
#include <hercules/util/common.h>

using std::move;
using std::string;

namespace nl = nlohmann;
namespace hercules {

HerculesJupyter::HerculesJupyter(const std::string &argv0,
                           const std::vector<std::string> &plugins)
    : argv0(argv0), plugins(plugins) {}

nl::json HerculesJupyter::execute_request_impl(int execution_counter, const string &code,
                                            bool silent, bool store_history,
                                            nl::json user_expressions,
                                            bool allow_stdin) {
  LOG("[hercules-jupyter] execute_request_impl");
  auto result = jit->execute(code);
  string failed;
  llvm::handleAllErrors(
      result.takeError(),
      [&](const hercules::error::ParserErrorInfo &e) {
        std::vector<string> backtrace;
        for (auto &msg : e)
          for (auto &s : msg)
            backtrace.push_back(s.getMessage());
        string err = backtrace[0];
        backtrace.erase(backtrace.begin());
        failed = fmt::format("Compile error: {}\nBacktrace:\n{}", err,
                             ast::join(backtrace, "  \n"));
      },
      [&](const hercules::error::RuntimeErrorInfo &e) {
        auto backtrace = e.getBacktrace();
        failed = fmt::format("Runtime error: {}\nBacktrace:\n{}", e.getMessage(),
                             ast::join(backtrace, "  \n"));
      });
  if (failed.empty()) {
    std::string out = *result;
    nl::json pub_data;
    using std::string_literals::operator""s;
    std::string herculesMimeMagic = "\x00\x00__hercules/mime__\x00"s;
    if (ast::startswith(out, herculesMimeMagic)) {
      std::string mime = "";
      int i = herculesMimeMagic.size();
      for (; i < out.size() && out[i]; i++)
        mime += out[i];
      if (i < out.size() && !out[i]) {
        i += 1;
      } else {
        mime = "text/plain";
        i = 0;
      }
      pub_data[mime] = out.substr(i);
      LOG("> {}: {}", mime, out.substr(i));
    } else {
      pub_data["text/plain"] = out;
    }
    if (!out.empty())
      publish_execution_result(execution_counter, move(pub_data), nl::json::object());
    return nl::json{{"status", "ok"},
                    {"payload", nl::json::array()},
                    {"user_expressions", nl::json::object()}};
  } else {
    publish_stream("stderr", failed);
    return nl::json{{"status", "error"}};
  }
}

void HerculesJupyter::configure_impl() {
  jit = std::make_unique<hercules::jit::JIT>(argv0, "jupyter");
  jit->getCompiler()->getLLVMVisitor()->setCapture();

  for (const auto &plugin : plugins) {
    // TODO: error handling on plugin init
    bool failed = false;
    llvm::handleAllErrors(jit->getCompiler()->load(plugin),
                          [&failed](const hercules::error::PluginErrorInfo &e) {
                            hercules::compilationError(e.getMessage(), /*file=*/"",
                                                    /*line=*/0, /*col=*/0,
                                                    /*terminate=*/false);
                            failed = true;
                          });
  }
  llvm::cantFail(jit->init());
}

nl::json HerculesJupyter::complete_request_impl(const string &code, int cursor_pos) {
  LOG("[hercules-jupyter] complete_request_impl");
  return nl::json{{"status", "ok"}};
}

nl::json HerculesJupyter::inspect_request_impl(const string &code, int cursor_pos,
                                            int detail_level) {
  LOG("[hercules-jupyter] inspect_request_impl");
  return nl::json{{"status", "ok"}};
}

nl::json HerculesJupyter::is_complete_request_impl(const string &code) {
  LOG("[hercules-jupyter] is_complete_request_impl");
  return nl::json{{"status", "complete"}};
}

nl::json HerculesJupyter::kernel_info_request_impl() {
  LOG("[hercules-jupyter] kernel_info_request_impl");
  return xeus::create_info_reply("", "hercules_kernel", HERCULES_VERSION, "python", "3.7",
                                 "text/x-python", ".hs", "python", "", "",
                                 "Hercules Kernel");
}

void HerculesJupyter::shutdown_request_impl() {
  LOG("[hercules-jupyter] shutdown_request_impl");
}

int startJupyterKernel(const std::string &argv0,
                       const std::vector<std::string> &plugins,
                       const std::string &configPath) {
  xeus::xconfiguration config = xeus::load_configuration(configPath);

  auto context = xeus::make_context<zmq::context_t>();

  LOG("[hercules-jupyter] startJupyterKernel");
  auto interpreter = std::make_unique<HerculesJupyter>(argv0, plugins);
  xeus::xkernel kernel(config, xeus::get_user_name(), move(context), move(interpreter),
                       xeus::make_xserver_zmq);
  kernel.start();

  return 0;
}

} // namespace hercules
