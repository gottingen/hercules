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
//
// Created by jeff on 24-4-2.
//
#include <hercules/app/jupyter/jupyter.h>
#include <hercules/engine/vm.h>
#include <hercules/util/jupyter.h>

namespace hercules {

    void run_jupyter_mode() {
        auto &ins = hercules::VmContext::instance();
        ins.ret_code = hercules::startJupyterKernel(ins.argv0, ins.plugins, ins.connection_file);
    }

    void set_up_jupyter_command(collie::App* app) {
        auto &ins = hercules::VmContext::instance();
        app->add_option("-p, --plugin", ins.plugins, "Load specified plugin");
        app->add_option("connection", hercules::VmContext::instance().connection_file, "connection file")->default_str("connection.json");
        app->callback([](){
            hercules::VmContext::instance().mode = "jupyter";
            hercules::VmContext::instance().argv0 = hercules::VmContext::instance().orig_argv0 + " jupyter";
            run_jupyter_mode();
        });
    }

} // namespace hercules