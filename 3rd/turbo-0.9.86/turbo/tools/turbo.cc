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

#include "turbo/flags/flags.h"
#include "turbo/tools/context.h"
#include "turbo/tools/hash.h"
#include "turbo/version.h"
#include "turbo/format/table.h"
#include "turbo/format/print.h"

void run_version_cmd();

int main(int argc, char **argv) {
    turbo::App app("turbo", "Turbo is command line tool for convenient usage");
    app.add_flag("-v, --version", turbo::tools::Context::get_instance().version, "0.0.1");
    turbo::tools::set_up_hash_cmdline(app);
    app.callback([&]() {
        if (turbo::tools::Context::get_instance().version) {
            run_version_cmd();
            return;
        }
        if (app.get_subcommands().empty()) {
            turbo::Println("{}", app.help());
            return;
        }
    });

    TURBO_FLAGS_PARSE(app, argc, argv);
}

void run_version_cmd() {
    turbo::Table versions;
    versions.add_row({"Turbo", turbo::version.to_string()});
    versions.add_row({"Turbo-Base", turbo::base_version.to_string()});
    versions.add_row({"Turbo-Hash", turbo::hash_version.to_string()});
    versions.add_row({"Turbo-concurrent", turbo::concurrent_version.to_string()});
    versions.add_row({"Turbo-container", turbo::container_version.to_string()});
    versions.add_row({"Turbo-flags", turbo::flags_version.to_string()});
    versions.add_row({"Turbo-format", turbo::format_version.to_string()});
    versions.add_row({"Turbo-hash", turbo::hash_version.to_string()});
    versions.add_row({"Turbo-log", turbo::log_version.to_string()});
    versions.add_row({"Turbo-memory", turbo::memory_version.to_string()});
    versions.add_row({"Turbo-meta", turbo::meta_version.to_string()});
    versions.add_row({"Turbo-module", turbo::module_version.to_string()});
    versions.add_row({"Turbo-platform", turbo::platform_version.to_string()});
    versions.add_row({"Turbo-random", turbo::random_version.to_string()});
    versions.add_row({"Turbo-strings", turbo::strings_version.to_string()});
    versions.add_row({"Turbo-times", turbo::times_version.to_string()});
    versions.column(0).format().font_color(turbo::terminal_color::yellow);
    versions.column(1).format().font_color(turbo::terminal_color::green);
    std::cout<<versions<<std::endl;
}