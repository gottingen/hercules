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
#include "subcommand_a.h"
#include "turbo/flags/flags.h"

int main(int argc, char **argv) {
    turbo::App app{"..."};

    // Call the setup functions for the subcommands.
    // They are kept alive by a shared pointer in the
    // lambda function
    setup_subcommand_a(app);

    // Make sure we get at least one subcommand
    app.require_subcommand();

    // More setup if needed, i.e., other subcommands etc.

    TURBO_FLAGS_PARSE(app, argc, argv);

    return 0;
}
