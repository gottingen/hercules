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
#include <hercules/builtin/debug/trace.h>


namespace hercules::builtin {

    void BuiltinTrace::handle(hercules::ir::AssignInstr *v) {
        auto *M = v->getModule();
        auto *call = hercules::ir::cast<hercules::ir::CallInstr>(v->getRhs());
        if (!call)
            return;

        auto *foo = hercules::ir::util::getFunc(call->getCallee());
        if (!foo)
            return;
        if (!hercules::ir::util::hasAttribute(foo, "std.internal.attributes.trace_debug")) {
            return;
        }
        auto f = collie::filesystem::path(foo->getSrcInfo().file).filename().string();
        auto l = foo->getSrcInfo().line;
        auto cf = collie::filesystem::path(v->getSrcInfo().file).filename().string();
        auto cl = v->getSrcInfo().line;
        auto debug_str = collie::format(FMT_STRING("{} called [{}:{}] at [{}:{}]"), foo->getUnmangledName(), f, l, cf,
                                        cl);
        auto *vstr = M->getString(debug_str);
        auto *builtin_pass_debug =
                M->getOrRealizeFunc("builtin_trace_debug", {vstr->getType()}, {}, "std.internal.builtin");
        assert(builtin_pass_debug && "builtin_pass_debug not found");
        auto *debug_call = hercules::ir::util::call(builtin_pass_debug, {vstr});

        insertAfter(debug_call);  // call 'validate' after 'foo'
    }

    const std::string BuiltinTrace::KEY = "builtin-trace";
}  // namespace hercules::builtin

