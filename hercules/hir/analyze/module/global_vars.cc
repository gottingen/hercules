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

#include <hercules/hir/analyze/module/global_vars.h>
#include <hercules/hir/util/operator.h>

namespace hercules::ir::analyze::module {
    namespace {
        struct GlobalVarAnalyzer : public util::Operator {
            std::unordered_map<id_t, id_t> assignments;

            void handle(PointerValue *v) override {
                if (v->getVar()->isGlobal())
                    assignments[v->getVar()->getId()] = -1;
            }

            void handle(AssignInstr *v) override {
                auto *lhs = v->getLhs();
                auto id = lhs->getId();
                if (lhs->isGlobal()) {
                    if (assignments.find(id) != assignments.end()) {
                        assignments[id] = -1;
                    } else {
                        assignments[id] = v->getRhs()->getId();
                    }
                }
            }
        };
    } // namespace

    const std::string GlobalVarsAnalyses::KEY = "core-analyses-global-vars";

    std::unique_ptr<Result> GlobalVarsAnalyses::run(const Module *m) {
        GlobalVarAnalyzer gva;
        gva.visit(const_cast<Module *>(m)); // TODO: any way around this cast?
        return std::make_unique<GlobalVarsResult>(std::move(gva.assignments));
    }

} // namespace hercules::ir::analyze::module
