// Copyright 2023 The titan-search Authors.
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

#include "str.h"

#include <algorithm>

#include "hercules/cir/util/cloning.h"
#include "hercules/cir/util/irtools.h"

namespace hercules::ir::transform::pythonic {
    namespace {
        struct InspectionResult {
            bool valid = true;
            std::vector<Value *> args;
        };

        bool isString(Value *v) {
            auto *M = v->getModule();
            return v->getType()->is(M->getStringType());
        }

        void inspect(Value *v, InspectionResult &r) {
            // check if add first then go from there
            if (isString(v)) {
                if (auto *c = cast<CallInstr>(v)) {
                    auto *func = util::getFunc(c->getCallee());
                    if (func && func->getUnmangledName() == Module::ADD_MAGIC_NAME &&
                        c->numArgs() == 2 && isString(c->front()) && isString(c->back())) {
                        inspect(c->front(), r);
                        inspect(c->back(), r);
                        return;
                    }
                }
                r.args.push_back(v);
            } else {
                r.valid = false;
            }
        }
    } // namespace

    const std::string StrAdditionOptimization::KEY = "core-pythonic-str-addition-opt";

    void StrAdditionOptimization::handle(CallInstr *v) {
        auto *M = v->getModule();

        auto *f = util::getFunc(v->getCallee());
        if (!f || f->getUnmangledName() != Module::ADD_MAGIC_NAME)
            return;

        InspectionResult r;
        inspect(v, r);

        if (r.valid && r.args.size() > 2) {
            std::vector<Value *> args;
            util::CloneVisitor cv(M);

            for (auto *arg: r.args) {
                args.push_back(cv.clone(arg));
            }

            auto *arg = util::makeTuple(args, M);
            args = {arg};
            auto *replacementFunc =
                    M->getOrRealizeMethod(M->getStringType(), "cat", {arg->getType()});
            seqassertn(replacementFunc, "could not find cat function [{}]", v->getSrcInfo());
            v->replaceAll(util::call(replacementFunc, args));
        }
    }

} // namespace hercules::ir::transform::pythonic
