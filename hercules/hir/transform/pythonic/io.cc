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

#include "io.h"

#include <algorithm>

#include "hercules/hir/util/cloning.h"
#include "hercules/hir/util/irtools.h"

namespace hercules::ir::transform::pythonic {
    namespace {
        void optimizePrint(CallInstr *v) {
            auto *M = v->getModule();

            auto *inner = cast<CallInstr>(v->front());
            if (!inner)
                return;
            auto *innerFunc = util::getFunc(inner->getCallee());
            if (!innerFunc || innerFunc->getUnmangledName() != "__new__" ||
                std::distance(inner->begin(), inner->end()) != 1)
                return;

            auto *cat = cast<CallInstr>(inner->front());
            if (!cat)
                return;
            auto *catFunc = util::getFunc(cat->getCallee());
            if (!catFunc || catFunc->getUnmangledName() != "cat")
                return;

            auto *realCat =
                    M->getOrRealizeMethod(M->getStringType(), "cat", {cat->front()->getType()});
            if (realCat->getId() != catFunc->getId())
                return;

            util::CloneVisitor cv(M);
            std::vector<Value *> args;
            std::vector<types::Type *> types;
            for (auto *printArg: *v) {
                args.push_back(cv.clone(printArg));
                types.push_back(printArg->getType());
            }
            args[0] = cv.clone(cat->front());
            types[0] = args[0]->getType();
            args[1] = M->getString("");

            auto *replacement = M->getOrRealizeFunc("print", types, {}, "std.internal.builtin");
            if (!replacement)
                return;

            v->replaceAll(util::call(replacement, args));
        }

        void optimizeWrite(CallInstr *v) {
            auto *M = v->getModule();

            auto it = v->begin();
            auto *file = *it++;

            auto *cat = cast<CallInstr>(*it++);
            if (!cat)
                return;
            auto *catFunc = util::getFunc(cat->getCallee());
            if (!catFunc || catFunc->getUnmangledName() != "cat")
                return;

            auto *realCat =
                    M->getOrRealizeMethod(M->getStringType(), "cat", {cat->front()->getType()});
            if (realCat->getId() != catFunc->getId())
                return;

            util::CloneVisitor cv(M);
            auto *iter = cv.clone(cat->front())->iter();
            if (!iter)
                return;

            std::vector<Value *> args = {cv.clone(file), iter};

            auto *replacement = M->getOrRealizeMethod(file->getType(), "__file_write_gen__",
                                                      {args[0]->getType(), args[1]->getType()});
            if (!replacement)
                return;

            v->replaceAll(util::call(replacement, args));
        }
    } // namespace

    const std::string IOCatOptimization::KEY = "core-pythonic-io-cat-opt";

    void IOCatOptimization::handle(CallInstr *v) {
        if (util::getStdlibFunc(v->getCallee(), "print")) {
            optimizePrint(v);
        } else if (auto *f = cast<Func>(util::getFunc(v->getCallee()))) {
            if (f->getUnmangledName() == "write")
                optimizeWrite(v);
        }
    }

} // namespace hercules::ir::transform::pythonic
