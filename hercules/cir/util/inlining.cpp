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

#include "inlining.h"

#include <algorithm>

#include "hercules/cir/util/cloning.h"
#include "hercules/cir/util/irtools.h"
#include "hercules/cir/util/operator.h"

namespace hercules {
namespace ir {
namespace util {

namespace {

class ReturnVerifier : public util::Operator {
public:
  bool needLoop = false;

  void handle(ReturnInstr *v) {
    if (needLoop) {
      return;
    }

    auto it = parent_begin();
    if (it == parent_end()) {
      needLoop = true;
      return;
    }

    SeriesFlow *prev = nullptr;
    while (it != parent_end()) {
      Value *v = cast<Value>(*it++);
      auto *cur = cast<SeriesFlow>(v);
      if (!cur || (prev && prev->back()->getId() != cur->getId())) {
        needLoop = true;
        return;
      }
      prev = cur;
    }
    needLoop = prev->back()->getId() != v->getId();
  }
};

class ReturnReplacer : public util::Operator {
private:
  Value *implicitLoop;
  Var *var;
  bool aggressive;
  util::CloneVisitor &cv;

public:
  ReturnReplacer(Value *implicitLoop, Var *var, bool aggressive, util::CloneVisitor &cv)
      : implicitLoop(implicitLoop), var(var), aggressive(aggressive), cv(cv) {}

  void handle(ReturnInstr *v) {
    auto *M = v->getModule();
    auto *rep = M->N<SeriesFlow>(v);
    if (var) {
      rep->push_back(M->N<AssignInstr>(v, var, cv.clone(v->getValue())));
    }
    if (aggressive)
      rep->push_back(M->N<BreakInstr>(v, implicitLoop));

    v->replaceAll(rep);
  }
};

} // namespace

InlineResult inlineFunction(Func *func, std::vector<Value *> args, bool aggressive,
                            hercules::SrcInfo info) {
  auto *bodied = cast<BodiedFunc>(func);
  if (!bodied)
    return {nullptr, {}};
  auto *fType = cast<types::FuncType>(bodied->getType());
  if (!fType || args.size() != std::distance(bodied->arg_begin(), bodied->arg_end()))
    return {nullptr, {}};
  auto *M = bodied->getModule();

  util::CloneVisitor cv(M);
  auto *newFlow = M->N<SeriesFlow>(info, bodied->getName() + "_inlined");

  std::vector<Var *> newVars;
  auto arg_it = bodied->arg_begin();
  for (auto i = 0; i < args.size(); ++i) {
    newVars.push_back(cv.forceClone(*arg_it++));
    newFlow->push_back(M->N<AssignInstr>(info, newVars.back(), cv.clone(args[i])));
  }
  for (auto *v : *bodied) {
    newVars.push_back(cv.forceClone(v));
  }
  Var *retVal = nullptr;
  if (!fType->getReturnType()->is(M->getVoidType()) &&
      !fType->getReturnType()->is(M->getNoneType())) {
    retVal = M->N<Var>(info, fType->getReturnType());
    newVars.push_back(retVal);
  }

  Flow *clonedBody = cv.clone(bodied->getBody());

  ReturnVerifier rv;
  rv.process(clonedBody);

  if (!aggressive && rv.needLoop)
    return {nullptr, {}};

  WhileFlow *implicit = nullptr;
  if (rv.needLoop) {
    auto *loopBody = M->N<SeriesFlow>(info);
    implicit = M->N<WhileFlow>(info, M->getBool(true), loopBody);
    loopBody->push_back(clonedBody);
    if (!retVal)
      loopBody->push_back(M->N<BreakInstr>(info, implicit));
  }

  ReturnReplacer rr(implicit, retVal, rv.needLoop, cv);
  rr.process(clonedBody);

  newFlow->push_back(implicit ? implicit : clonedBody);

  if (retVal) {
    return {M->N<FlowInstr>(info, newFlow, M->N<VarValue>(info, retVal)),
            std::move(newVars)};
  }
  return {newFlow, std::move(newVars)};
}

InlineResult inlineCall(CallInstr *v, bool aggressive) {
  return inlineFunction(util::getFunc(v->getCallee()),
                        std::vector<Value *>(v->begin(), v->end()), aggressive,
                        v->getSrcInfo());
}

} // namespace util
} // namespace ir
} // namespace hercules
