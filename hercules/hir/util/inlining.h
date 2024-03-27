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

#pragma once

#include "hercules/hir/cir.h"

namespace hercules::ir::util {

    /// Result of an inlining operation.
    struct InlineResult {
        /// the result, either a SeriesFlow or FlowInstr
        Value *result;
        /// variables added by the inlining
        std::vector<Var *> newVars;

        operator bool() const { return bool(result); }
    };

    /// Inline the given function with the supplied arguments.
    /// @param func the function
    /// @param args the arguments
    /// @param callInfo the call information
    /// @param aggressive true if should inline complex functions
    /// @return the inlined result, nullptr if unsuccessful
    InlineResult inlineFunction(Func *func, std::vector<Value *> args,
                                bool aggressive = false, hercules::SrcInfo callInfo = {});

    /// Inline the given call.
    /// @param v the instruction
    /// @param aggressive true if should inline complex functions
    /// @return the inlined result, nullptr if unsuccessful
    InlineResult inlineCall(CallInstr *v, bool aggressive = false);


} // namespace hercules::ir::util
