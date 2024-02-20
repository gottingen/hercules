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

#pragma once

#include <unordered_map>

#include "hercules/cir/llvm/llvm.h"
#include "hercules/cir/types/types.h"

namespace hercules::ir::analyze::dataflow {
    class CFVisitor;

}  // namespace hercules::ir::analyze::dataflow

class LLVMVisitor;

namespace hercules::ir::dsl::codegen {

    /// Builder for LLVM types.
    struct TypeBuilder {
        virtual ~TypeBuilder() noexcept = default;

        /// Construct the LLVM type.
        /// @param the LLVM visitor
        /// @return the LLVM type
        virtual llvm::Type *buildType(LLVMVisitor *visitor) = 0;

        /// Construct the LLVM debug type.
        /// @param the LLVM visitor
        /// @return the LLVM debug type
        virtual llvm::DIType *buildDebugType(LLVMVisitor *visitor) = 0;
    };

    /// Builder for LLVM values.
    struct ValueBuilder {
        virtual ~ValueBuilder() noexcept = default;

        /// Construct the LLVM value.
        /// @param the LLVM visitor
        /// @return the LLVM value
        virtual llvm::Value *buildValue(LLVMVisitor *visitor) = 0;
    };

    /// Builder for control flow graphs.
    struct CFBuilder {
        virtual ~CFBuilder() noexcept = default;

        /// Construct the control-flow nodes.
        /// @param graph the graph
        virtual void buildCFNodes(analyze::dataflow::CFVisitor *visitor) = 0;
    };

}  // namespace hercules::ir::dsl::codegen
