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

    /// Base class for IR nodes that match anything.
    class Any {
    };

    /// Any value.
    class AnyValue : public AcceptorExtend<AnyValue, Value>, public Any {
    public:
        static const char NodeId;
        using AcceptorExtend::AcceptorExtend;

    private:
        types::Type *doGetType() const override { return getModule()->getVoidType(); }
    };

    /// Any flow.
    class AnyFlow : public AcceptorExtend<AnyFlow, Flow>, public Any {
    public:
        static const char NodeId;
        using AcceptorExtend::AcceptorExtend;
    };

    /// Any variable.
    class AnyVar : public AcceptorExtend<AnyVar, Var>, public Any {
    public:
        static const char NodeId;
        using AcceptorExtend::AcceptorExtend;
    };

    /// Any function.
    class AnyFunc : public AcceptorExtend<AnyFunc, Func>, public Any {
    public:
        static const char NodeId;
        using AcceptorExtend::AcceptorExtend;

        AnyFunc() : AcceptorExtend() { setUnmangledName("any"); }
    };

    /// Checks if IR nodes match.
    /// @param a the first IR node
    /// @param b the second IR node
    /// @param checkNames whether or not to check the node names
    /// @param varIdMatch whether or not variable ids must match
    /// @return true if the nodes are equal
    bool match(Node *a, Node *b, bool checkNames = false, bool varIdMatch = false);

} // namespace hercules::ir::util
