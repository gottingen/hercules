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

#include <memory>
#include <stdexcept>
#include <string>

#define VISIT(x) virtual void visit(hercules::ir::x *)
#define CONST_VISIT(x) virtual void visit(const hercules::ir::x *)

namespace hercules::ir {
    class Node;
}  // namespace hercules::ir

namespace hercules::ir::types {
    class Type;

    class PrimitiveType;

    class IntType;

    class FloatType;

    class Float32Type;

    class Float16Type;

    class BFloat16Type;

    class Float128Type;

    class BoolType;

    class ByteType;

    class VoidType;

    class RecordType;

    class RefType;

    class FuncType;

    class OptionalType;

    class PointerType;

    class GeneratorType;

    class IntNType;

    class VectorType;

    class UnionType;
} // namespace hercules::ir::types

namespace hercules::ir::dsl {

    namespace types {
        class CustomType;
    }

    class CustomConst;

    class CustomFlow;

    class CustomInstr;
} // namespace hercules::ir::dsl

namespace hercules::ir {
    class Module;

    class Var;

    class Func;

    class BodiedFunc;

    class ExternalFunc;

    class InternalFunc;

    class LLVMFunc;

    class Value;

    class VarValue;

    class PointerValue;

    class Flow;

    class SeriesFlow;

    class IfFlow;

    class WhileFlow;

    class ForFlow;

    class ImperativeForFlow;

    class TryCatchFlow;

    class PipelineFlow;

    class Const;

    template<typename ValueType>
    class TemplatedConst;

    class Instr;

    class AssignInstr;

    class ExtractInstr;

    class InsertInstr;

    class CallInstr;

    class StackAllocInstr;

    class TypePropertyInstr;

    class YieldInInstr;

    class TernaryInstr;

    class BreakInstr;

    class ContinueInstr;

    class ReturnInstr;

    class YieldInstr;

    class ThrowInstr;

    class FlowInstr;
}  // namespace hercules::ir

namespace hercules::ir::util {

    /// Base for HIR visitors
    class Visitor {
    protected:
        virtual void defaultVisit(hercules::ir::Node *) {
            throw std::runtime_error("cannot visit node");
        }

    public:
        virtual ~Visitor() noexcept = default;

        VISIT(Module);

        VISIT(Var);

        VISIT(Func);

        VISIT(BodiedFunc);

        VISIT(ExternalFunc);

        VISIT(InternalFunc);

        VISIT(LLVMFunc);

        VISIT(Value);

        VISIT(VarValue);

        VISIT(PointerValue);

        VISIT(Flow);

        VISIT(SeriesFlow);

        VISIT(IfFlow);

        VISIT(WhileFlow);

        VISIT(ForFlow);

        VISIT(ImperativeForFlow);

        VISIT(TryCatchFlow);

        VISIT(PipelineFlow);

        VISIT(dsl::CustomFlow);

        VISIT(Const);

        VISIT(TemplatedConst<int64_t>);

        VISIT(TemplatedConst<double>);

        VISIT(TemplatedConst<bool>);

        VISIT(TemplatedConst<std::string>);

        VISIT(dsl::CustomConst);

        VISIT(Instr);

        VISIT(AssignInstr);

        VISIT(ExtractInstr);

        VISIT(InsertInstr);

        VISIT(CallInstr);

        VISIT(StackAllocInstr);

        VISIT(TypePropertyInstr);

        VISIT(YieldInInstr);

        VISIT(TernaryInstr);

        VISIT(BreakInstr);

        VISIT(ContinueInstr);

        VISIT(ReturnInstr);

        VISIT(YieldInstr);

        VISIT(ThrowInstr);

        VISIT(FlowInstr);

        VISIT(dsl::CustomInstr);

        VISIT(types::Type);

        VISIT(types::PrimitiveType);

        VISIT(types::IntType);

        VISIT(types::FloatType);

        VISIT(types::Float32Type);

        VISIT(types::Float16Type);

        VISIT(types::BFloat16Type);

        VISIT(types::Float128Type);

        VISIT(types::BoolType);

        VISIT(types::ByteType);

        VISIT(types::VoidType);

        VISIT(types::RecordType);

        VISIT(types::RefType);

        VISIT(types::FuncType);

        VISIT(types::OptionalType);

        VISIT(types::PointerType);

        VISIT(types::GeneratorType);

        VISIT(types::IntNType);

        VISIT(types::VectorType);

        VISIT(types::UnionType);

        VISIT(dsl::types::CustomType);
    };

    class ConstVisitor {
    protected:
        virtual void defaultVisit(const hercules::ir::Node *) {
            throw std::runtime_error("cannot visit const node");
        }

    public:
        virtual ~ConstVisitor() noexcept = default;

        CONST_VISIT(Module);

        CONST_VISIT(Var);

        CONST_VISIT(Func);

        CONST_VISIT(BodiedFunc);

        CONST_VISIT(ExternalFunc);

        CONST_VISIT(InternalFunc);

        CONST_VISIT(LLVMFunc);

        CONST_VISIT(Value);

        CONST_VISIT(VarValue);

        CONST_VISIT(PointerValue);

        CONST_VISIT(Flow);

        CONST_VISIT(SeriesFlow);

        CONST_VISIT(IfFlow);

        CONST_VISIT(WhileFlow);

        CONST_VISIT(ForFlow);

        CONST_VISIT(ImperativeForFlow);

        CONST_VISIT(TryCatchFlow);

        CONST_VISIT(PipelineFlow);

        CONST_VISIT(dsl::CustomFlow);

        CONST_VISIT(Const);

        CONST_VISIT(TemplatedConst<int64_t>);

        CONST_VISIT(TemplatedConst<double>);

        CONST_VISIT(TemplatedConst<bool>);

        CONST_VISIT(TemplatedConst<std::string>);

        CONST_VISIT(dsl::CustomConst);

        CONST_VISIT(Instr);

        CONST_VISIT(AssignInstr);

        CONST_VISIT(ExtractInstr);

        CONST_VISIT(InsertInstr);

        CONST_VISIT(CallInstr);

        CONST_VISIT(StackAllocInstr);

        CONST_VISIT(TypePropertyInstr);

        CONST_VISIT(YieldInInstr);

        CONST_VISIT(TernaryInstr);

        CONST_VISIT(BreakInstr);

        CONST_VISIT(ContinueInstr);

        CONST_VISIT(ReturnInstr);

        CONST_VISIT(YieldInstr);

        CONST_VISIT(ThrowInstr);

        CONST_VISIT(FlowInstr);

        CONST_VISIT(dsl::CustomInstr);

        CONST_VISIT(types::Type);

        CONST_VISIT(types::PrimitiveType);

        CONST_VISIT(types::IntType);

        CONST_VISIT(types::FloatType);

        CONST_VISIT(types::Float32Type);

        CONST_VISIT(types::Float16Type);

        CONST_VISIT(types::BFloat16Type);

        CONST_VISIT(types::Float128Type);

        CONST_VISIT(types::BoolType);

        CONST_VISIT(types::ByteType);

        CONST_VISIT(types::VoidType);

        CONST_VISIT(types::RecordType);

        CONST_VISIT(types::RefType);

        CONST_VISIT(types::FuncType);

        CONST_VISIT(types::OptionalType);

        CONST_VISIT(types::PointerType);

        CONST_VISIT(types::GeneratorType);

        CONST_VISIT(types::IntNType);

        CONST_VISIT(types::VectorType);

        CONST_VISIT(types::UnionType);

        CONST_VISIT(dsl::types::CustomType);
    };

}  // namespace hercules::ir::util

#undef VISIT
#undef CONST_VISIT
