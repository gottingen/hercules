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

#include <hercules/hir/module.h>
#include <hercules/hir/value.h>

namespace hercules::ir {

    /// HIR constant base. Once created, constants are immutable.
    class Const : public AcceptorExtend<Const, Value> {
    private:
        /// the type
        types::Type *type;

    public:
        static const char NodeId;

        /// Constructs a constant.
        /// @param type the type
        /// @param name the name
        explicit Const(types::Type *type, std::string name = "")
                : AcceptorExtend(std::move(name)), type(type) {}

    private:
        types::Type *doGetType() const override { return type; }

        std::vector<types::Type *> doGetUsedTypes() const override { return {type}; }

        int doReplaceUsedType(const std::string &name, types::Type *newType) override;
    };

    template<typename ValueType>
    class TemplatedConst : public AcceptorExtend<TemplatedConst<ValueType>, Const> {
    private:
        ValueType val;

    public:
        static const char NodeId;

        using AcceptorExtend<TemplatedConst<ValueType>, Const>::getModule;
        using AcceptorExtend<TemplatedConst<ValueType>, Const>::getSrcInfo;
        using AcceptorExtend<TemplatedConst<ValueType>, Const>::getType;

        TemplatedConst(ValueType v, types::Type *type, std::string name = "")
                : AcceptorExtend<TemplatedConst<ValueType>, Const>(type, std::move(name)),
                  val(v) {}

        /// @return the internal value.
        ValueType getVal() const { return val; }

        /// Sets the value.
        /// @param v the value
        void setVal(ValueType v) { val = v; }
    };

    using IntConst = TemplatedConst<int64_t>;
    using FloatConst = TemplatedConst<double>;
    using BoolConst = TemplatedConst<bool>;
    using StringConst = TemplatedConst<std::string>;

    template<typename T> const char TemplatedConst<T>::NodeId = 0;

    template<>
    class TemplatedConst<std::string>
            : public AcceptorExtend<TemplatedConst<std::string>, Const> {
    private:
        std::string val;

    public:
        static const char NodeId;

        TemplatedConst(std::string v, types::Type *type, std::string name = "")
                : AcceptorExtend(type, std::move(name)), val(std::move(v)) {}

        /// @return the internal value.
        std::string getVal() const { return val; }

        /// Sets the value.
        /// @param v the value
        void setVal(std::string v) { val = std::move(v); }
    };

} // namespace hercules::ir
