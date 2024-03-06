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
#include <string>
#include <utility>
#include <vector>

#include "hercules/hir/types/types.h"
#include "hercules/hir/value.h"
#include "hercules/util/common.h"
#include <fmt/format.h>
#include <fmt/ostream.h>

namespace hercules::ir {

    class Func;

    class Var;

    /// HIR object representing a variable.
    class Var : public ReplaceableNodeBase<Var>, public IdMixin {
    private:
        /// the variable's type
        types::Type *type;
        /// true if the variable is global
        bool global;
        /// true if the variable is external
        bool external;

    public:
        static const char NodeId;

        /// Constructs a variable.
        /// @param type the variable's type
        /// @param global true if the variable is global
        /// @param external true if the variable is external
        /// @param name the variable's name
        explicit Var(types::Type *type, bool global = false, bool external = false,
                     std::string name = "")
                : ReplaceableNodeBase(std::move(name)), type(type), global(global),
                  external(external) {}

        virtual ~Var() noexcept = default;

        std::vector<Value *> getUsedValues() final { return getActual()->doGetUsedValues(); }

        std::vector<const Value *> getUsedValues() const final {
            auto ret = getActual()->doGetUsedValues();
            return std::vector<const Value *>(ret.begin(), ret.end());
        }

        int replaceUsedValue(id_t id, Value *newValue) final {
            return doReplaceUsedValue(id, newValue);
        }

        using Node::replaceUsedValue;

        std::vector<types::Type *> getUsedTypes() const final {
            return getActual()->doGetUsedTypes();
        }

        int replaceUsedType(const std::string &name, types::Type *newType) final {
            return getActual()->doReplaceUsedType(name, newType);
        }

        using Node::replaceUsedType;

        std::vector<Var *> getUsedVariables() final { return doGetUsedVariables(); }

        std::vector<const Var *> getUsedVariables() const final {
            auto ret = doGetUsedVariables();
            return std::vector<const Var *>(ret.begin(), ret.end());
        }

        int replaceUsedVariable(id_t id, Var *newVar) final {
            return getActual()->doReplaceUsedVariable(id, newVar);
        }

        using Node::replaceUsedVariable;

        /// @return the type
        types::Type *getType() const { return getActual()->type; }

        /// Sets the type.
        /// @param t the new type
        void setType(types::Type *t) { getActual()->type = t; }

        /// @return true if the variable is global
        bool isGlobal() const { return getActual()->global; }

        /// Sets the global flag.
        /// @param v the new value
        void setGlobal(bool v = true) { getActual()->global = v; }

        /// @return true if the variable is external
        bool isExternal() const { return getActual()->external; }

        /// Sets the external flag.
        /// @param v the new value
        void setExternal(bool v = true) { getActual()->external = v; }

        std::string referenceString() const final {
            return fmt::format(FMT_STRING("{}.{}"), getName(), getId());
        }

        id_t getId() const override { return getActual()->id; }

    protected:
        virtual std::vector<Value *> doGetUsedValues() const { return {}; }

        virtual int doReplaceUsedValue(id_t id, Value *newValue) { return 0; }

        virtual std::vector<types::Type *> doGetUsedTypes() const { return {type}; }

        virtual int doReplaceUsedType(const std::string &name, types::Type *newType);

        virtual std::vector<Var *> doGetUsedVariables() const { return {}; }

        virtual int doReplaceUsedVariable(id_t id, Var *newVar) { return 0; }
    };

    /// Value that contains an unowned variable reference.
    class VarValue : public AcceptorExtend<VarValue, Value> {
    private:
        /// the referenced var
        Var *val;

    public:
        static const char NodeId;

        /// Constructs a variable value.
        /// @param val the referenced value
        /// @param name the name
        explicit VarValue(Var *val, std::string name = "")
                : AcceptorExtend(std::move(name)), val(val) {}

        /// @return the variable
        Var *getVar() { return val; }

        /// @return the variable
        const Var *getVar() const { return val; }

        /// Sets the variable.
        /// @param v the new variable
        void setVar(Var *v) { val = v; }

    private:
        types::Type *doGetType() const override { return val->getType(); }

        std::vector<Var *> doGetUsedVariables() const override { return {val}; }

        int doReplaceUsedVariable(id_t id, Var *newVar) override;
    };

    /// Value that represents a pointer.
    class PointerValue : public AcceptorExtend<PointerValue, Value> {
    private:
        /// the referenced var
        Var *val;

    public:
        static const char NodeId;

        /// Constructs a variable value.
        /// @param val the referenced value
        /// @param name the name
        explicit PointerValue(Var *val, std::string name = "")
                : AcceptorExtend(std::move(name)), val(val) {}

        /// @return the variable
        Var *getVar() { return val; }

        /// @return the variable
        const Var *getVar() const { return val; }

        /// Sets the variable.
        /// @param v the new variable
        void setVar(Var *v) { val = v; }

    private:
        types::Type *doGetType() const override;

        std::vector<Var *> doGetUsedVariables() const override { return {val}; }

        int doReplaceUsedVariable(id_t id, Var *newVar) override;
    };

} // namespace hercules::ir

template<typename T>
struct fmt::formatter<T,
        std::enable_if_t<std::is_base_of<hercules::ir::Var, T>::value, char>>
        : fmt::ostream_formatter {
};
