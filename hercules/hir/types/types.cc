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

#include <hercules/hir/types/types.h>

#include <algorithm>
#include <memory>
#include <utility>

#include <hercules/hir/module.h>
#include <hercules/hir/util/irtools.h>
#include <hercules/hir/util/iterators.h>
#include <hercules/hir/util/visitor.h>
#include <hercules/hir/value.h>
#include <hercules/parser/cache.h>
#include <collie/strings/format.h>

namespace hercules::ir::types {
    namespace {
        std::vector<hercules::ast::types::TypePtr>
        extractTypes(const std::vector<hercules::ast::types::ClassType::Generic> &gens) {
            std::vector<hercules::ast::types::TypePtr> ret;
            for (auto &g: gens)
                ret.push_back(g.type);
            return ret;
        }
    } // namespace

    const char Type::NodeId = 0;

    std::vector<Generic> Type::doGetGenerics() const {
        if (!astType)
            return {};

        std::vector<Generic> ret;
        for (auto &g: astType->getClass()->generics) {
            if (auto cls = g.type->getClass())
                ret.emplace_back(
                        getModule()->getCache()->realizeType(cls, extractTypes(cls->generics)));
            else {
                seqassertn(g.type->getStatic()->expr->staticValue.type == ast::StaticValue::INT,
                           "IR only supports int statics [{}]", g.type->getSrcInfo());
                ret.emplace_back(g.type->getStatic()->expr->staticValue.getInt());
            }
        }

        return ret;
    }

    Value *Type::doConstruct(std::vector<Value *> args) {
        auto *module = getModule();
        std::vector<Type *> argTypes;
        for (auto *a: args)
            argTypes.push_back(a->getType());

        auto *fn = module->getOrRealizeMethod(this, Module::NEW_MAGIC_NAME, argTypes);
        if (!fn)
            return nullptr;

        return module->Nr<CallInstr>(module->Nr<VarValue>(fn), args);
    }

    const char PrimitiveType::NodeId = 0;

    const char IntType::NodeId = 0;

    const char FloatType::NodeId = 0;

    const char Float32Type::NodeId = 0;

    const char Float16Type::NodeId = 0;

    const char BFloat16Type::NodeId = 0;

    const char Float128Type::NodeId = 0;

    const char BoolType::NodeId = 0;

    const char ByteType::NodeId = 0;

    const char VoidType::NodeId = 0;

    const char MemberedType::NodeId = 0;

    const char RecordType::NodeId = 0;

    RecordType::RecordType(std::string name, std::vector<Type *> fieldTypes,
                           std::vector<std::string> fieldNames)
            : AcceptorExtend(std::move(name)) {
        for (auto i = 0; i < fieldTypes.size(); ++i) {
            fields.emplace_back(fieldNames[i], fieldTypes[i]);
        }
    }

    RecordType::RecordType(std::string name, std::vector<Type *> mTypes)
            : AcceptorExtend(std::move(name)) {
        for (int i = 0; i < mTypes.size(); ++i) {
            fields.emplace_back(std::to_string(i + 1), mTypes[i]);
        }
    }

    std::vector<Type *> RecordType::doGetUsedTypes() const {
        std::vector<Type *> ret;
        for (auto &f: fields)
            ret.push_back(const_cast<Type *>(f.getType()));
        return ret;
    }

    Type *RecordType::getMemberType(const std::string &n) const {
        auto it = std::find_if(fields.begin(), fields.end(),
                               [n](auto &x) { return x.getName() == n; });
        return it->getType();
    }

    int RecordType::getMemberIndex(const std::string &n) const {
        auto it = std::find_if(fields.begin(), fields.end(),
                               [n](auto &x) { return x.getName() == n; });
        int index = std::distance(fields.begin(), it);
        return (index < fields.size()) ? index : -1;
    }

    void RecordType::realize(std::vector<Type *> mTypes, std::vector<std::string> mNames) {
        fields.clear();
        for (auto i = 0; i < mTypes.size(); ++i) {
            fields.emplace_back(mNames[i], mTypes[i]);
        }
    }

    const char RefType::NodeId = 0;

    bool RefType::doIsContentAtomic() const {
        auto *contents = getContents();
        return !std::any_of(contents->begin(), contents->end(), [](auto &field) {
            return field.getName().rfind(".__vtable__", 0) != 0 && !field.getType()->isAtomic();
        });
    }

    Value *RefType::doConstruct(std::vector<Value *> args) {
        auto *module = getModule();
        auto *argsTuple = util::makeTuple(args, module);
        auto *constructFn = module->getOrRealizeFunc("construct_ref", {argsTuple->getType()},
                                                     {this}, "std.internal.gc");
        if (!constructFn)
            return nullptr;

        std::vector<Value *> callArgs = {argsTuple};
        return module->Nr<CallInstr>(module->Nr<VarValue>(constructFn), callArgs);
    }

    const char FuncType::NodeId = 0;

    std::vector<Generic> FuncType::doGetGenerics() const {
        auto t = getAstType();
        if (!t)
            return {};
        auto astType = t->getFunc();
        if (!astType)
            return {};

        std::vector<Generic> ret;
        for (auto &g: astType->funcGenerics) {
            if (auto cls = g.type->getClass())
                ret.emplace_back(
                        getModule()->getCache()->realizeType(cls, extractTypes(cls->generics)));
            else {
                seqassertn(g.type->getStatic()->expr->staticValue.type == ast::StaticValue::INT,
                           "IR only supports int statics [{}]", getSrcInfo());
                ret.emplace_back(g.type->getStatic()->expr->staticValue.getInt());
            }
        }

        return ret;
    }

    std::vector<Type *> FuncType::doGetUsedTypes() const {
        auto ret = argTypes;
        ret.push_back(rType);
        return ret;
    }

    const char DerivedType::NodeId = 0;

    const char PointerType::NodeId = 0;

    std::string PointerType::getInstanceName(Type *base) {
        return collie::format(FMT_STRING("Pointer[{}]"), base->referenceString());
    }

    const char OptionalType::NodeId = 0;

    std::string OptionalType::getInstanceName(Type *base) {
        return collie::format(FMT_STRING("Optional[{}]"), base->referenceString());
    }

    const char GeneratorType::NodeId = 0;

    std::string GeneratorType::getInstanceName(Type *base) {
        return collie::format(FMT_STRING("Generator[{}]"), base->referenceString());
    }

    const char IntNType::NodeId = 0;

    std::string IntNType::getInstanceName(unsigned int len, bool sign) {
        return collie::format(FMT_STRING("{}Int{}"), sign ? "" : "U", len);
    }

    const char VectorType::NodeId = 0;

    std::string VectorType::getInstanceName(unsigned int count, PrimitiveType *base) {
        return collie::format(FMT_STRING("Vector[{}, {}]"), count, base->referenceString());
    }

    const char UnionType::NodeId = 0;

    std::string UnionType::getInstanceName(const std::vector<types::Type *> &types) {
        std::vector<std::string> names;
        for (auto *type: types) {
            names.push_back(type->referenceString());
        }
        return collie::format(FMT_STRING("Union[{}]"),
                           collie::join(names.begin(), names.end(), ", "));
    }

} // namespace hercules::ir::types
