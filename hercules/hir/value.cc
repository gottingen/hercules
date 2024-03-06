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

#include "hercules/hir/value.h"

#include "hercules/hir/instr.h"
#include "hercules/hir/module.h"

namespace hercules::ir {

    const char Value::NodeId = 0;

    Value *Value::operator==(Value &other) {
        return doBinaryOp(Module::EQ_MAGIC_NAME, other);
    }

    Value *Value::operator!=(Value &other) {
        return doBinaryOp(Module::NE_MAGIC_NAME, other);
    }

    Value *Value::operator<(Value &other) {
        return doBinaryOp(Module::LT_MAGIC_NAME, other);
    }

    Value *Value::operator>(Value &other) {
        return doBinaryOp(Module::GT_MAGIC_NAME, other);
    }

    Value *Value::operator<=(Value &other) {
        return doBinaryOp(Module::LE_MAGIC_NAME, other);
    }

    Value *Value::operator>=(Value &other) {
        return doBinaryOp(Module::GE_MAGIC_NAME, other);
    }

    Value *Value::operator+() { return doUnaryOp(Module::POS_MAGIC_NAME); }

    Value *Value::operator-() { return doUnaryOp(Module::NEG_MAGIC_NAME); }

    Value *Value::operator~() { return doUnaryOp(Module::INVERT_MAGIC_NAME); }

    Value *Value::operator+(Value &other) {
        return doBinaryOp(Module::ADD_MAGIC_NAME, other);
    }

    Value *Value::operator-(Value &other) {
        return doBinaryOp(Module::SUB_MAGIC_NAME, other);
    }

    Value *Value::operator*(Value &other) {
        return doBinaryOp(Module::MUL_MAGIC_NAME, other);
    }

    Value *Value::matMul(Value &other) {
        return doBinaryOp(Module::MATMUL_MAGIC_NAME, other);
    }

    Value *Value::trueDiv(Value &other) {
        return doBinaryOp(Module::TRUE_DIV_MAGIC_NAME, other);
    }

    Value *Value::operator/(Value &other) {
        return doBinaryOp(Module::FLOOR_DIV_MAGIC_NAME, other);
    }

    Value *Value::operator%(Value &other) {
        return doBinaryOp(Module::MOD_MAGIC_NAME, other);
    }

    Value *Value::pow(Value &other) { return doBinaryOp(Module::POW_MAGIC_NAME, other); }

    Value *Value::operator<<(Value &other) {
        return doBinaryOp(Module::LSHIFT_MAGIC_NAME, other);
    }

    Value *Value::operator>>(Value &other) {
        return doBinaryOp(Module::RSHIFT_MAGIC_NAME, other);
    }

    Value *Value::operator&(Value &other) {
        return doBinaryOp(Module::AND_MAGIC_NAME, other);
    }

    Value *Value::operator|(Value &other) {
        return doBinaryOp(Module::OR_MAGIC_NAME, other);
    }

    Value *Value::operator^(Value &other) {
        return doBinaryOp(Module::XOR_MAGIC_NAME, other);
    }

    Value *Value::operator||(Value &other) {
        auto *module = getModule();
        return module->Nr<TernaryInstr>(toBool(), module->getBool(true), other.toBool());
    }

    Value *Value::operator&&(Value &other) {
        auto *module = getModule();
        return module->Nr<TernaryInstr>(toBool(), other.toBool(), module->getBool(false));
    }

    Value *Value::operator[](Value &other) {
        return doBinaryOp(Module::GETITEM_MAGIC_NAME, other);
    }

    Value *Value::toInt() { return doUnaryOp(Module::INT_MAGIC_NAME); }

    Value *Value::toFloat() { return doUnaryOp(Module::FLOAT_MAGIC_NAME); }

    Value *Value::toBool() { return doUnaryOp(Module::BOOL_MAGIC_NAME); }

    Value *Value::toStr() { return doUnaryOp(Module::REPR_MAGIC_NAME); }

    Value *Value::len() { return doUnaryOp(Module::LEN_MAGIC_NAME); }

    Value *Value::iter() { return doUnaryOp(Module::ITER_MAGIC_NAME); }

    Value *Value::doUnaryOp(const std::string &name) {
        auto *module = getModule();
        auto *fn = module->getOrRealizeMethod(getType(), name,
                                              std::vector<types::Type *>{getType()});

        if (!fn)
            return nullptr;

        auto *fnVal = module->Nr<VarValue>(fn);
        return (*fnVal)(*this);
    }

    Value *Value::doBinaryOp(const std::string &name, Value &other) {
        auto *module = getModule();
        auto *fn = module->getOrRealizeMethod(
                getType(), name, std::vector<types::Type *>{getType(), other.getType()});

        if (!fn)
            return nullptr;

        auto *fnVal = module->Nr<VarValue>(fn);
        return (*fnVal)(*this, other);
    }

    Value *Value::doCall(const std::vector<Value *> &args) {
        auto *module = getModule();
        return module->Nr<CallInstr>(this, args);
    }

} // namespace hercules:;ir
