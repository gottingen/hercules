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

#include "hercules/cir/base.h"
#include "hercules/cir/types/types.h"
#include "hercules/cir/util/packs.h"

namespace hercules {
namespace ir {

class Func;

class Value : public ReplaceableNodeBase<Value>, public IdMixin {
public:
  static const char NodeId;

  /// Constructs a value.
  /// @param the value's name
  explicit Value(std::string name = "") : ReplaceableNodeBase(std::move(name)) {}

  virtual ~Value() noexcept = default;

  std::string referenceString() const final {
    return fmt::format(FMT_STRING("{}.{}"), getName(), getId());
  }

  std::vector<Value *> getUsedValues() final { return getActual()->doGetUsedValues(); }
  std::vector<const Value *> getUsedValues() const final {
    auto ret = getActual()->doGetUsedValues();
    return std::vector<const Value *>(ret.begin(), ret.end());
  }
  int replaceUsedValue(id_t id, Value *newValue) final {
    return getActual()->doReplaceUsedValue(id, newValue);
  }
  using Node::replaceUsedValue;

  std::vector<types::Type *> getUsedTypes() const final {
    return getActual()->doGetUsedTypes();
  }
  int replaceUsedType(const std::string &name, types::Type *newType) final {
    return getActual()->doReplaceUsedType(name, newType);
  }
  using Node::replaceUsedType;

  std::vector<Var *> getUsedVariables() final {
    return getActual()->doGetUsedVariables();
  }
  std::vector<const Var *> getUsedVariables() const final {
    auto ret = getActual()->doGetUsedVariables();
    return std::vector<const Var *>(ret.begin(), ret.end());
  }
  int replaceUsedVariable(id_t id, Var *newVar) final {
    return getActual()->doReplaceUsedVariable(id, newVar);
  }
  using Node::replaceUsedVariable;

  /// @return the value's type
  types::Type *getType() const { return getActual()->doGetType(); }

  id_t getId() const override { return getActual()->id; }

  Value *operator==(Value &other);
  Value *operator!=(Value &other);
  Value *operator<(Value &other);
  Value *operator>(Value &other);
  Value *operator<=(Value &other);
  Value *operator>=(Value &other);

  Value *operator+();
  Value *operator-();
  Value *operator~();

  Value *operator+(Value &other);
  Value *operator-(Value &other);
  Value *operator*(Value &other);
  Value *matMul(Value &other);
  Value *trueDiv(Value &other);
  Value *operator/(Value &other);
  Value *operator%(Value &other);
  Value *pow(Value &other);
  Value *operator<<(Value &other);
  Value *operator>>(Value &other);
  Value *operator&(Value &other);
  Value *operator|(Value &other);
  Value *operator^(Value &other);

  Value *operator||(Value &other);
  Value *operator&&(Value &other);

  template <typename... Args> Value *operator()(Args &&...args) {
    std::vector<Value *> dst;
    util::stripPack(dst, std::forward<Args>(args)...);
    return doCall(dst);
  }
  Value *operator[](Value &other);

  Value *toInt();
  Value *toFloat();
  Value *toBool();
  Value *toStr();

  Value *len();
  Value *iter();

private:
  Value *doUnaryOp(const std::string &name);
  Value *doBinaryOp(const std::string &name, Value &other);

  Value *doCall(const std::vector<Value *> &args);

  virtual types::Type *doGetType() const = 0;

  virtual std::vector<Value *> doGetUsedValues() const { return {}; }
  virtual int doReplaceUsedValue(id_t id, Value *newValue) { return 0; }

  virtual std::vector<types::Type *> doGetUsedTypes() const { return {}; }
  virtual int doReplaceUsedType(const std::string &name, types::Type *newType) {
    return 0;
  }

  virtual std::vector<Var *> doGetUsedVariables() const { return {}; }
  virtual int doReplaceUsedVariable(id_t id, Var *newVar) { return 0; }
};

} // namespace ir
} // namespace hercules

template <typename T>
struct fmt::formatter<
    T, std::enable_if_t<std::is_base_of<hercules::ir::Value, T>::value, char>>
    : fmt::ostream_formatter {};
