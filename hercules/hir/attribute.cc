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

#include <hercules/hir/attribute.h>

#include <hercules/hir/func.h>
#include <hercules/hir/util/cloning.h>
#include <hercules/hir/value.h>
#include <collie/strings/format.h>

namespace hercules::ir {

    const std::string KeyValueAttribute::AttributeName = "kvAttribute";

    bool KeyValueAttribute::has(const std::string &key) const {
        return attributes.find(key) != attributes.end();
    }

    std::string KeyValueAttribute::get(const std::string &key) const {
        auto it = attributes.find(key);
        return it != attributes.end() ? it->second : "";
    }

    std::ostream &KeyValueAttribute::doFormat(std::ostream &os) const {
        std::vector<std::string> keys;
        for (auto &val: attributes)
            keys.push_back(val.second);
        collie::print(os, FMT_STRING("{}"), collie::join(keys.begin(), keys.end(), ","));
        return os;
    }

    const std::string MemberAttribute::AttributeName = "memberAttribute";

    std::ostream &MemberAttribute::doFormat(std::ostream &os) const {
        std::vector<std::string> strings;
        for (auto &val: memberSrcInfo)
            strings.push_back(collie::format(FMT_STRING("{}={}"), val.first, val.second));
        collie::print(os, FMT_STRING("({})"), collie::join(strings.begin(), strings.end(), ","));
        return os;
    }

    const std::string SrcInfoAttribute::AttributeName = "srcInfoAttribute";

    const std::string DocstringAttribute::AttributeName = "docstringAttribute";

    const std::string TupleLiteralAttribute::AttributeName = "tupleLiteralAttribute";

    std::unique_ptr<Attribute> TupleLiteralAttribute::clone(util::CloneVisitor &cv) const {
        std::vector<Value *> elementsCloned;
        for (auto *val: elements)
            elementsCloned.push_back(cv.clone(val));
        return std::make_unique<TupleLiteralAttribute>(elementsCloned);
    }

    std::unique_ptr<Attribute>
    TupleLiteralAttribute::forceClone(util::CloneVisitor &cv) const {
        std::vector<Value *> elementsCloned;
        for (auto *val: elements)
            elementsCloned.push_back(cv.forceClone(val));
        return std::make_unique<TupleLiteralAttribute>(elementsCloned);
    }

    std::ostream &TupleLiteralAttribute::doFormat(std::ostream &os) const {
        std::vector<std::string> strings;
        for (auto *val: elements)
            strings.push_back(collie::format(FMT_STRING("{}"), *val));
        collie::print(os, FMT_STRING("({})"), collie::join(strings.begin(), strings.end(), ","));
        return os;
    }

    const std::string ListLiteralAttribute::AttributeName = "listLiteralAttribute";

    std::unique_ptr<Attribute> ListLiteralAttribute::clone(util::CloneVisitor &cv) const {
        std::vector<LiteralElement> elementsCloned;
        for (auto &e: elements)
            elementsCloned.push_back({cv.clone(e.value), e.star});
        return std::make_unique<ListLiteralAttribute>(elementsCloned);
    }

    std::unique_ptr<Attribute>
    ListLiteralAttribute::forceClone(util::CloneVisitor &cv) const {
        std::vector<LiteralElement> elementsCloned;
        for (auto &e: elements)
            elementsCloned.push_back({cv.forceClone(e.value), e.star});
        return std::make_unique<ListLiteralAttribute>(elementsCloned);
    }

    std::ostream &ListLiteralAttribute::doFormat(std::ostream &os) const {
        std::vector<std::string> strings;
        for (auto &e: elements)
            strings.push_back(collie::format(FMT_STRING("{}{}"), e.star ? "*" : "", *e.value));
        collie::print(os, FMT_STRING("[{}]"), collie::join(strings.begin(), strings.end(), ","));
        return os;
    }

    const std::string SetLiteralAttribute::AttributeName = "setLiteralAttribute";

    std::unique_ptr<Attribute> SetLiteralAttribute::clone(util::CloneVisitor &cv) const {
        std::vector<LiteralElement> elementsCloned;
        for (auto &e: elements)
            elementsCloned.push_back({cv.clone(e.value), e.star});
        return std::make_unique<SetLiteralAttribute>(elementsCloned);
    }

    std::unique_ptr<Attribute>
    SetLiteralAttribute::forceClone(util::CloneVisitor &cv) const {
        std::vector<LiteralElement> elementsCloned;
        for (auto &e: elements)
            elementsCloned.push_back({cv.forceClone(e.value), e.star});
        return std::make_unique<SetLiteralAttribute>(elementsCloned);
    }

    std::ostream &SetLiteralAttribute::doFormat(std::ostream &os) const {
        std::vector<std::string> strings;
        for (auto &e: elements)
            strings.push_back(collie::format(FMT_STRING("{}{}"), e.star ? "*" : "", *e.value));
        collie::print(os, FMT_STRING("set([{}])"),
                   collie::join(strings.begin(), strings.end(), ","));
        return os;
    }

    const std::string DictLiteralAttribute::AttributeName = "dictLiteralAttribute";

    std::unique_ptr<Attribute> DictLiteralAttribute::clone(util::CloneVisitor &cv) const {
        std::vector<DictLiteralAttribute::KeyValuePair> elementsCloned;
        for (auto &val: elements)
            elementsCloned.push_back(
                    {cv.clone(val.key), val.value ? cv.clone(val.value) : nullptr});
        return std::make_unique<DictLiteralAttribute>(elementsCloned);
    }

    std::unique_ptr<Attribute>
    DictLiteralAttribute::forceClone(util::CloneVisitor &cv) const {
        std::vector<DictLiteralAttribute::KeyValuePair> elementsCloned;
        for (auto &val: elements)
            elementsCloned.push_back(
                    {cv.forceClone(val.key), val.value ? cv.forceClone(val.value) : nullptr});
        return std::make_unique<DictLiteralAttribute>(elementsCloned);
    }

    std::ostream &DictLiteralAttribute::doFormat(std::ostream &os) const {
        std::vector<std::string> strings;
        for (auto &val: elements) {
            if (val.value) {
                strings.push_back(collie::format(FMT_STRING("{}:{}"), *val.key, *val.value));
            } else {
                strings.push_back(collie::format(FMT_STRING("**{}"), *val.key));
            }
        }
        collie::print(os, FMT_STRING("dict([{}])"),
                   collie::join(strings.begin(), strings.end(), ","));
        return os;
    }

    const std::string PartialFunctionAttribute::AttributeName = "partialFunctionAttribute";

    std::unique_ptr<Attribute>
    PartialFunctionAttribute::clone(util::CloneVisitor &cv) const {
        std::vector<Value *> argsCloned;
        for (auto *val: args)
            argsCloned.push_back(cv.clone(val));
        return std::make_unique<PartialFunctionAttribute>(name, argsCloned);
    }

    std::unique_ptr<Attribute>
    PartialFunctionAttribute::forceClone(util::CloneVisitor &cv) const {
        std::vector<Value *> argsCloned;
        for (auto *val: args)
            argsCloned.push_back(cv.forceClone(val));
        return std::make_unique<PartialFunctionAttribute>(name, argsCloned);
    }

    std::ostream &PartialFunctionAttribute::doFormat(std::ostream &os) const {
        std::vector<std::string> strings;
        for (auto *val: args)
            strings.push_back(val ? collie::format(FMT_STRING("{}"), *val) : "...");
        collie::print(os, FMT_STRING("{}({})"), name,
                   collie::join(strings.begin(), strings.end(), ","));
        return os;
    }

} // namespace hercules::ir
