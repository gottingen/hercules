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

#include <algorithm>
#include <collie/strings/format.h>
#include <sstream>
#include <unordered_set>

#include <hercules/hir/util/format.h>
#include <hercules/hir/util/visitor.h>

namespace hercules::ir::util {

    struct NodeFormatter {
        const types::Type *type = nullptr;
        const Value *value = nullptr;
        const Var *var = nullptr;
        bool canShowFull = false;

        std::unordered_set<id_t> &seenNodes;
        std::unordered_set<std::string> &seenTypes;

        NodeFormatter(const types::Type *type, std::unordered_set<id_t> &seenNodes,
                      std::unordered_set<std::string> &seenTypes)
                : type(type), seenNodes(seenNodes), seenTypes(seenTypes) {}

        NodeFormatter(const Value *value, std::unordered_set<id_t> &seenNodes,
                      std::unordered_set<std::string> &seenTypes)
                : value(value), seenNodes(seenNodes), seenTypes(seenTypes) {}

        NodeFormatter(const Var *var, std::unordered_set<id_t> &seenNodes,
                      std::unordered_set<std::string> &seenTypes)
                : var(var), seenNodes(seenNodes), seenTypes(seenTypes) {}

        friend std::ostream &operator<<(std::ostream &os, const NodeFormatter &n);
    };

    namespace {
        std::string escapeString(const std::string &str) {
            std::stringstream escaped;
            for (char c: str) {
                switch (c) {
                    case '\a':
                        escaped << "\\a";
                        break;
                    case '\b':
                        escaped << "\\b";
                        break;
                    case '\f':
                        escaped << "\\f";
                        break;
                    case '\n':
                        escaped << "\\n";
                        break;
                    case '\r':
                        escaped << "\\r";
                        break;
                    case '\t':
                        escaped << "\\t";
                        break;
                    case '\v':
                        escaped << "\\v";
                        break;
                    case '\\':
                        escaped << "\\\\";
                        break;
                    case '\'':
                        escaped << "\\'";
                        break;
                    case '\"':
                        escaped << "\\\"";
                        break;
                    case '\?':
                        escaped << "\\\?";
                        break;
                    default:
                        escaped << c;
                }
            }
            return escaped.str();
        }

        class FormatVisitor : util::ConstVisitor {
        private:
            std::ostream &os;
            std::unordered_set<id_t> &seenNodes;
            std::unordered_set<std::string> &seenTypes;

        public:
            FormatVisitor(std::ostream &os, std::unordered_set<id_t> &seenNodes,
                          std::unordered_set<std::string> &seenTypes)
                    : os(os), seenNodes(seenNodes), seenTypes(seenTypes) {}

            virtual ~FormatVisitor() noexcept = default;

            void visit(const Module *v) override {
                auto types = makeFormatters(v->types_begin(), v->types_end(), true);
                auto vars = makeFormatters(v->begin(), v->end(), true);
                collie::print(os, FMT_STRING("(module\n(argv {})\n(types {})\n(vars {})\n{})"),
                           makeFormatter(v->getArgVar(), true),
                              collie::join(types.begin(), types.end(), "\n"),
                              collie::join(vars.begin(), vars.end(), "\n"),
                           makeFormatter(v->getMainFunc(), true));
            }

            void defaultVisit(const Node *) override { os << "(unknown_node)"; }

            void visit(const Var *v) override {
                collie::print(os, FMT_STRING("(var '\"{}\" {} (global {}) (external {}))"),
                           v->referenceString(), makeFormatter(v->getType()), v->isGlobal(),
                           v->isExternal());
            }

            void visit(const BodiedFunc *v) override {
                auto args = makeFormatters(v->arg_begin(), v->arg_end(), true);
                auto symbols = makeFormatters(v->begin(), v->end(), true);
                collie::print(os, FMT_STRING("(bodied_func '\"{}\" {}\n(args {})\n(vars {})\n{})"),
                           v->referenceString(), makeFormatter(v->getType()),
                              collie::join(args.begin(), args.end(), " "),
                              collie::join(symbols.begin(), symbols.end(), " "),
                           makeFormatter(v->getBody()));
            }

            void visit(const ExternalFunc *v) override {
                collie::print(os, FMT_STRING("(external_func '\"{}\" {})"), v->referenceString(),
                           makeFormatter(v->getType()));
            }

            void visit(const InternalFunc *v) override {
                collie::print(os, FMT_STRING("(internal_func '\"{}\" {})"), v->referenceString(),
                           makeFormatter(v->getType()));
            }

            void visit(const LLVMFunc *v) override {
                std::vector<std::string> literals;

                for (auto it = v->literal_begin(); it != v->literal_end(); ++it) {
                    const auto &l = *it;
                    if (l.isStatic()) {
                        literals.push_back(collie::format(FMT_STRING("(static {})"), l.getStaticValue()));
                    } else if (l.isStaticStr()) {
                        literals.push_back(
                                collie::format(FMT_STRING("(static \"{}\")"), l.getStaticStringValue()));
                    } else {
                        literals.push_back(
                                collie::format(FMT_STRING("(type {})"), makeFormatter(l.getTypeValue())));
                    }
                }

                collie::print(os,
                           FMT_STRING("(llvm_func '\"{}\" {}\n(decls \"{}\")\n"
                                      "\"{}\"\n(literals {}))"),
                           v->referenceString(), makeFormatter(v->getType()),
                           escapeString(v->getLLVMDeclarations()), escapeString(v->getLLVMBody()),
                           collie::join(literals.begin(), literals.end(), "\n"));
            }

            void visit(const VarValue *v) override {
                collie::print(os, FMT_STRING("'\"{}\""), v->getVar()->referenceString());
            }

            void visit(const PointerValue *v) override {
                collie::print(os, FMT_STRING("(ptr '\"{}\")"), v->getVar()->referenceString());
            }

            void visit(const SeriesFlow *v) override {
                auto series = makeFormatters(v->begin(), v->end());
                collie::print(os, FMT_STRING("(series\n{}\n)"),
                           collie::join(series.begin(), series.end(), "\n"));
            }

            void visit(const IfFlow *v) override {
                collie::print(os, FMT_STRING("(if {}\n{}\n{}\n)"), makeFormatter(v->getCond()),
                           makeFormatter(v->getTrueBranch()), makeFormatter(v->getFalseBranch()));
            }

            void visit(const WhileFlow *v) override {
                collie::print(os, FMT_STRING("(while {}\n{}\n)"), makeFormatter(v->getCond()),
                           makeFormatter(v->getBody()));
            }

            void visit(const ForFlow *v) override {
                collie::print(os, FMT_STRING("({}for {}\n{}\n{}\n)"), v->isParallel() ? "par_" : "",
                           makeFormatter(v->getIter()), makeFormatter(v->getVar()),
                           makeFormatter(v->getBody()));
            }

            void visit(const ImperativeForFlow *v) override {
                collie::print(os, FMT_STRING("({}imp_for {}\n{}\n{}\n{}\n{}\n)"),
                           v->isParallel() ? "par_" : "", makeFormatter(v->getStart()),
                           v->getStep(), makeFormatter(v->getEnd()), makeFormatter(v->getVar()),
                           makeFormatter(v->getBody()));
            }

            void visit(const TryCatchFlow *v) override {
                std::vector<std::string> catches;

                for (auto &c: *v) {
                    catches.push_back(
                            collie::format(FMT_STRING("(catch {} {}\n{}\n)"), makeFormatter(c.getType()),
                                        makeFormatter(c.getVar()), makeFormatter(c.getHandler())));
                }

                collie::print(os, FMT_STRING("(try {}\n{}\n(finally\n{}\n)\n)"),
                           makeFormatter(v->getBody()),
                           collie::join(catches.begin(), catches.end(), "\n"),
                           makeFormatter(v->getFinally()));
            }

            void visit(const PipelineFlow *v) override {
                std::vector<std::string> stages;
                for (const auto &s: *v) {
                    auto args = makeFormatters(s.begin(), s.end());
                    stages.push_back(collie::format(
                            FMT_STRING("(stage {} {}\n(generator {})\n(parallel {}))"),
                            makeFormatter(s.getCallee()), collie::join(args.begin(), args.end(), "\n"),
                            s.isGenerator(), s.isParallel()));
                }
                collie::print(os, FMT_STRING("(pipeline {})"),
                           collie::join(stages.begin(), stages.end(), "\n"));
            }

            void visit(const dsl::CustomFlow *v) override { v->doFormat(os); }

            void visit(const IntConst *v) override {
                collie::print(os, FMT_STRING("{}"), v->getVal());
            }

            void visit(const FloatConst *v) override {
                collie::print(os, FMT_STRING("{}"), v->getVal());
            }

            void visit(const BoolConst *v) override {
                collie::print(os, FMT_STRING("{}"), v->getVal());
            }

            void visit(const StringConst *v) override {
                collie::print(os, FMT_STRING("\"{}\""), escapeString(v->getVal()));
            }

            void visit(const dsl::CustomConst *v) override { v->doFormat(os); }

            void visit(const AssignInstr *v) override {
                collie::print(os, FMT_STRING("(assign {} {})"), makeFormatter(v->getLhs()),
                           makeFormatter(v->getRhs()));
            }

            void visit(const ExtractInstr *v) override {
                collie::print(os, FMT_STRING("(extract {} \"{}\")"), makeFormatter(v->getVal()),
                           v->getField());
            }

            void visit(const InsertInstr *v) override {
                collie::print(os, FMT_STRING("(insert {} \"{}\" {})"), makeFormatter(v->getLhs()),
                           v->getField(), makeFormatter(v->getRhs()));
            }

            void visit(const CallInstr *v) override {
                auto args = makeFormatters(v->begin(), v->end());
                collie::print(os, FMT_STRING("(call {}\n{}\n)"), makeFormatter(v->getCallee()),
                           collie::join(args.begin(), args.end(), "\n"));
            }

            void visit(const StackAllocInstr *v) override {
                collie::print(os, FMT_STRING("(stack_alloc {} {})"), makeFormatter(v->getArrayType()),
                           v->getCount());
            }

            void visit(const TypePropertyInstr *v) override {
                std::string property;
                if (v->getProperty() == TypePropertyInstr::Property::IS_ATOMIC) {
                    property = "atomic";
                } else if (v->getProperty() == TypePropertyInstr::Property::SIZEOF) {
                    property = "sizeof";
                } else {
                    property = "unknown";
                }
                collie::print(os, FMT_STRING("(property {} {})"), property,
                           makeFormatter(v->getInspectType()));
            }

            void visit(const YieldInInstr *v) override {
                collie::print(os, FMT_STRING("(yield_in {})"), makeFormatter(v->getType()));
            }

            void visit(const TernaryInstr *v) override {
                collie::print(os, FMT_STRING("(select {}\n{}\n{}\n)"), makeFormatter(v->getCond()),
                           makeFormatter(v->getTrueValue()), makeFormatter(v->getFalseValue()));
            }

            void visit(const BreakInstr *v) override {
                os << "(break " << (v->getLoop() ? v->getLoop()->getId() : -1) << ')';
            }

            void visit(const ContinueInstr *v) override {
                os << "(continue " << (v->getLoop() ? v->getLoop()->getId() : -1) << ')';
            }

            void visit(const ReturnInstr *v) override {
                collie::print(os, FMT_STRING("(return {})"), makeFormatter(v->getValue()));
            }

            void visit(const YieldInstr *v) override {
                collie::print(os, FMT_STRING("(yield {})"), makeFormatter(v->getValue()));
            }

            void visit(const ThrowInstr *v) override {
                collie::print(os, FMT_STRING("(throw {})"), makeFormatter(v->getValue()));
            }

            void visit(const FlowInstr *v) override {
                collie::print(os, FMT_STRING("(flow {} {})"), makeFormatter(v->getFlow()),
                           makeFormatter(v->getValue()));
            }

            void visit(const dsl::CustomInstr *v) override { v->doFormat(os); }

            void visit(const types::IntType *v) override {
                collie::print(os, FMT_STRING("(int '\"{}\")"), v->referenceString());
            }

            void visit(const types::FloatType *v) override {
                collie::print(os, FMT_STRING("(float '\"{}\")"), v->referenceString());
            }

            void visit(const types::Float32Type *v) override {
                collie::print(os, FMT_STRING("(float32 '\"{}\")"), v->referenceString());
            }

            void visit(const types::Float16Type *v) override {
                collie::print(os, FMT_STRING("(float16 '\"{}\")"), v->referenceString());
            }

            void visit(const types::BFloat16Type *v) override {
                collie::print(os, FMT_STRING("(bfloat16 '\"{}\")"), v->referenceString());
            }

            void visit(const types::Float128Type *v) override {
                collie::print(os, FMT_STRING("(float128 '\"{}\")"), v->referenceString());
            }

            void visit(const types::BoolType *v) override {
                collie::print(os, FMT_STRING("(bool '\"{}\")"), v->referenceString());
            }

            void visit(const types::ByteType *v) override {
                collie::print(os, FMT_STRING("(byte '\"{}\")"), v->referenceString());
            }

            void visit(const types::VoidType *v) override {
                collie::print(os, FMT_STRING("(void '\"{}\")"), v->referenceString());
            }

            void visit(const types::RecordType *v) override {
                std::vector<std::string> fields;
                std::vector<NodeFormatter> formatters;
                for (const auto &m: *v) {
                    fields.push_back(collie::format(FMT_STRING("(\"{}\" {})"), m.getName(),
                                                 makeFormatter(m.getType())));
                }

                collie::print(os, FMT_STRING("(record '\"{}\" {})"), v->referenceString(),
                           collie::join(fields.begin(), fields.end(), " "));
            }

            void visit(const types::RefType *v) override {
                collie::print(os, FMT_STRING("(ref '\"{}\" {})"), v->referenceString(),
                           makeFormatter(v->getContents()));
            }

            void visit(const types::FuncType *v) override {
                auto args = makeFormatters(v->begin(), v->end());
                collie::print(os, FMT_STRING("(func '\"{}\" {}{} {})"), v->referenceString(),
                           collie::join(args.begin(), args.end(), " "),
                           (v->isVariadic() ? " ..." : ""), makeFormatter(v->getReturnType()));
            }

            void visit(const types::OptionalType *v) override {
                collie::print(os, FMT_STRING("(optional '\"{}\" {})"), v->referenceString(),
                           makeFormatter(v->getBase()));
            }

            void visit(const types::PointerType *v) override {
                collie::print(os, FMT_STRING("(pointer '\"{}\" {})"), v->referenceString(),
                           makeFormatter(v->getBase()));
            }

            void visit(const types::GeneratorType *v) override {
                collie::print(os, FMT_STRING("(generator '\"{}\" {})"), v->referenceString(),
                           makeFormatter(v->getBase()));
            }

            void visit(const types::IntNType *v) override {
                collie::print(os, FMT_STRING("(intn '\"{}\" {} (signed {}))"), v->referenceString(),
                           v->getLen(), v->isSigned());
            }

            void visit(const types::VectorType *v) override {
                collie::print(os, FMT_STRING("(vector '\"{}\" {} (count {}))"), v->referenceString(),
                           makeFormatter(v->getBase()), v->getCount());
            }

            void visit(const types::UnionType *v) override {
                auto types = makeFormatters(v->begin(), v->end());
                collie::print(os, FMT_STRING("(union '\"{}\" {})"), v->referenceString(),
                           collie::join(types.begin(), types.end(), " "));
            }

            void visit(const dsl::types::CustomType *v) override { v->doFormat(os); }

            void format(const Node *n) {
                if (n)
                    n->accept(*this);
                else
                    os << "(null)";
            }

            void format(const types::Type *t, bool canShowFull = false) {
                if (t) {
                    if (seenTypes.find(t->getName()) != seenTypes.end() || !canShowFull)
                        collie::print(os, FMT_STRING("(type '\"{}\")"), t->referenceString());
                    else {
                        seenTypes.insert(t->getName());
                        t->accept(*this);
                    }
                } else
                    os << "(null)";
            }

            void format(const Value *t) {
                if (t) {
                    if (seenNodes.find(t->getId()) != seenNodes.end())
                        collie::print(os, FMT_STRING("(value '\"{}\")"), t->referenceString());
                    else {
                        seenNodes.insert(t->getId());
                        t->accept(*this);
                    }

                } else
                    os << "(null)";
            }

            void format(const Var *t, bool canShowFull = false) {
                if (t) {
                    if (seenNodes.find(t->getId()) != seenNodes.end() || !canShowFull)
                        collie::print(os, FMT_STRING("(var '\"{}\")"), t->referenceString());
                    else {
                        seenNodes.insert(t->getId());
                        t->accept(*this);
                    }
                } else
                    os << "(null)";
            }

        private:
            NodeFormatter makeFormatter(const types::Type *node, bool canShowFull = false) {
                auto ret = NodeFormatter(node, seenNodes, seenTypes);
                ret.canShowFull = canShowFull;
                return ret;
            }

            NodeFormatter makeFormatter(const Value *node) {
                return NodeFormatter(node, seenNodes, seenTypes);
            }

            NodeFormatter makeFormatter(const Var *node, bool canShowFull = false) {
                auto ret = NodeFormatter(node, seenNodes, seenTypes);
                ret.canShowFull = canShowFull;
                return ret;
            }

            template<typename It>
            std::vector<NodeFormatter> makeFormatters(It begin, It end) {
                std::vector<NodeFormatter> ret;
                while (begin != end) {
                    ret.push_back(makeFormatter(*begin));
                    ++begin;
                }
                return ret;
            }

            template<typename It>
            std::vector<NodeFormatter> makeFormatters(It begin, It end, bool canShowFull) {
                std::vector<NodeFormatter> ret;
                while (begin != end) {
                    ret.push_back(makeFormatter(*begin, canShowFull));
                    ++begin;
                }
                return ret;
            }
        };
    } // namespace

    std::ostream &operator<<(std::ostream &os, const NodeFormatter &n) {
        FormatVisitor fv(os, n.seenNodes, n.seenTypes);
        if (n.type)
            fv.format(n.type, n.canShowFull);
        else if (n.value)
            fv.format(n.value);
        else
            fv.format(n.var, n.canShowFull);
        return os;
    }

    std::string format(const Node *node) {
        std::stringstream ss;
        format(ss, node);
        return ss.str();
    }

    std::ostream &format(std::ostream &os, const Node *node) {
        std::unordered_set<id_t> seenNodes;
        std::unordered_set<std::string> seenTypes;

        FormatVisitor fv(os, seenNodes, seenTypes);
        fv.format(node);

        return os;
    }

} // namespace hercules::ir::util

template<>
struct collie::formatter<hercules::ir::util::NodeFormatter> : collie::ostream_formatter {
};
