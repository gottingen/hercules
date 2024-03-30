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
#include <any>
#include <cassert>
#include <cctype>
#include <cstring>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <hercules/parser/peg/peglib.h>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <collie/strings/format.h>
#include <collie/cli/cli.h>
#include <hercules/config/config.h>

using namespace std;

std::string grammar_file;
std::string output_file;
std::string ns;

string escape(const string &str) {
    string r;
    for (unsigned char c: str) {
        switch (c) {
            case '\n':
                r += "\\\\n";
                break;
            case '\r':
                r += "\\\\r";
                break;
            case '\t':
                r += "\\\\t";
                break;
            case '\\':
                r += "\\\\";
                break;
            case '"':
                r += "\\\"";
                break;
            default:
                if (c < 32 || c >= 127)
                    r += collie::format("\\\\x{:x}", c);
                else
                    r += c;
        }
    }
    return r;
}

template<typename T>
string join(const T &items, const string &delim = " ", int start = 0, int end = -1) {
    string s;
    if (end == -1)
        end = items.size();
    for (int i = start; i < end; i++)
        s += (i > start ? delim : "") + items[i];
    return s;
}

// const string PREDICATE = ".predicate";
// bool is_predicate(const std::string &name) {
//   return (name.size() > PREDICATE.size() && name.substr(name.size() -
//   PREDICATE.size()) == PREDICATE);
// }

class PrintVisitor : public peg::Ope::Visitor {
    vector<string> v;

public:
    static string parse(const shared_ptr<peg::Ope> &op) {
        PrintVisitor v;
        op->accept(v);
        if (v.v.size()) {
            if (v.v[0].empty())
                return collie::format("P[\"{}\"]", v.v[1]);
            else
                return collie::format("{}({})", v.v[0], join(v.v, ", ", 1));
        }
        return "-";
    };

private:
    void visit(peg::Sequence &s) override {
        v = {"seq"};
        for (auto &o: s.opes_)
            v.push_back(parse(o));
    }

    void visit(peg::PrioritizedChoice &s) override {
        v = {"cho"};
        for (auto &o: s.opes_)
            v.push_back(parse(o));
    }

    void visit(peg::Repetition &s) override {
        if (s.is_zom())
            v = {"zom", parse(s.ope_)};
        else if (s.min_ == 1 && s.max_ == std::numeric_limits<size_t>::max())
            v = {"oom", parse(s.ope_)};
        else if (s.min_ == 0 && s.max_ == 1)
            v = {"opt", parse(s.ope_)};
        else
            v = {"rep", parse(s.ope_), to_string(s.min_), to_string(s.max_)};
    }

    void visit(peg::AndPredicate &s) override { v = {"apd", parse(s.ope_)}; }

    void visit(peg::NotPredicate &s) override { v = {"npd", parse(s.ope_)}; }

    void visit(peg::LiteralString &s) override {
        v = {s.ignore_case_ ? "liti" : "lit", collie::format("\"{}\"", escape(s.lit_))};
    }

    void visit(peg::CharacterClass &s) override {
        vector<string> sv;
        for (auto &c: s.ranges_)
            sv.push_back(collie::format("{{0x{:x}, 0x{:x}}}", (int) c.first, (int) c.second));
        v = {s.negated_ ? "ncls" : "cls", "vc{" + join(sv, ",") + "}"};
    }

    void visit(peg::Character &s) override { v = {"chr", collie::format("'{}'", s.ch_)}; }

    void visit(peg::AnyCharacter &s) override { v = {"dot"}; }

    void visit(peg::Cut &s) override { v = {"cut"}; }

    void visit(peg::Reference &s) override {
        if (s.is_macro_) {
            vector<string> vs;
            for (auto &o: s.args_)
                vs.push_back(parse(o));
            v = {"ref", "P", collie::format("\"{}\"", s.name_),
                 "\"\"", "true", "{" + join(vs, ", ") + "}"};
        } else {
            v = {"ref", "P", collie::format("\"{}\"", s.name_)};
        }
    }

    void visit(peg::TokenBoundary &s) override { v = {"tok", parse(s.ope_)}; }

    void visit(peg::Ignore &s) override { v = {"ign", parse(s.ope_)}; }

    void visit(peg::Recovery &s) override { v = {"rec", parse(s.ope_)}; }
    // infix TODO
};

void generate() {
    peg::parser parser;
    collie::print("Generating grammar from {}\n", grammar_file);
    ifstream ifs(grammar_file);
    string g((istreambuf_iterator<char>(ifs)), istreambuf_iterator<char>());
    ifs.close();

    string start;
    peg::Rules dummy = {};
    if (string(ns) == "hercules")
        dummy["NLP"] = peg::usr([](const char *, size_t, peg::SemanticValues &,
                                   any &) -> size_t { return -1; });
    bool enablePackratParsing;
    string preamble;
    peg::Log log = [](size_t line, size_t col, const string &msg, const string &rule) {
        cerr << line << ":" << col << ": " << msg << " (" << rule << ")\n";
    };
    auto grammar = peg::ParserGenerator::get_instance().perform_core(
            g.c_str(), g.size(), dummy, start, enablePackratParsing, preamble, log);
    assert(grammar);

    string rules, actions, actionFns;
    string action_preamble = "  auto &CTX = any_cast<ParseContext &>(DT);\n";
    string const_action_preamble =
            "  const auto &CTX = any_cast<const ParseContext &>(DT);\n";
    string loc_preamble = "  const auto &LI = VS.line_info();\n"
                          "  auto LOC = hercules::SrcInfo(\n"
                          "    VS.path, LI.first + CTX.line_offset,\n"
                          "    LI.second + CTX.col_offset,\n"
                          "    VS.sv().size());\n";

    for (auto &[name, def]: *grammar) {
        auto op = def.get_core_operator();
        if (dummy.find(name) != dummy.end())
            continue;

        rules += collie::format("  {}P[\"{}\"] <= {};\n", def.ignoreSemanticValue ? "~" : "",
                                name, PrintVisitor::parse(op));
        rules += collie::format("  P[\"{}\"].name = \"{}\";\n", name, escape(name));
        if (def.is_macro)
            rules += collie::format("  P[\"{}\"].is_macro = true;\n", name);
        if (!def.enable_memoize)
            rules += collie::format("  P[\"{}\"].enable_memoize = false;\n", name);
        if (!def.params.empty()) {
            vector<string> params;
            for (auto &p: def.params)
                params.push_back(collie::format("\"{}\"", escape(p)));
            rules += collie::format("  P[\"{}\"].params = {{{}}};\n", name, join(params, ", "));
        }

        string code = op->code;
        if (code.empty()) {
            bool all_empty = true;
            if (auto ope = dynamic_cast<peg::PrioritizedChoice *>(op.get())) {
                for (int i = 0; i < ope->opes_.size(); i++)
                    if (!ope->opes_[i]->code.empty()) {
                        code +=
                                collie::format("  if (VS.choice() == {}) {}\n", i, ope->opes_[i]->code);
                        all_empty = false;
                    } else {
                        code += collie::format("  if (VS.choice() == {}) return V0;\n", i);
                    }
            }
            if (all_empty)
                code = "";
            if (!code.empty())
                code = "{\n" + code + "}";
        }
        if (!code.empty()) {
            code = code.substr(1, code.size() - 2);
            if (code.find("LOC") != std::string::npos)
                code = loc_preamble + code;
            if (code.find("CTX") != std::string::npos)
                code = action_preamble + code;
            actions += collie::format("P[\"{}\"] = fn_{};\n", name, name);
            actionFns += collie::format(
                    "auto fn_{}(peg::SemanticValues &VS, any &DT) {{\n{}\n}};\n", name, code);
        }
        if (!(code = def.predicate_code).empty()) {
            code = code.substr(1, code.size() - 2);
            if (code.find("LOC") != std::string::npos)
                code = loc_preamble + code;
            if (code.find("CTX") != std::string::npos)
                code = const_action_preamble + code;
            actions += collie::format("P[\"{}\"].predicate = pred_{};\n", name, name);
            actionFns += collie::format("auto pred_{}(const peg::SemanticValues &VS, const any "
                                        "&DT, std::string &MSG) {{\n{}\n}};\n",
                                        name, code);
        }
    };

    FILE *fout = fopen(output_file.c_str(), "w");
    collie::print(fout, "// clang-format off\n");
    collie::print(fout, "#pragma clang diagnostic push\n");
    collie::print(fout, "#pragma clang diagnostic ignored \"-Wreturn-type\"\n");
    if (!preamble.empty())
        collie::print(fout, "{}\n", preamble.substr(1, preamble.size() - 2));
    string rules_preamble = "  using namespace peg;\n"
                            "  using peg::seq;\n"
                            "  using vc = vector<pair<char32_t, char32_t>>;\n";
    collie::print(fout, "void init_{}_rules(peg::Grammar &P) {{\n{}\n{}\n}}\n", ns,
                  rules_preamble, rules);
    collie::print(fout, "{}\n", actionFns);
    collie::print(fout, "void init_{}_actions(peg::Grammar &P) {{\n  {}\n}}\n", ns,
                  actions);
    collie::print(fout, "// clang-format on\n");
    collie::print(fout, "#pragma clang diagnostic pop\n");
    fclose(fout);
}

int main(int argc, char **argv) {
    collie::App app("peg2cc", "Converts a PEG grammar to C++ code");
    auto *sub = app.add_subcommand("generate", "Generate C++ code from a PEG grammar");
    sub->add_option("-g,--grammar", grammar_file, "The PEG grammar file")->required(true);
    sub->add_option("-o,--output", output_file, "The output C++ file")->required(true);
    sub->add_option("-n,--namespace", ns, "The namespace for the generated code")->required(true);
    app.add_subcommand("version", "Print the version of the program")->callback([]() {
        collie::print("peg2cc {}\n", HERCULES_VERSION);
    });
    sub->callback([]() { generate(); });
    app.require_subcommand();
    COLLIE_CLI_PARSE(app, argc, argv);
    return 0;
}
