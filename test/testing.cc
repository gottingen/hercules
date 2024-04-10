// Copyright 2023 The Elastic-AI Authors.
// part of Elastic AI Search
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
#include "testing.h"


string argv0;

void TestOutliner::handle(ir::SeriesFlow *v) {
    auto *M = v->getModule();
    auto begin = v->begin(), end = v->end();
    bool sawBegin = false, sawEnd = false;
    for (auto it = v->begin(); it != v->end(); ++it) {
        if (ir::util::isCallOf(*it, "__outline_begin__") && !sawBegin) {
            begin = it;
            sawBegin = true;
        } else if (ir::util::isCallOf(*it, "__outline_end__") && !sawEnd) {
            end = it;
            sawEnd = true;
        }
    }
    if (sawBegin && sawEnd) {
        auto result = ir::util::outlineRegion(ir::cast<ir::BodiedFunc>(getParentFunc()),
                                              v, begin, end);
        ++(result ? successes : failures);
        if (successesReturn)
            successesReturn->setValue(M->getInt(successes));
        if (failuresReturn)
            failuresReturn->setValue(M->getInt(failures));
    }
}

void TestOutliner::handle(ir::ReturnInstr *v) {
    auto *M = v->getModule();
    if (getParentFunc()->getUnmangledName() == "__outline_successes__") {
        v->setValue(M->getInt(successes));
        successesReturn = v;
    }
    if (getParentFunc()->getUnmangledName() == "__outline_failures__") {
        v->setValue(M->getInt(failures));
        failuresReturn = v;
    }
}

void TestInliner::handle(ir::CallInstr *v) {
    auto *M = v->getModule();
    auto *f = ir::cast<ir::BodiedFunc>(ir::util::getFunc(v->getCallee()));
    auto *neg = M->getOrRealizeMethod(M->getIntType(), ir::Module::NEG_MAGIC_NAME,
                                      {M->getIntType()});
    if (!f)
        return;
    auto name = f->getUnmangledName();
    if (name.find("inline_me") != std::string::npos) {
        auto aggressive = name.find("aggressive") != std::string::npos;
        auto res = ir::util::inlineCall(v, aggressive);
        if (!res)
            return;
        for (auto *var: res.newVars)
            ir::cast<ir::BodiedFunc>(getParentFunc())->push_back(var);
        v->replaceAll(ir::util::call(neg, {res.result}));
    }
}

void PartitionArgsByEscape::handle(ir::CallInstr *v) {
    using namespace hercules::ir;
    if (auto *f = cast<Func>(util::getFunc(v->getCallee()))) {
        if (f->getUnmangledName() == "expect_capture") {
            // Format is:
            //   - Return captures (bool)
            //   - Extern captures (bool)
            //   - Captured arg indices (int tuple)
            std::vector<Value *> args(v->begin(), v->end());
            seqassertn(args.size() == 3, "bad escape-test call (size)");
            seqassertn(isA<BoolConst>(args[0]) && isA<BoolConst>(args[1]),
                       "bad escape-test call (arg types)");

            ir::analyze::dataflow::CaptureInfo info;
            info.returnCaptures = cast<BoolConst>(args[0])->getVal();
            info.externCaptures = cast<BoolConst>(args[1])->getVal();
            auto *tuple = cast<CallInstr>(args[2]);
            seqassertn(tuple,
                       "last escape-test call argument should be a const tuple literal");

            for (auto *arg: *tuple) {
                seqassertn(isA<IntConst>(arg), "final args should be int");
                info.argCaptures.push_back(cast<IntConst>(arg)->getVal());
            }

            expected.push_back(info);
            calls.push_back(v);
        }
    }
}

void EscapeValidator::run(ir::Module *m) {
    using namespace hercules::ir;
    auto *capResult =
            getAnalysisResult<ir::analyze::dataflow::CaptureResult>(capAnalysisKey);
    for (auto *var: *m) {
        if (auto *f = cast<Func>(var)) {
            PartitionArgsByEscape pabe;
            f->accept(pabe);
            auto expected = pabe.expected;
            if (expected.empty())
                continue;

            auto it = capResult->results.find(f->getId());
            seqassertn(it != capResult->results.end(),
                       "function not found in capture results");
            auto received = it->second;
            seqassertn(expected.size() == received.size(),
                       "size mismatch in capture results");

            for (unsigned i = 0; i < expected.size(); i++) {
                auto exp = expected[i];
                auto got = received[i];
                std::sort(exp.argCaptures.begin(), exp.argCaptures.end());
                std::sort(got.argCaptures.begin(), got.argCaptures.end());

                bool good = (exp.returnCaptures == got.returnCaptures) &&
                            (exp.externCaptures == got.externCaptures) &&
                            (exp.argCaptures == got.argCaptures);
                pabe.calls[i]->replaceAll(m->getBool(good));
            }
        }
    }
}

vector<string> splitLines(const string &output) {
    vector<string> result;
    string line;
    istringstream stream(output);
    const char delim = '\n';

    while (getline(stream, line, delim))
        result.push_back(line);

    return result;
}

pair<bool, string> findExpectOnLine(const string &line) {
    for (auto EXPECT_STR: vector<pair<bool, string>>{
            {false, "# EXPECT: "},
            {false, "#: "},
            {true,  "#! "}}) {
        size_t pos = line.find(EXPECT_STR.second);
        if (pos != string::npos)
            return {EXPECT_STR.first, line.substr(pos + EXPECT_STR.second.length())};
    }
    return {false, ""};
}


pair<vector<string>, bool> findExpects(const string &filename, bool isCode) {
    vector<string> result;
    bool isError = false;
    string line;
    if (!isCode) {
        ifstream file(filename);
        if (!file.good()) {
            cerr << "error: could not open " << filename << endl;
            exit(EXIT_FAILURE);
        }

        while (getline(file, line)) {
            auto expect = findExpectOnLine(line);
            if (!expect.second.empty()) {
                result.push_back(expect.second);
                isError |= expect.first;
            }
        }
        file.close();
    } else {
        istringstream file(filename);
        while (getline(file, line)) {
            auto expect = findExpectOnLine(line);
            if (!expect.second.empty()) {
                result.push_back(expect.second);
                isError |= expect.first;
            }
        }
    }
    return {result, isError};
}

string getTestNameFromParam(const testing::TestParamInfo<HerculesTest::ParamType> &info) {
    const string basename = get<0>(info.param);
    const bool debug = get<1>(info.param);

    // normalize basename
    // size_t found1 = basename.find('/');
    // size_t found2 = basename.find('.');
    // assert(found1 != string::npos);
    // assert(found2 != string::npos);
    // assert(found2 > found1);
    // string normname = basename.substr(found1 + 1, found2 - found1 - 1);
    string normname = basename;
    replace(normname.begin(), normname.end(), '/', '_');
    replace(normname.begin(), normname.end(), '.', '_');
    return normname + (debug ? "_debug" : "");
}

string getTypeTestNameFromParam(const testing::TestParamInfo<HerculesTest::ParamType> &info) {
    return getTestNameFromParam(info) + "_" + get<2>(info.param);
}


vector<tuple<string, bool, string, string, int, bool, bool>> getTypeTests(const vector<string> &files) {
    vector<tuple<string, bool, string, string, int, bool, bool>> cases;
    for (auto &f: files) {
        bool barebones = false;
        string l;
        ifstream fin(string(TEST_DIR) + "/" + f);
        string code, testName;
        int test = 0;
        int codeLine = 0;
        int line = 0;
        while (getline(fin, l)) {
            if (l.substr(0, 3) == "#%%") {
                if (line)
                    cases.emplace_back(make_tuple(f, true, to_string(line) + "_" + testName, code,
                                                  codeLine, barebones, false));
                auto t = ast::split(l.substr(4), ',');
                barebones = (t.size() > 1 && t[1] == "barebones");
                testName = t[0];
                code = l + "\n";
                codeLine = line;
                test++;
            } else {
                code += l + "\n";
            }
            line++;
        }
        if (line)
            cases.emplace_back(make_tuple(f, true, to_string(line) + "_" + testName, code,
                                          codeLine, barebones, false));
    }
    return cases;
}

TEST_P(HerculesTest, Run) {
    const string file = get<0>(GetParam());
    int status;
    bool isCase = !get<2>(GetParam()).empty();
    if (!isCase)
        status = runInChildProcess();
    else
        status = runInChildProcess();
    ASSERT_TRUE(WIFEXITED(status));

    string output = result();

    auto expects = findExpects(!isCase ? getFilename(file) : get<3>(GetParam()), isCase);
    if (WEXITSTATUS(status) != int(expects.second))
        fprintf(stderr, "%s\n", output.c_str());
    ASSERT_EQ(WEXITSTATUS(status), int(expects.second));
    const bool assertsFailed = output.find("TEST FAILED") != string::npos;
    EXPECT_FALSE(assertsFailed);
    if (assertsFailed)
        std::cerr << output << std::endl;

    if (!expects.first.empty()) {
        vector<string> results = splitLines(output);
        for (unsigned i = 0; i < min(results.size(), expects.first.size()); i++)
            if (expects.second)
                EXPECT_EQ(results[i], expects.first[i]);
            else
                EXPECT_EQ(results[i], expects.first[i]);
        EXPECT_EQ(results.size(), expects.first.size());
    }
}