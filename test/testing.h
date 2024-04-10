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
#pragma once


#include <algorithm>
#include <cstdio>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <gc.h>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <tuple>
#include <unistd.h>
#include <vector>

#include <hercules/hir/analyze/dataflow/capture.h>
#include <hercules/hir/analyze/dataflow/reaching.h>
#include "hercules/hir/util/inlining.h"
#include <hercules/hir/util/irtools.h>
#include <hercules/hir/util/operator.h>
#include <hercules/hir/util/outlining.h>
#include <hercules/compiler/compiler.h>
#include "hercules/compiler/error.h"
#include <hercules/parser/common.h>
#include <hercules/util/common.h>

#include "gtest/gtest.h"

using namespace hercules;
using namespace std;

class TestOutliner : public ir::transform::OperatorPass {
    int successes = 0;
    int failures = 0;
    ir::ReturnInstr *successesReturn = nullptr;
    ir::ReturnInstr *failuresReturn = nullptr;

    const std::string KEY = "test-outliner-pass";

    std::string getKey() const override { return KEY; }

    void handle(ir::SeriesFlow *v) override;

    void handle(ir::ReturnInstr *v) override;
};

class TestInliner : public ir::transform::OperatorPass {
    const std::string KEY = "test-inliner-pass";

    std::string getKey() const override { return KEY; }

    void handle(ir::CallInstr *v) override;
};

struct PartitionArgsByEscape : public ir::util::Operator {
    std::vector<ir::analyze::dataflow::CaptureInfo> expected;
    std::vector<ir::Value *> calls;

    void handle(ir::CallInstr *v) override;
};

struct EscapeValidator : public ir::transform::Pass {
    const std::string KEY = "test-escape-validator-pass";

    std::string getKey() const override { return KEY; }

    std::string capAnalysisKey;

    explicit EscapeValidator(const std::string &capAnalysisKey)
            : ir::transform::Pass(), capAnalysisKey(capAnalysisKey) {}

    void run(ir::Module *m) override;
};

vector<string> splitLines(const string &output);

pair<bool, string> findExpectOnLine(const string &line);

pair<vector<string>, bool> findExpects(const string &filename, bool isCode);

extern string argv0;

class HerculesTest : public testing::TestWithParam<
        tuple<string /*filename*/, bool /*debug*/, string /* case name */,
                string /* case code */, int /* case line */,
                bool /* barebones stdlib */, bool /* Python numerics */>> {
    vector<char> buf;
    int out_pipe[2];
    pid_t pid;

public:
    HerculesTest() : buf(65536), out_pipe(), pid() {}

    string getFilename(const string &basename) {
        return string(TEST_DIR) + "/" + basename;
    }

    int runInChildProcess() {
        assert(pipe(out_pipe) != -1);
        pid = fork();
        GC_atfork_prepare();
        assert(pid != -1);

        if (pid == 0) {
            GC_atfork_child();
            dup2(out_pipe[1], STDOUT_FILENO);
            close(out_pipe[0]);
            close(out_pipe[1]);

            auto file = getFilename(get<0>(GetParam()));
            bool debug = get<1>(GetParam());
            auto code = get<3>(GetParam());
            auto startLine = get<4>(GetParam());
            int testFlags = 1 + get<5>(GetParam());
            bool pyNumerics = get<6>(GetParam());

            auto compiler = std::make_unique<Compiler>(
                    argv0, debug, /*disabledPasses=*/std::vector<std::string>{}, /*isTest=*/true,
                    pyNumerics);
            compiler->getLLVMVisitor()->setStandalone(
                    true); // make sure we abort() on runtime error
            llvm::handleAllErrors(code.empty()
                                  ? compiler->parseFile(file, testFlags)
                                  : compiler->parseCode(file, code, startLine, testFlags),
                                  [](const error::ParserErrorInfo &e) {
                                      for (auto &group: e) {
                                          for (auto &msg: group) {
                                              getLogger().level = 0;
                                              printf("%s\n", msg.getMessage().c_str());
                                          }
                                      }
                                      fflush(stdout);
                                      exit(EXIT_FAILURE);
                                  });

            auto *pm = compiler->getPassManager();
            pm->registerPass(std::make_unique<TestOutliner>());
            pm->registerPass(std::make_unique<TestInliner>());
            auto capKey =
                    pm->registerAnalysis(std::make_unique<ir::analyze::dataflow::CaptureAnalysis>(
                                                 ir::analyze::dataflow::RDAnalysis::KEY,
                                                 ir::analyze::dataflow::DominatorAnalysis::KEY),
                                         {ir::analyze::dataflow::RDAnalysis::KEY,
                                          ir::analyze::dataflow::DominatorAnalysis::KEY});
            pm->registerPass(std::make_unique<EscapeValidator>(capKey), /*insertBefore=*/"",
                             {capKey});

            llvm::cantFail(compiler->compile());
            compiler->getLLVMVisitor()->run({file});
            fflush(stdout);
            exit(EXIT_SUCCESS);
        } else {
            GC_atfork_parent();
            int status = -1;
            close(out_pipe[1]);
            assert(waitpid(pid, &status, 0) == pid);
            auto r = read(out_pipe[0], buf.data(), buf.size() - 1);
            (void) r;
            close(out_pipe[0]);
            return status;
        }
        return -1;
    }

    string result() { return string(buf.data()); }
};

string getTestNameFromParam(const testing::TestParamInfo<HerculesTest::ParamType> &info);

string getTypeTestNameFromParam(const testing::TestParamInfo<HerculesTest::ParamType> &info);
/*
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
*/
vector<tuple<string, bool, string, string, int, bool, bool>> getTypeTests(const vector<string> &files);