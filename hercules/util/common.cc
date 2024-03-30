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

#include <hercules/util/common.h>
#include <llvm/Support/Path.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace hercules {
    namespace {
        void compilationMessage(const std::string &header, const std::string &msg,
                                const std::string &file, int line, int col, int len,
                                int errorCode, MessageGroupPos pos) {
            auto &out = getLogger().err;
            seqassertn(!(file.empty() && (line > 0 || col > 0)),
                       "empty filename with non-zero line/col: file={}, line={}, col={}", file,
                       line, col);
            seqassertn(!(col > 0 && line <= 0), "col but no line: file={}, line={}, col={}", file,
                       line, col);

            switch (pos) {
                case MessageGroupPos::NONE:
                    break;
                case MessageGroupPos::HEAD:
                    break;
                case MessageGroupPos::MID:
                    collie::print("├─ ");
                    break;
                case MessageGroupPos::LAST:
                    collie::print("╰─ ");
                    break;
            }

            collie::print(out, "\033[1m");
            if (!file.empty()) {
                auto f = file.substr(file.rfind('/') + 1);
                collie::print(out, "{}", f == "-" ? "<stdin>" : f);
            }
            if (line > 0)
                collie::print(out, ":{}", line);
            if (col > 0)
                collie::print(out, ":{}", col);
            if (len > 0)
                collie::print(out, "-{}", col + len);
            if (!file.empty())
                collie::print(out, ": ");
            collie::print(out, "{}\033[1m {}\033[0m{}\n", header, msg,
                       errorCode != -1
                       ? collie::format(" (see https://github.com/gottingen/hercules/error/{:04d})", errorCode)
                       : "");
        }

        std::vector<Logger> loggers;
    } // namespace

    std::ostream &operator<<(std::ostream &out, const hercules::SrcInfo &src) {
        out << llvm::sys::path::filename(src.file).str() << ":" << src.line << ":" << src.col;
        return out;
    }

    void compilationError(const std::string &msg, const std::string &file, int line,
                          int col, int len, int errorCode, bool terminate,
                          MessageGroupPos pos) {
        compilationMessage("\033[1;31merror:\033[0m", msg, file, line, col, len, errorCode,
                           pos);
        if (terminate)
            exit(EXIT_FAILURE);
    }

    void compilationWarning(const std::string &msg, const std::string &file, int line,
                            int col, int len, int errorCode, bool terminate,
                            MessageGroupPos pos) {
        compilationMessage("\033[1;33mwarning:\033[0m", msg, file, line, col, len, errorCode,
                           pos);
        if (terminate)
            exit(EXIT_FAILURE);
    }

    void Logger::parse(const std::string &s) {
        flags |= s.find('t') != std::string::npos ? FLAG_TIME : 0;
        flags |= s.find('r') != std::string::npos ? FLAG_REALIZE : 0;
        flags |= s.find('T') != std::string::npos ? FLAG_TYPECHECK : 0;
        flags |= s.find('i') != std::string::npos ? FLAG_IR : 0;
        flags |= s.find('l') != std::string::npos ? FLAG_USER : 0;
    }
} // namespace hercules

hercules::Logger &hercules::getLogger() {
    if (loggers.empty())
        loggers.emplace_back();
    return loggers.back();
}

void hercules::pushLogger() { loggers.emplace_back(); }

bool hercules::popLogger() {
    if (loggers.empty())
        return false;
    loggers.pop_back();
    return true;
}

void hercules::assertionFailure(const char *expr_str, const char *file, int line,
                                const std::string &msg) {
    auto &out = getLogger().err;
    out << "Assert failed:\t" << msg << "\n"
        << "Expression:\t" << expr_str << "\n"
        << "Source:\t\t" << file << ":" << line << "\n";
    abort();
}
