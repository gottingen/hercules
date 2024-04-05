// Copyright 2024 The titan-search Authors.
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

#include <hercules/ast/cc/cpp_preprocessor.h>
#include <hercules/ast/cc/libclang_parser.h>

namespace hercules::ccast {
    namespace detail {
        struct pp_macro {
            std::unique_ptr<cpp_macro_definition> macro;
            unsigned line;
        };

        struct pp_include {
            std::string file_name, full_path;
            cpp_include_kind kind;
            unsigned line;
        };

        struct pp_doc_comment {
            std::string comment;
            unsigned line;
            enum {
                c,
                cpp,
                end_of_line,
            } kind;

            bool matches(const cpp_entity &e, unsigned line);
        };

        struct preprocessor_output {
            std::string source;
            std::vector<pp_include> includes;
            std::vector<pp_macro> macros;
            std::vector<pp_doc_comment> comments;
        };

        preprocessor_output preprocess(const libclang_compile_config &config, const char *path,
                                       const diagnostic_logger &logger);
    } // namespace detail
} // namespace hercules::ccast
