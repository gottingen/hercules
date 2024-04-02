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

#include <hercules/app/version.h>
#include <collie/table/table.h>
#include <collie/table/markdown_exporter.h>
#include <collie/table/asciidoc_exporter.h>
#include <collie/table/latex_exporter.h>
#include <hercules/config/config.h>
#include <collie/strings/format.h>
#include <map>

namespace hercules {

    std::vector<std::pair<std::string, std::string>> hercules_ref = {
            {"backtrace", "master"},
            {"bdwgc",     "8.0.5"},
            {"bz2",       "1.0.8"},
            {"llvm",      "17.0.6"},
            {"clang",     "17.0.6"},
            {"collie",    "0.2.7"},
            {"googletest","release-1.12.1"},
            {"re2",       "2022-06-01"},
            {"xz",      "5.2.5"},
            {"zlibng",  "2.0.5"}
    };

    enum ExportFormat {
        None, Markdown, Latex, Ascii
    };
    bool color_enabled = true;
    std::string out = "stdout";
    ExportFormat format = ExportFormat::None;

    void set_up_version_cmd(collie::App *app) {
        app->add_option("-c, --color", color_enabled, "Enable color output")->default_str("true");
        app->add_option("-o, --output", out, "Output file")->default_str("stdout");
        std::map<std::string, ExportFormat> format_map = {
                {"none",     ExportFormat::None},
                {"markdown", ExportFormat::Markdown},
                {"latex",    ExportFormat::Latex},
                {"ascii",    ExportFormat::Ascii}
        };
        app->add_option("-f, --format", format, "Output format")->default_str("none")->transform(
                collie::CheckedTransformer(format_map, collie::ignore_case));
        app->callback([]() {
            version_dump();
        });
    }

    static void dump_reference(collie::table::Table &tb);
    void version_dump() {
        collie::table::Table tb;
        tb.add_row({"Hercules", HERCULES_VERSION});
        tb.add_row({"Build Type", HERCULES_BUILD_TYPE});
        tb.add_row({"Compiler", HERCULES_COMPILER_ID});
        tb.add_row({"Compiler Version",
                    collie::format("{}.{}", HERCULES_COMPILER_VERSION_MAJOR, HERCULES_COMPILER_VERSION_MINOR)});
        tb.add_row({"c++ Standard", HERCULES_CXX_STANDARD});
        tb.add_row({"cxx abi", HERCULES_CXX11_ABI});
        dump_reference(tb);
        tb.column(0).format().width(20);
        tb.column(1).format().width(20);
        if (color_enabled) {
            tb.column(0).format().font_style({collie::table::FontStyle::bold});
            tb.column(0).format().font_color(collie::table::Color::yellow);
            tb.column(1).format().font_style({collie::table::FontStyle::bold});
            tb.column(1).format().font_color(collie::table::Color::green);
        }
        std::string table_str;
        if (format == ExportFormat::Markdown) {
            collie::table::MarkdownExporter exporter;
            table_str = exporter.dump(tb);
        } else if (format == ExportFormat::Ascii) {
            collie::table::AsciiDocExporter exporter;
            table_str = exporter.dump(tb);
        } else if (format == ExportFormat::Latex) {
            collie::table::LatexExporter exporter;
            table_str = exporter.dump(tb);
        } else {
            table_str = tb.str();
        }
        if (out == "stdout") {
            std::cout << tb << std::endl;
        } else {
            std::ofstream out_stream(out, std::ios::out|std::ios::trunc|std::ios::binary);
            out_stream << table_str<<std::endl;
        }
    }

    void dump_reference(collie::table::Table &tb) {
        for (auto &it: hercules_ref) {
            tb.add_row({it.first, it.second});
        }
    }

}  // namespace hercules