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
    struct ThirdPartyEntry {
        std::string name;
        std::string version;
        std::string url;
    };
    std::vector<ThirdPartyEntry> hercules_ref = {
            {"backtrace", "master", "https://github.com/ianlancetaylor/libbacktrace"},
            {"bdwgc",     "8.0.5", "https://github.com/ivmai/bdwgc"},
            {"bz2",       "1.0.8", "https://www.sourceware.org/pub/bzip2/bzip2-1.0.8.tar.gz"},
            {"llvm",      "17.0.6", "https://github.com/llvm/llvm-project"},
            {"clang",     "17.0.6", "https://github.com/llvm/llvm-project"},
            {"collie",    "0.2.7", "https://github.com/gottingen/collie"},
            {"googletest","release-1.12.1", "https://github.com/google/googletest"},
            {"re2",       "2022-06-01","https://github.com/google/re2"},
            {"xz",      "5.2.5", "https://github.com/xz-mirror/xz"},
            {"zlibng",  "2.0.5", "https://github.com/zlib-ng/zlib-ng"}
    };

    enum ExportFormat {
        None, Markdown, Latex, Ascii
    };

    void show_full_version(collie::table::Table &tb);
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
        show_full_version(tb);
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
        tb.add_row({"project", "version", "URL"});
        tb[0].format().font_style({collie::table::FontStyle::bold})
                .font_color(collie::table::Color::yellow)
                .font_align(collie::table::FontAlign::center);
        for (auto &it: hercules_ref) {
            tb.add_row({it.name, it.version, it.url});
        }
    }

    void show_full_version(collie::table::Table &readme) {
        using namespace collie::table;
        using Row_t = Table::Row_t;

        readme.format().border_color(Color::yellow);

        readme.add_row(Row_t{"hercules is a AOT compiler for python"});
        readme[0].format().font_align(FontAlign::center).font_color(Color::yellow);

        readme.add_row(Row_t{"https://github.com/gottingen/hercules"});
        readme[1]
                .format()
                .font_align(FontAlign::center)
                .font_style({FontStyle::underline, FontStyle::italic})
                .font_color(Color::white)
                .hide_border_top();

        readme.add_row(Row_t{"hercules is an AOT compiler for python that compiles python code to native code which can be called from C++"});
        readme[2].format().font_style({FontStyle::italic}).font_color(Color::magenta);

        Table highlights;
        highlights.add_row(Row_t{"AOT compiler", "Requires C++17", "Apache 2 License"});
        readme.add_row(Row_t{highlights});
        readme[3].format().font_align(FontAlign::center).hide_border_top();
        Table author;
        author.add_row(Row_t{"Author: Jeff", "Email: lijippy@163.com"});
        author[0].format().hide_border().font_align(FontAlign::center).width(40);
        author.format().hide_border();
        Table empty_row;
        empty_row.format().hide_border();
        readme.add_row(Row_t{author});
        readme[4].format()
        .font_align(FontAlign::center)
        .font_color(Color::green)
        .font_style({FontStyle::underline});

        readme.add_row(Row_t{"Hercules runtime information"});
        readme[5].format().font_align(FontAlign::center);

        Table format;
        format.add_row(Row_t{"Hercules Information", "Compiler Information", "extra options"});
        format[0].format().font_align(FontAlign::center);
        format[0][0].format().font_color(Color::green).column_separator(":");

        format.column(1).format().width(40).font_align(FontAlign::center);
        format.column(2).format().width(40).font_align(FontAlign::center);
        auto h_info = collie::format("version: {} \nbuild type: {}", HERCULES_VERSION, HERCULES_BUILD_TYPE);
        auto c_info = collie::format("compiler: {} \ncompiler version: {}.{}", HERCULES_COMPILER_ID,
                                     HERCULES_COMPILER_VERSION_MAJOR, HERCULES_COMPILER_VERSION_MINOR);
        auto e_info = collie::format("cxx standard: {} \ncxx abi: {}", HERCULES_CXX_STANDARD, HERCULES_CXX11_ABI);
        format.add_row({h_info,
                        c_info,
                        e_info});
        format[1][0].format().font_align(FontAlign::center);
        format[1][1].format().font_align(FontAlign::center);
        format[1][2].format().font_align(FontAlign::center);

        format.column(0).format().width(23);
        format.column(1).format().border_left(":");

        readme.add_row(Row_t{format});

        readme[5]
                .format()
                .border_color(Color::green)
                .font_color(Color::cyan)
                .font_style({FontStyle::underline})
                .padding_top(0)
                .padding_bottom(0);
        readme[6].format().hide_border_top().padding_top(0);

        readme.add_row(Row_t{empty_row});
        readme[7].format().hide_border_left().hide_border_right();
        readme.add_row(Row_t{"Acknowledgements Third party libraries"});
        readme[8].format().font_align(FontAlign::center).font_style({FontStyle::underline}).font_color(Color::cyan);
        Table third_party;
        dump_reference(third_party);
        readme.add_row(Row_t{third_party});
        readme[9].format().hide_border_top().border_color(Color::white).font_color(Color::green).font_align(FontAlign::center);
        readme.add_row(Row_t{"🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥"});
        readme[10]
                .format()
                .font_align(FontAlign::center)
                .hide_border_top()
                .multi_byte_characters(true);
    }

}  // namespace hercules