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


#include <iostream>
#include <collie/cli/cli.h>

#include <hercules/ast/cc/code_generator.h>
#include <hercules/ast/cc/cpp_entity_kind.h>
#include <hercules/ast/cc/cpp_forward_declarable.h>
#include <hercules/ast/cc/cpp_namespace.h>
#include <hercules/ast/cc/libclang_parser.h>
#include <hercules/ast/cc/visitor.h>
#include <hercules/app/ast/ast_context.h>
#include <hercules/config/config.h>
#include <collie/table/table.h>
#include <hercules/app/ast/cmd_parse.h>

static void dump_version() {
    collie::table::Table table;
    table.add_row({"ccast version", HERCULES_VERSION});
    table[0][1].format().font_color(collie::table::Color::green);
    table.add_row({"Using libclang version", "17"});
    table[1][1].format().font_color(collie::table::Color::green);
    std::cout << table << std::endl;
}

void setup_version_command(collie::App &command) {
    auto *vn = command.add_subcommand("version", "display version information and exit");
    vn->callback([]() {
        dump_version();
    });
}
// prints the AST entry of a cpp_entity (base class for all entities),
// will only print a single line
void print_entity(std::ostream& out, const hercules::ccast::cpp_entity& e)
{
    // print name and the kind of the entity
    if (!e.name().empty())
        out << e.name();
    else
        out << "<anonymous>";
    out << " (" << hercules::ccast::to_string(e.kind()) << ")";

    // print whether or not it is a definition
    if (hercules::ccast::is_definition(e))
        out << " [definition]";

    // print number of attributes
    if (!e.attributes().empty())
        out << " [" << e.attributes().size() << " attribute(s)]";

    if (e.kind() == hercules::ccast::cpp_entity_kind::language_linkage_t)
        // no need to print additional information for language linkages
        out << '\n';
    else if (e.kind() == hercules::ccast::cpp_entity_kind::namespace_t)
    {
        // cast to cpp_namespace
        auto& ns = static_cast<const hercules::ccast::cpp_namespace&>(e);
        // print whether or not it is inline
        if (ns.is_inline())
            out << " [inline]";
        out << '\n';
    }
    else
    {
        // print the declaration of the entity
        // it will only use a single line
        // derive from code_generator and implement various callbacks for printing
        // it will print into a std::string
        class code_generator : public hercules::ccast::code_generator
        {
            std::string str_;                 // the result
            bool        was_newline_ = false; // whether or not the last token was a newline
            // needed for lazily printing them

        public:
            code_generator(const hercules::ccast::cpp_entity& e)
            {
                // kickoff code generation here
                hercules::ccast::generate_code(*this, e);
            }

            // return the result
            const std::string& str() const noexcept
            {
                return str_;
            }

        private:
            // called to retrieve the generation options of an entity
            generation_options do_get_options(const hercules::ccast::cpp_entity&,
                                              hercules::ccast::cpp_access_specifier_kind) override
            {
                // generate declaration only
                return code_generator::declaration;
            }

            // no need to handle indentation, as only a single line is used
            void do_indent() override {}
            void do_unindent() override {}

            // called when a generic token sequence should be generated
            // there are specialized callbacks for various token kinds,
            // to e.g. implement syntax highlighting
            void do_write_token_seq(hercules::ccast::string_view tokens) override
            {
                if (was_newline_)
                {
                    // lazily append newline as space
                    str_ += ' ';
                    was_newline_ = false;
                }
                // append tokens
                str_ += tokens.c_str();
            }

            // called when a newline should be generated
            // we're lazy as it will always generate a trailing newline,
            // we don't want
            void do_write_newline() override
            {
                was_newline_ = true;
            }

        } generator(e);
        // print generated code
        out << ": `" << generator.str() << '`' << '\n';
    }
}

// prints the AST of a file
void print_ast(std::ostream& out, const hercules::ccast::cpp_file& file)
{
    // print file name
    out << "AST for '" << file.name() << "':\n";
    std::string prefix; // the current prefix string
    // recursively visit file and all children
    hercules::ccast::visit(file, [&](const hercules::ccast::cpp_entity& e, hercules::ccast::visitor_info info) {
        if (e.kind() == hercules::ccast::cpp_entity_kind::file_t || hercules::ccast::is_templated(e)
            || hercules::ccast::is_friended(e))
            // no need to do anything for a file,
            // templated and friended entities are just proxies, so skip those as well
            // return true to continue visit for children
            return true;
        else if (info.event == hercules::ccast::visitor_info::container_entity_exit)
        {
            // we have visited all children of a container,
            // remove prefix
            prefix.pop_back();
            prefix.pop_back();
        }
        else
        {
            out << prefix; // print prefix for previous entities
            // calculate next prefix
            if (info.last_child)
            {
                if (info.event == hercules::ccast::visitor_info::container_entity_enter)
                    prefix += "  ";
                out << "+-";
            }
            else
            {
                if (info.event == hercules::ccast::visitor_info::container_entity_enter)
                    prefix += "| ";
                out << "|-";
            }

            print_entity(out, e);
        }

        return true;
    });
}

// parse a file
std::unique_ptr<hercules::ccast::cpp_file> parse_file(const hercules::ccast::libclang_compile_config& config,
                                             const hercules::ccast::diagnostic_logger&       logger,
                                             const std::string& filename, bool fatal_error)
{
    // the entity index is used to resolve cross references in the AST
    // we don't need that, so it will not be needed afterwards
    hercules::ccast::cpp_entity_index idx;
    // the parser is used to parse the entity
    // there can be multiple parser implementations
    hercules::ccast::libclang_parser parser(collie::ts::ref(logger));
    // parse the file
    auto file = parser.parse(idx, filename, config);
    if (fatal_error && parser.error())
        return nullptr;
    return file;
}

int main(int argc, char* argv[]) {
    collie::App app("ccast - The commandline interface to the ccast library.\n");
    app.set_help_all_flag("--help-all", "Expand all help");
    auto &ins = hercules::AstContext::instance();
    app.add_flag("-v, --verbose", ins.verbose, "be verbose when parsing");
    app.add_flag("-F, --fatal_errors", ins.fatal_error, "abort program when a parser error occurs, instead of doing error correction");
    // setup the version command
    setup_version_command(app);
    // setup the parse command
    hercules::setup_parse_command(app);
    // parse the command line
    app.require_subcommand();  // 1 or more

    COLLIE_CLI_PARSE(app, argc, argv);
}
/*
int main(int argc, char* argv[])
try
{
    cxxopts::Options option_list("cppast",
                                 "cppast - The commandline interface to the cppast library.\n");
    // clang-format off
    option_list.add_options()
            ("h,help", "display this help and exit")
            ("version", "display version information and exit")
            ("v,verbose", "be verbose when parsing")
            ("fatal_errors", "abort program when a parser error occurs, instead of doing error correction")
            ("file", "the file that is being parsed (last positional argument)",
             cxxopts::value<std::string>());
    option_list.add_options("compilation")
            ("database_dir", "set the directory where a 'compile_commands.json' file is located containing build information",
             cxxopts::value<std::string>())
            ("database_file", "set the file name whose configuration will be used regardless of the current file name",
             cxxopts::value<std::string>())
            ("std", "set the C++ standard (c++98, c++03, c++11, c++14, c++1z (experimental), c++17, c++2a, c++20)",
             cxxopts::value<std::string>()->default_value(cppast::to_string(cppast::cpp_standard::cpp_latest)))
            ("I,include_directory", "add directory to include search path",
             cxxopts::value<std::vector<std::string>>())
            ("D,macro_definition", "define a macro on the command line",
             cxxopts::value<std::vector<std::string>>())
            ("U,macro_undefinition", "undefine a macro on the command line",
             cxxopts::value<std::vector<std::string>>())
            ("f,feature", "enable a custom feature (-fXX flag)",
             cxxopts::value<std::vector<std::string>>())
            ("gnu_extensions", "enable GNU extensions (equivalent to -std=gnu++XX)")
            ("msvc_extensions", "enable MSVC extensions (equivalent to -fms-extensions)")
            ("msvc_compatibility", "enable MSVC compatibility (equivalent to -fms-compatibility)")
            ("fast_preprocessing", "enable fast preprocessing, be careful, this breaks if you e.g. redefine macros in the same file!")
            ("remove_comments_in_macro", "whether or not comments generated by macro are kept, enable if you run into errors");
    // clang-format on
    option_list.parse_positional("file");

    auto options = option_list.parse(argc, argv);
    if (options.count("help"))
        print_help(option_list);
    else if (options.count("version"))
    {
        std::cout << "cppast version " << CPPAST_VERSION_STRING << "\n";
        std::cout << '\n';
        std::cout << "Using libclang version " << CPPAST_CLANG_VERSION_STRING << '\n';
    }
    else if (!options.count("file") || options["file"].as<std::string>().empty())
    {
        print_error("missing file argument");
        return 1;
    }
    else
    {
        // the compile config stores compilation flags
        cppast::libclang_compile_config config;
        if (options.count("database_dir"))
        {
            cppast::libclang_compilation_database database(
                    options["database_dir"].as<std::string>());
            if (options.count("database_file"))
                config
                        = cppast::libclang_compile_config(database,
                                                          options["database_file"].as<std::string>());
            else
                config
                        = cppast::libclang_compile_config(database, options["file"].as<std::string>());
        }

        if (options.count("verbose"))
            config.write_preprocessed(true);

        if (options.count("fast_preprocessing"))
            config.fast_preprocessing(true);

        if (options.count("remove_comments_in_macro"))
            config.remove_comments_in_macro(true);

        if (options.count("include_directory"))
            for (auto& include : options["include_directory"].as<std::vector<std::string>>())
                config.add_include_dir(include);
        if (options.count("macro_definition"))
            for (auto& macro : options["macro_definition"].as<std::vector<std::string>>())
            {
                auto equal = macro.find('=');
                auto name  = macro.substr(0, equal);
                if (equal == std::string::npos)
                    config.define_macro(std::move(name), "");
                else
                {
                    auto def = macro.substr(equal + 1u);
                    config.define_macro(std::move(name), std::move(def));
                }
            }
        if (options.count("macro_undefinition"))
            for (auto& name : options["macro_undefinition"].as<std::vector<std::string>>())
                config.undefine_macro(name);
        if (options.count("feature"))
            for (auto& name : options["feature"].as<std::vector<std::string>>())
                config.enable_feature(name);

        // the compile_flags are generic flags
        cppast::compile_flags flags;
        if (options.count("gnu_extensions"))
            flags |= cppast::compile_flag::gnu_extensions;
        if (options.count("msvc_extensions"))
            flags |= cppast::compile_flag::ms_extensions;
        if (options.count("msvc_compatibility"))
            flags |= cppast::compile_flag::ms_compatibility;

        if (options["std"].as<std::string>() == "c++98")
            config.set_flags(cppast::cpp_standard::cpp_98, flags);
        else if (options["std"].as<std::string>() == "c++03")
            config.set_flags(cppast::cpp_standard::cpp_03, flags);
        else if (options["std"].as<std::string>() == "c++11")
            config.set_flags(cppast::cpp_standard::cpp_11, flags);
        else if (options["std"].as<std::string>() == "c++14")
            config.set_flags(cppast::cpp_standard::cpp_14, flags);
        else if (options["std"].as<std::string>() == "c++1z")
            config.set_flags(cppast::cpp_standard::cpp_1z, flags);
        else if (options["std"].as<std::string>() == "c++17")
            config.set_flags(cppast::cpp_standard::cpp_17, flags);
        else if (options["std"].as<std::string>() == "c++2a")
            config.set_flags(cppast::cpp_standard::cpp_2a, flags);
        else if (options["std"].as<std::string>() == "c++20")
            config.set_flags(cppast::cpp_standard::cpp_20, flags);
        else if (options["std"].as<std::string>() == "c++2b")
            config.set_flags(cppast::cpp_standard::cpp_2b, flags);
        else
        {
            print_error("invalid value '" + options["std"].as<std::string>() + "' for std flag");
            return 1;
        }

        // the logger is used to print diagnostics
        cppast::stderr_diagnostic_logger logger;
        if (options.count("verbose"))
            logger.set_verbose(true);

        auto file = parse_file(config, logger, options["file"].as<std::string>(),
                               options.count("fatal_errors") == 1);
        if (!file)
            return 2;
        print_ast(std::cout, *file);
    }
}
catch (const cppast::libclang_error& ex)
{
    print_error(std::string("[fatal parsing error] ") + ex.what());
    return 2;
}
 */