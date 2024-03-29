// Copyright (c) 2017-2024, University of Cincinnati, developed by Henry Schreiner
// under NSF AWARD 1414736 and by the respective contributors.
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <collie/cli/encoding.h>
#include <collie/cli/macros.h>
#include <collie/cli/string_tools.h>
#include <collie/cli/type_tools.h>
#include <map>
#include <string>
#include <utility>

namespace collie {

    inline std::string Validator::operator()(std::string &str) const {
        std::string retstring;
        if (active_) {
            if (non_modifying_) {
                std::string value = str;
                retstring = func_(value);
            } else {
                retstring = func_(str);
            }
        }
        return retstring;
    }

    [[nodiscard]] inline Validator Validator::description(std::string validator_desc) const {
        Validator newval(*this);
        newval.desc_function_ = [validator_desc]() { return validator_desc; };
        return newval;
    }

    inline Validator Validator::operator&(const Validator &other) const {
        Validator newval;

        newval._merge_description(*this, other, " AND ");

        // Give references (will make a copy in lambda function)
        const std::function<std::string(std::string &filename)> &f1 = func_;
        const std::function<std::string(std::string &filename)> &f2 = other.func_;

        newval.func_ = [f1, f2](std::string &input) {
            std::string s1 = f1(input);
            std::string s2 = f2(input);
            if (!s1.empty() && !s2.empty())
                return std::string("(") + s1 + ") AND (" + s2 + ")";
            return s1 + s2;
        };

        newval.active_ = active_ && other.active_;
        newval.application_index_ = application_index_;
        return newval;
    }

    inline Validator Validator::operator|(const Validator &other) const {
        Validator newval;

        newval._merge_description(*this, other, " OR ");

        // Give references (will make a copy in lambda function)
        const std::function<std::string(std::string &)> &f1 = func_;
        const std::function<std::string(std::string &)> &f2 = other.func_;

        newval.func_ = [f1, f2](std::string &input) {
            std::string s1 = f1(input);
            std::string s2 = f2(input);
            if (s1.empty() || s2.empty())
                return std::string();

            return std::string("(") + s1 + ") OR (" + s2 + ")";
        };
        newval.active_ = active_ && other.active_;
        newval.application_index_ = application_index_;
        return newval;
    }

    inline Validator Validator::operator!() const {
        Validator newval;
        const std::function<std::string()> &dfunc1 = desc_function_;
        newval.desc_function_ = [dfunc1]() {
            auto str = dfunc1();
            return (!str.empty()) ? std::string("NOT ") + str : std::string{};
        };
        // Give references (will make a copy in lambda function)
        const std::function<std::string(std::string &res)> &f1 = func_;

        newval.func_ = [f1, dfunc1](std::string &test) -> std::string {
            std::string s1 = f1(test);
            if (s1.empty()) {
                return std::string("check ") + dfunc1() + " succeeded improperly";
            }
            return std::string{};
        };
        newval.active_ = active_;
        newval.application_index_ = application_index_;
        return newval;
    }

    inline void
    Validator::_merge_description(const Validator &val1, const Validator &val2, const std::string &merger) {

        const std::function<std::string()> &dfunc1 = val1.desc_function_;
        const std::function<std::string()> &dfunc2 = val2.desc_function_;

        desc_function_ = [=]() {
            std::string f1 = dfunc1();
            std::string f2 = dfunc2();
            if ((f1.empty()) || (f2.empty())) {
                return f1 + f2;
            }
            return std::string(1, '(') + f1 + ')' + merger + '(' + f2 + ')';
        };
    }

    namespace detail {

        inline path_type check_path(const char *file) noexcept {
            std::error_code ec;
            auto stat = collie::filesystem::status(to_path(file), ec);
            if (ec) {
                return path_type::nonexistent;
            }
            switch (stat.type()) {
                case collie::filesystem::file_type::none:  // LCOV_EXCL_LINE
                case collie::filesystem::file_type::not_found:
                    return path_type::nonexistent;  // LCOV_EXCL_LINE
                case collie::filesystem::file_type::directory:
                    return path_type::directory;
                case collie::filesystem::file_type::symlink:
                case collie::filesystem::file_type::block:
                case collie::filesystem::file_type::character:
                case collie::filesystem::file_type::fifo:
                case collie::filesystem::file_type::socket:
                case collie::filesystem::file_type::regular:
                case collie::filesystem::file_type::unknown:
                default:
                    return path_type::file;
            }
        }

        inline ExistingFileValidator::ExistingFileValidator() : Validator("FILE") {
            func_ = [](std::string &filename) {
                auto path_result = check_path(filename.c_str());
                if (path_result == path_type::nonexistent) {
                    return "File does not exist: " + filename;
                }
                if (path_result == path_type::directory) {
                    return "File is actually a directory: " + filename;
                }
                return std::string();
            };
        }

        inline ExistingDirectoryValidator::ExistingDirectoryValidator() : Validator("DIR") {
            func_ = [](std::string &filename) {
                auto path_result = check_path(filename.c_str());
                if (path_result == path_type::nonexistent) {
                    return "Directory does not exist: " + filename;
                }
                if (path_result == path_type::file) {
                    return "Directory is actually a file: " + filename;
                }
                return std::string();
            };
        }

        inline ExistingPathValidator::ExistingPathValidator() : Validator("PATH(existing)") {
            func_ = [](std::string &filename) {
                auto path_result = check_path(filename.c_str());
                if (path_result == path_type::nonexistent) {
                    return "Path does not exist: " + filename;
                }
                return std::string();
            };
        }

        inline NonexistentPathValidator::NonexistentPathValidator() : Validator("PATH(non-existing)") {
            func_ = [](std::string &filename) {
                auto path_result = check_path(filename.c_str());
                if (path_result != path_type::nonexistent) {
                    return "Path already exists: " + filename;
                }
                return std::string();
            };
        }

        inline IPV4Validator::IPV4Validator() : Validator("IPV4") {
            func_ = [](std::string &ip_addr) {
                auto result = collie::detail::split(ip_addr, '.');
                if (result.size() != 4) {
                    return std::string("Invalid IPV4 address must have four parts (") + ip_addr + ')';
                }
                int num = 0;
                for (const auto &var: result) {
                    using collie::detail::lexical_cast;
                    bool retval = lexical_cast(var, num);
                    if (!retval) {
                        return std::string("Failed parsing number (") + var + ')';
                    }
                    if (num < 0 || num > 255) {
                        return std::string("Each IP number must be between 0 and 255 ") + var;
                    }
                }
                return std::string{};
            };
        }

        inline EscapedStringTransformer::EscapedStringTransformer() {
            func_ = [](std::string &str) {
                try {
                    if (str.size() > 1 && (str.front() == '\"' || str.front() == '\'' || str.front() == '`') &&
                        str.front() == str.back()) {
                        process_quoted_string(str);
                    } else if (str.find_first_of('\\') != std::string::npos) {
                        if (detail::is_binary_escaped_string(str)) {
                            str = detail::extract_binary_string(str);
                        } else {
                            str = remove_escaped_characters(str);
                        }
                    }
                    return std::string{};
                } catch (const std::invalid_argument &ia) {
                    return std::string(ia.what());
                }
            };
        }
    }  // namespace detail

    inline FileOnDefaultPath::FileOnDefaultPath(std::string default_path, bool enableErrorReturn)
            : Validator("FILE") {
        func_ = [default_path, enableErrorReturn](std::string &filename) {
            auto path_result = detail::check_path(filename.c_str());
            if (path_result == detail::path_type::nonexistent) {
                std::string test_file_path = default_path;
                if (default_path.back() != '/' && default_path.back() != '\\') {
                    // Add folder separator
                    test_file_path += '/';
                }
                test_file_path.append(filename);
                path_result = detail::check_path(test_file_path.c_str());
                if (path_result == detail::path_type::file) {
                    filename = test_file_path;
                } else {
                    if (enableErrorReturn) {
                        return "File does not exist: " + filename;
                    }
                }
            }
            return std::string{};
        };
    }

    inline AsSizeValue::AsSizeValue(bool kb_is_1000) : AsNumberWithUnit(get_mapping(kb_is_1000)) {
        if (kb_is_1000) {
            description("SIZE [b, kb(=1000b), kib(=1024b), ...]");
        } else {
            description("SIZE [b, kb(=1024b), ...]");
        }
    }

    inline std::map<std::string, AsSizeValue::result_t> AsSizeValue::init_mapping(bool kb_is_1000) {
        std::map<std::string, result_t> m;
        result_t k_factor = kb_is_1000 ? 1000 : 1024;
        result_t ki_factor = 1024;
        result_t k = 1;
        result_t ki = 1;
        m["b"] = 1;
        for (std::string p: {"k", "m", "g", "t", "p", "e"}) {
            k *= k_factor;
            ki *= ki_factor;
            m[p] = k;
            m[p + "b"] = k;
            m[p + "i"] = ki;
            m[p + "ib"] = ki;
        }
        return m;
    }

    inline std::map<std::string, AsSizeValue::result_t> AsSizeValue::get_mapping(bool kb_is_1000) {
        if (kb_is_1000) {
            static auto m = init_mapping(true);
            return m;
        }
        static auto m = init_mapping(false);
        return m;
    }

    namespace detail {

        inline std::pair<std::string, std::string> split_program_name(std::string commandline) {
            // try to determine the programName
            std::pair<std::string, std::string> vals;
            trim(commandline);
            auto esp = commandline.find_first_of(' ', 1);
            while (detail::check_path(commandline.substr(0, esp).c_str()) != path_type::file) {
                esp = commandline.find_first_of(' ', esp + 1);
                if (esp == std::string::npos) {
                    // if we have reached the end and haven't found a valid file just assume the first argument is the
                    // program name
                    if (commandline[0] == '"' || commandline[0] == '\'' || commandline[0] == '`') {
                        bool embeddedQuote = false;
                        auto keyChar = commandline[0];
                        auto end = commandline.find_first_of(keyChar, 1);
                        while ((end != std::string::npos) &&
                               (commandline[end - 1] == '\\')) {  // deal with escaped quotes
                            end = commandline.find_first_of(keyChar, end + 1);
                            embeddedQuote = true;
                        }
                        if (end != std::string::npos) {
                            vals.first = commandline.substr(1, end - 1);
                            esp = end + 1;
                            if (embeddedQuote) {
                                vals.first = find_and_replace(vals.first, std::string("\\") + keyChar,
                                                              std::string(1, keyChar));
                            }
                        } else {
                            esp = commandline.find_first_of(' ', 1);
                        }
                    } else {
                        esp = commandline.find_first_of(' ', 1);
                    }

                    break;
                }
            }
            if (vals.first.empty()) {
                vals.first = commandline.substr(0, esp);
                rtrim(vals.first);
            }

            // strip the program name
            vals.second = (esp < commandline.length() - 1) ? commandline.substr(esp + 1) : std::string{};
            ltrim(vals.second);
            return vals;
        }

    }  // namespace detail
/// @}
}  // namespace collie
