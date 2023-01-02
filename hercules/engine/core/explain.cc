/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#include "hercules/engine/core/explain.h"
#include <regex>

namespace hercules::engine {

    void query_explain::add_line(const std::string &line) { desc_ += "\n" + line; }

    void query_explain::add_sub_explanation(const query_explain &exp, const std::string &prefix) {
        std::string sub_desc = "  " + exp.get_desc();
        sub_desc = std::regex_replace(sub_desc, std::regex("\n"), "\n  ");
        desc_ += prefix + sub_desc;
    }

    explain_result_code query_explain::get_explain_result_code() { return explain_code_; }

    void query_explain::set_explain_result_code(explain_result_code code) { explain_code_ = code; }

    void query_explain::set_match() { set_explain_result_code(kExplainSuccess); }

    bool query_explain::is_match() { return get_explain_result_code() == kExplainSuccess; }

}  // namespace hercules::engine
