
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#ifndef HERCULES_ENGINE_CORE_EXPLAIN_H_
#define HERCULES_ENGINE_CORE_EXPLAIN_H_

#include <string>

namespace hercules::engine {

    enum explain_result_code {
        kExplainSuccess = 0,
        kExplainTableFail = 1,
        kExplainMatcherFail = 2,
        kExplainReleFail = 3,
        kExplainTopNFail = 4,
        kExplainCollectorFail = 8,
    };

    class query_explain {
    public:
        query_explain() = default;

        explicit query_explain(const std::string &desc) : desc_(desc) {}

        void add_line(const std::string &line);

        std::string get_desc() const { return desc_; }

        void add_sub_explanation(const query_explain &exp, const std::string &prefix = "\n");

        void set_match();

        bool is_match();

        explain_result_code get_explain_result_code();

        void set_explain_result_code(explain_result_code code);

    private:
        explain_result_code explain_code_ = kExplainTableFail;
        std::string desc_;
    };

}  // namespace hercules::engine

#endif  // HERCULES_ENGINE_CORE_EXPLAIN_H_