// Copyright 2024 The Elastic-AI Authors.
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

#ifndef COLLIE_STRINGS_REPLACE_H_
#define COLLIE_STRINGS_REPLACE_H_

#include <string>
#include <utility>
#include <vector>
#include <string_view>
#include <collie/strings/format.h>

namespace collie {
    /**
     * @ingroup collie_strings_replace
     * @brief replace string
     * @details Replaces character sequences within a given string with replacements provided
     *          within an initializer list of key/value pairs. Candidate replacements are
     *          considered in order as they occur within the string, with earlier matches
     *          taking precedence, and longer matches taking precedence for candidates
     *          starting at the same position in the string. Once a substitution is made, the
     *          replaced text is not considered for any further substitutions.
     *          Example:
     *          @code
     *          std::string s = collie::str_replace_all(
     *              "$who bought $count #Noun. Thanks $who!",
     *              {{"$count", collie::format(5)},
     *               {"$who", "Bob"},
     *               {"#Noun", "Apples"}});
     *          EXPECT_EQ("Bob bought 5 Apples. Thanks Bob!", s);
     *          @endcode
     * @param s The string to be replaced.
     * @param replacements The string to replace.
     * @return The replaced string.
     */
    [[nodiscard]] std::string str_replace_all(
            std::string_view s,
            std::initializer_list<std::pair<std::string_view, std::string_view>>
            replacements);

    /**
     * @ingroup collie_strings_replace
     * @brief replace string
     * @details Overload of `str_replace_all()` to accept a container of key/value replacement
     *          pairs (typically either an associative map or a `std::vector` of `std::pair`
     *          elements). A vector of pairs is generally more efficient.
     *          Examples:
     *          @code
     *          std::map<const std::string_view, const std::string_view> replacements;
     *          replacements["$who"] = "Bob";
     *          replacements["$count"] = "5";
     *          replacements["#Noun"] = "Apples";
     *          std::string s = collie::str_replace_all(
     *              "$who bought $count #Noun. Thanks $who!",
     *              replacements);
     *          EXPECT_EQ("Bob bought 5 Apples. Thanks Bob!", s);
     *          @endcode
     *
     *          // A std::vector of std::pair elements can be more efficient.
     *          std::vector<std::pair<const std::string_view, std::string>> replacements;
     *          replacements.push_back({"&", "&amp;"});
     *          replacements.push_back({"<", "&lt;"});
     *          replacements.push_back({">", "&gt;"});
     *          std::string s = collie::str_replace_all("if (ptr < &foo)",
     *                        replacements);
     *          EXPECT_EQ("if (ptr &lt; &amp;foo)", s);
     *          @endcode
     * @param s The string to be replaced.
     * @param replacements The string to replace.
     * @return The replaced string.
     */
    template<typename StrToStrMapping>
    std::string str_replace_all(std::string_view s,
                              const StrToStrMapping &replacements);

    /**
     * @ingroup collie_strings_replace
     * @brief replace string
     * @details Overload of `str_replace_all()` to replace character sequences within a given
     *          output string *in place* with replacements provided within an initializer
     *          list of key/value pairs, returning the number of substitutions that occurred.
     *          Example:
     *          @code
     *          std::string s = std::string("$who bought $count #Noun. Thanks $who!");
     *          int count;
     *          count = collie::str_replace_all({{"$count", collie::format(5)},
     *                                      {"$who", "Bob"},
     *                                      {"#Noun", "Apples"}}, &s);
     *          EXPECT_EQ(count, 4);
     *          EXPECT_EQ("Bob bought 5 Apples. Thanks Bob!", s);
     *          @endcode
     * @param replacements The string to replace.
     * @param target The string to be replaced.
     * @return The replaced string.
     */
    int str_replace_all(
            std::initializer_list<std::pair<std::string_view, std::string_view>>
            replacements,
            std::string *target);

    /**
     * @ingroup collie_strings_replace
     * @brief replace string
     * @details Overload of `str_replace_all()` to replace patterns within a given output
     *          string *in place* with replacements provided within a container of key/value
     *          pairs.
     *          Example:
     *          @code
     *          std::string s = std::string("if (ptr < &foo)");
     *          int count = collie::str_replace_all({{"&", "&amp;"},
     *                                           {"<", "&lt;"},
     *                                           {">", "&gt;"}}, &s);
     *          EXPECT_EQ(count, 2);
     *          EXPECT_EQ("if (ptr &lt; &amp;foo)", s);
     *          @endcode
     * @param replacements The string to replace.
     * @param target The string to be replaced.
     * @return The replaced string count.
     */
    template<typename StrToStrMapping>
    int str_replace_all(const StrToStrMapping &replacements, std::string *target);

    // Implementation details only, past this point.
    namespace strings_internal {

        struct ViableSubstitution {
            std::string_view old;
            std::string_view replacement;
            size_t offset;

            ViableSubstitution(std::string_view old_str,
                               std::string_view replacement_str, size_t offset_val)
                    : old(old_str), replacement(replacement_str), offset(offset_val) {}

            // One substitution occurs "before" another (takes priority) if either
            // it has the lowest offset, or it has the same offset but a larger size.
            bool OccursBefore(const ViableSubstitution &y) const {
                if (offset != y.offset) return offset < y.offset;
                return old.size() > y.old.size();
            }
        };

        // Build a vector of ViableSubstitutions based on the given list of
        // replacements. subs can be implemented as a priority_queue. However, it turns
        // out that most callers have small enough a list of substitutions that the
        // overhead of such a queue isn't worth it.
        template<typename StrToStrMapping>
        std::vector<ViableSubstitution> FindSubstitutions(
                std::string_view s, const StrToStrMapping &replacements) {
            std::vector<ViableSubstitution> subs;
            subs.reserve(replacements.size());

            for (const auto &rep: replacements) {
                using std::get;
                std::string_view old(get<0>(rep));

                size_t pos = s.find(old);
                if (pos == s.npos) continue;

                // Ignore attempts to replace "". This condition is almost never true,
                // but above condition is frequently true. That's why we test for this
                // now and not before.
                if (old.empty()) continue;

                subs.emplace_back(old, get<1>(rep), pos);

                // Insertion sort to ensure the last ViableSubstitution comes before
                // all the others.
                size_t index = subs.size();
                while (--index && subs[index - 1].OccursBefore(subs[index])) {
                    std::swap(subs[index], subs[index - 1]);
                }
            }
            return subs;
        }

        using FixedMapping =
                std::initializer_list<std::pair<std::string_view, std::string_view>>;

        int ApplySubstitutions(std::string_view s,
                               std::vector<ViableSubstitution> *subs_ptr,
                               std::string *result_ptr) {
            auto &subs = *subs_ptr;
            int substitutions = 0;
            size_t pos = 0;
            while (!subs.empty()) {
                auto &sub = subs.back();
                if (sub.offset >= pos) {
                    if (pos <= s.size()) {
                        result_ptr->append(format("{}{}", s.substr(pos, sub.offset - pos), sub.replacement));
                    }
                    pos = sub.offset + sub.old.size();
                    substitutions += 1;
                }
                sub.offset = s.find(sub.old, pos);
                if (sub.offset == s.npos) {
                    subs.pop_back();
                } else {
                    // Insertion sort to ensure the last ViableSubstitution continues to be
                    // before all the others.
                    size_t index = subs.size();
                    while (--index && subs[index - 1].OccursBefore(subs[index])) {
                        std::swap(subs[index], subs[index - 1]);
                    }
                }
            }
            result_ptr->append(s.data() + pos, s.size() - pos);
            return substitutions;
        }

    }  // namespace strings_internal

    template<typename StrToStrMapping>
    std::string str_replace_all(std::string_view s,
                              const StrToStrMapping &replacements) {
        auto subs = strings_internal::FindSubstitutions(s, replacements);
        std::string result;
        result.reserve(s.size());
        strings_internal::ApplySubstitutions(s, &subs, &result);
        return result;
    }

    template<typename StrToStrMapping>
    int str_replace_all(const StrToStrMapping &replacements, std::string *target) {
        auto subs = strings_internal::FindSubstitutions(*target, replacements);
        if (subs.empty()) return 0;

        std::string result;
        result.reserve(target->size());
        int substitutions =
                strings_internal::ApplySubstitutions(*target, &subs, &result);
        target->swap(result);
        return substitutions;
    }

    [[nodiscard]] inline std::string str_replace_all(
            std::string_view s,
            std::initializer_list<std::pair<std::string_view, std::string_view>>
            replacements) {
        return str_replace_all<strings_internal::FixedMapping>(s, replacements);
    }

    inline int str_replace_all(strings_internal::FixedMapping replacements,
                        std::string *target) {
        return str_replace_all<strings_internal::FixedMapping>(replacements, target);
    }

}  // namespace collie

#endif  // COLLIE_STRINGS_REPLACE_H_
