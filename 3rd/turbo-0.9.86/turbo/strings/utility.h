//
// Created by jeff on 23-7-2.
//

#ifndef TURBO_STRINGS_UTILITY_H_
#define TURBO_STRINGS_UTILITY_H_


#include "turbo/strings/str_case_conv.h"
#include "turbo/strings/str_replace.h"
#include "turbo/strings/str_split.h"
#include "turbo/strings/str_trim.h"
#include "turbo/strings/str_strip.h"
#include "turbo/strings/numbers.h"
#include "turbo/strings/match.h"

namespace turbo {

    struct CharCompareBase {
        CharCompareBase() = default;

        constexpr operator bool() const { return ptr != nullptr; }

        constexpr  bool operator==(char c) const {
            return ptr && *ptr == c;
        }

        constexpr  bool operator!=(char c) const {
            return ptr && *ptr != c;
        }

        constexpr  bool operator<(char c) const {
            return ptr && *ptr < c;
        }

        constexpr  bool operator>(char c) const {
            return ptr && *ptr > c;
        }

        constexpr  bool operator<=(char c) const {
            return ptr && *ptr <= c;
        }

        constexpr  bool operator>=(char c) const {
            return ptr && *ptr >= c;
        }

        constexpr const char* value() const {
            return ptr;
        }

        const char *ptr{nullptr};
    };

    struct BackChar : public CharCompareBase {
        BackChar(std::string_view str) {
            if (!str.empty()) {
                ptr = &str.back();
            }
        }
    };

    struct FrontChar : public CharCompareBase {
        FrontChar(std::string_view str) {
            if (!str.empty()) {
                ptr = &str.front();
            }
        }
    };
}  // namespace turbo

#endif  // TURBO_STRINGS_UTILITY_H_
