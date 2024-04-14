//
// Created by 李寅斌 on 2023/3/4.
//

#ifndef TURBO_STRINGS_STRING_VIEW_H_
#define TURBO_STRINGS_STRING_VIEW_H_

#include "turbo/platform/port.h"
#include <string_view>

namespace turbo {

    // ClippedSubstr()
    //
    // Like `s.substr(pos, n)`, but clips `pos` to an upper bound of `s.size()`.
    // Provided because std::string_view::substr throws if `pos > size()`
    inline std::string_view ClippedSubstr(std::string_view s, size_t pos,
                                          size_t n = std::string_view::npos) {
        pos = (std::min)(pos, static_cast<size_t>(s.size()));
        return s.substr(pos, n);
    }

    // NullSafeStringView()
    //
    // Creates an `std::string_view` from a pointer `p` even if it's null-valued.
    // This function should be used where an `std::string_view` can be created from
    // a possibly-null pointer.
    constexpr std::string_view NullSafeStringView(const char *p) {
        return p ? std::string_view(p) : std::string_view();
    }

}  // namespace turbo

#endif // TURBO_STRINGS_STRING_VIEW_H_
