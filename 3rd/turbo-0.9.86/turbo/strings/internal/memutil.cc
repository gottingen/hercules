// Copyright 2020 The Turbo Authors.
//
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

#include "turbo/strings/internal/memutil.h"

#include <cstdlib>

namespace turbo::strings_internal {

    int memcasecmp(const char *s1, const char *s2, size_t len) {
        const unsigned char *us1 = reinterpret_cast<const unsigned char *>(s1);
        const unsigned char *us2 = reinterpret_cast<const unsigned char *>(s2);

        for (size_t i = 0; i < len; i++) {
            const int diff =
                    int{static_cast<unsigned char>(turbo::ascii_to_lower(us1[i]))} -
                    int{static_cast<unsigned char>(turbo::ascii_to_lower(us2[i]))};
            if (diff != 0) return diff;
        }
        return 0;
    }

    char *memdup(const char *s, size_t slen) {
        void *copy;
        if ((copy = malloc(slen)) == nullptr) return nullptr;
        memcpy(copy, s, slen);
        return reinterpret_cast<char *>(copy);
    }

    char *memrchr(const char *s, int c, size_t slen) {
        for (const char *e = s + slen - 1; e >= s; e--) {
            if (*e == c) return const_cast<char *>(e);
        }
        return nullptr;
    }

    size_t memspn(const char *s, size_t slen, const char *accept) {
        const char *p = s;
        const char *spanp;
        char c, sc;

        cont:
        c = *p++;
        if (slen-- == 0)
            return static_cast<size_t>(p - 1 - s);
        for (spanp = accept; (sc = *spanp++) != '\0';)
            if (sc == c) goto cont;
        return static_cast<size_t>(p - 1 - s);
    }

    size_t memcspn(const char *s, size_t slen, const char *reject) {
        const char *p = s;
        const char *spanp;
        char c, sc;

        while (slen-- != 0) {
            c = *p++;
            for (spanp = reject; (sc = *spanp++) != '\0';)
                if (sc == c)
                    return static_cast<size_t>(p - 1 - s);
        }
        return static_cast<size_t>(p - s);
    }

    char *mempbrk(const char *s, size_t slen, const char *accept) {
        const char *scanp;
        int sc;

        for (; slen; ++s, --slen) {
            for (scanp = accept; (sc = *scanp++) != '\0';)
                if (sc == *s) return const_cast<char *>(s);
        }
        return nullptr;
    }

    // This is significantly faster for case-sensitive matches with very
    // few possible matches.  See unit test for benchmarks.
    const char *memmatch(const char *phaystack, size_t haylen, const char *pneedle,
                         size_t neelen) {
        if (0 == neelen) {
            return phaystack;  // even if haylen is 0
        }
        if (haylen < neelen) return nullptr;

        const char *match;
        const char *hayend = phaystack + haylen - neelen + 1;
        // A static cast is used here to work around the fact that memchr returns
        // a void* on Posix-compliant systems and const void* on Windows.
        while (
                (match = static_cast<const char *>(memchr(
                        phaystack, pneedle[0], static_cast<size_t>(hayend - phaystack))))) {
            if (memcmp(match, pneedle, neelen) == 0)
                return match;
            else
                phaystack = match + 1;
        }
        return nullptr;
    }

    const char32_t *memmatch(const char32_t *phaystack,
                             size_t haylen,
                             const char32_t *pneedle,
                             size_t neelen) {
        if (0 == neelen) {
            return phaystack;  // even if haylen is 0
        }
        if (haylen < neelen)
            return nullptr;
        const char32_t *match;
        const char32_t *hayend = phaystack + haylen - neelen + 1;
        while ((match = std::char_traits<char32_t>::find(
                phaystack, hayend - phaystack, pneedle[0]))) {
            if (std::char_traits<char32_t>::compare(match, pneedle, neelen) == 0)
                return match;
            else
                phaystack = match + 1;
        }
        return nullptr;
    }

}  // namespace turbo::strings_internal
