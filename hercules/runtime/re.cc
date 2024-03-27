// Copyright 2024 The EA Authors.
// Copyright(c) 2015-present, Gabi Melman & spdlog contributors.
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

#include "hercules/runtime/lib.h"
#include <cstring>
#include <re2/re2.h>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

using Regex = re2::RE2;
using re2::StringPiece;

/*
 * Flags -- (!) must match Hercules's
 */

#define ASCII (1 << 0)
#define DEBUG (1 << 1)
#define IGNORECASE (1 << 2)
#define LOCALE (1 << 3)
#define MULTILINE (1 << 4)
#define DOTALL (1 << 5)
#define VERBOSE (1 << 6)

static inline Regex::Options flags2opt(hs_int_t flags) {
    Regex::Options opt;
    opt.set_log_errors(false);
    opt.set_encoding(Regex::Options::Encoding::EncodingLatin1);

    if (flags & ASCII) {
        // nothing
    }

    if (flags & DEBUG) {
        // nothing
    }

    if (flags & IGNORECASE) {
        opt.set_case_sensitive(false);
    }

    if (flags & LOCALE) {
        // nothing
    }

    if (flags & MULTILINE) {
        opt.set_one_line(false);
    }

    if (flags & DOTALL) {
        opt.set_dot_nl(true);
    }

    if (flags & VERBOSE) {
        // nothing
    }

    return opt;
}

/*
 * Internal helpers & utilities
 */

struct Span {
    hs_int_t start;
    hs_int_t end;
};

template<typename KV>
struct GCMapAllocator : public std::allocator<KV> {
    GCMapAllocator() = default;

    GCMapAllocator(GCMapAllocator<KV> const &) = default;

    template<typename KV1>
    GCMapAllocator(const GCMapAllocator<KV1> &) noexcept {}

    KV *allocate(std::size_t n) { return (KV *) hs_alloc_uncollectable(n * sizeof(KV)); }

    void deallocate(KV *p, std::size_t n) { hs_free(p); }

    template<typename U>
    struct rebind {
        using other = GCMapAllocator<U>;
    };
};

static inline hs_str_t convert(const std::string &p) {
    hs_int_t n = p.size();
    auto *s = (char *) hs_alloc_atomic(n);
    std::memcpy(s, p.data(), n);
    return {n, s};
}

static inline StringPiece str2sp(const hs_str_t &s) {
    return StringPiece(s.str, s.len);
}

using Key = std::pair<hs_str_t, hs_int_t>;

struct KeyEqual {
    bool operator()(const Key &a, const Key &b) const {
        return a.second == b.second && str2sp(a.first) == str2sp(b.first);
    }
};

struct KeyHash {
    std::size_t operator()(const Key &k) const {
        using sv = std::string_view;
        return std::hash<sv>()(sv(k.first.str, k.first.len)) ^ k.second;
    }
};

static thread_local std::unordered_map<const Key, Regex, KeyHash, KeyEqual,
        GCMapAllocator<std::pair<const Key, Regex>>>
        cache;

static inline Regex *get(const hs_str_t &p, hs_int_t flags) {
    auto key = std::make_pair(p, flags);
    auto it = cache.find(key);
    if (it == cache.end()) {
        auto result = cache.emplace(std::piecewise_construct, std::forward_as_tuple(key),
                                    std::forward_as_tuple(str2sp(p), flags2opt(flags)));
        return &result.first->second;
    } else {
        return &it->second;
    }
}

/*
 * Matching
 */

HS_FUNC Span *hs_re_match(Regex *re, hs_int_t anchor, hs_str_t s, hs_int_t pos,
                            hs_int_t endpos) {
    const int num_groups = re->NumberOfCapturingGroups() + 1; // need $0
    std::vector<StringPiece> groups;
    groups.resize(num_groups);

    if (!re->Match(str2sp(s), pos, endpos, static_cast<Regex::Anchor>(anchor),
                   groups.data(), groups.size())) {
        // Ensure that groups are null before converting to spans!
        for (auto &it: groups) {
            it = StringPiece();
        }
    }

    auto *spans = (Span *) hs_alloc_atomic(num_groups * sizeof(Span));
    unsigned i = 0;
    for (const auto &it: groups) {
        if (it.data() == nullptr) {
            spans[i++] = {-1, -1};
        } else {
            spans[i++] = {static_cast<hs_int_t>(it.data() - s.str),
                          static_cast<hs_int_t>(it.data() - s.str + it.size())};
        }
    }

    return spans;
}

HS_FUNC Span hs_re_match_one(Regex *re, hs_int_t anchor, hs_str_t s, hs_int_t pos,
                               hs_int_t endpos) {
    StringPiece m;
    if (!re->Match(str2sp(s), pos, endpos, static_cast<Regex::Anchor>(anchor), &m, 1))
        return {-1, -1};
    else
        return {static_cast<hs_int_t>(m.data() - s.str),
                static_cast<hs_int_t>(m.data() - s.str + m.size())};
}

/*
 * General functions
 */

HS_FUNC hs_str_t hs_re_escape(hs_str_t p) {
    return convert(Regex::QuoteMeta(str2sp(p)));
}

HS_FUNC Regex *hs_re_compile(hs_str_t p, hs_int_t flags) { return get(p, flags); }

HS_FUNC void hs_re_purge() { cache.clear(); }

/*
 * Pattern methods
 */

HS_FUNC hs_int_t hs_re_pattern_groups(Regex *pattern) {
    return pattern->NumberOfCapturingGroups();
}

HS_FUNC hs_int_t hs_re_group_name_to_index(Regex *pattern, hs_str_t name) {
    const auto &mapping = pattern->NamedCapturingGroups();
    auto it = mapping.find(std::string(name.str, name.len));
    return (it != mapping.end()) ? it->second : -1;
}

HS_FUNC hs_str_t hs_re_group_index_to_name(Regex *pattern, hs_int_t index) {
    const auto &mapping = pattern->CapturingGroupNames();
    auto it = mapping.find(index);
    hs_str_t empty = {0, nullptr};
    return (it != mapping.end()) ? convert(it->second) : empty;
}

HS_FUNC bool seq_re_check_rewrite_string(Regex *pattern, hs_str_t rewrite,
                                          hs_str_t *error) {
    std::string e;
    bool ans = pattern->CheckRewriteString(str2sp(rewrite), &e);
    if (!ans)
        *error = convert(e);
    return ans;
}

HS_FUNC hs_str_t hs_re_pattern_error(Regex *pattern) {
    if (pattern->ok())
        return {0, nullptr};
    return convert(pattern->error());
}
