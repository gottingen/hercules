// Copyright (C) 2016-2020 Jonathan Müller <jonathanmueller.dev@gmail.com>
// This file is subject to the license terms in the LICENSE file
// found in the top-level directory of this distribution.

#ifndef TYPE_SAFE_DETAIL_ASSERT_HPP_INCLUDED
#define TYPE_SAFE_DETAIL_ASSERT_HPP_INCLUDED

#include <collie/base/debug_assert.h>

#include <collie/type_safe/config.h>

namespace collie::ts::detail {

    struct assert_handler : collie::debug_assert::set_level<TYPE_SAFE_ENABLE_ASSERTIONS>,
                            collie::debug_assert::default_handler {
    };

    struct precondition_error_handler
            : collie::debug_assert::set_level<TYPE_SAFE_ENABLE_PRECONDITION_CHECKS>,
              collie::debug_assert::default_handler {
    };

    inline void on_disabled_exception() noexcept {
        struct handler : collie::debug_assert::set_level<1>, collie::debug_assert::default_handler {
        };
        DEBUG_UNREACHABLE(handler{}, "attempt to throw an exception but exceptions are disabled");
    }
} // namespace collie::ts::detail

#endif // TYPE_SAFE_DETAIL_ASSERT_HPP_INCLUDED`
