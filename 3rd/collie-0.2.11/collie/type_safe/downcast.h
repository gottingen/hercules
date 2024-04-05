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
//

#pragma once

#include <type_traits>
#include <collie/type_safe/config.h>
#include <collie/type_safe/detail/assert.h>

namespace collie::ts
{
/// Tag type to specify the derived type of a [ts::downcast]().
///
/// Pass it as first parameter to the appropriate overload.
template <typename T>
struct derived_type
{};

namespace detail
{
    // same type
    template <typename T>
    bool is_safe_downcast(derived_type<T>, const T&)
    {
        return true;
    }

    // polymorphic type, so we can check
    template <typename Derived, typename Base>
    auto is_safe_downcast(derived_type<Derived>, const Base& obj) ->
        typename std::enable_if<std::is_polymorphic<Base>::value, bool>::type
    {
#if TYPE_SAFE_USE_RTTI
        return dynamic_cast<const Derived*>(&obj) != nullptr;
#else
        return true;
#endif
    }

    // non-polymorphic type, no check possible
    template <typename Derived, typename Base>
    auto is_safe_downcast(derived_type<Derived>, const Base&) ->
        typename std::enable_if<!std::is_polymorphic<Base>::value, bool>::type
    {
        return true;
    }

    template <typename Derived, typename Base>
    void validate_downcast(const Base& obj) noexcept
    {
        using derived_t = typename std::decay<Derived>::type;
        static_assert(std::is_base_of<Base, derived_t>::value,
                      "can only downcast from base to derived class");
        DEBUG_ASSERT(detail::is_safe_downcast(derived_type<derived_t>{}, obj),
                     detail::precondition_error_handler{}, "not a safe downcast");
    }
} // namespace detail

/// Casts an object of base class type to the derived class type.
/// \returns The object converted as if `static_cast<Derived>(obj)`.
/// \requires `Base` must be a base class of `Derived`,
/// and the dynamic type of `obj` must be `Derived`.
template <typename Derived, typename Base>
Derived downcast(Base& obj) noexcept
{
    detail::validate_downcast<Derived>(obj);
    return static_cast<Derived>(obj);
}

/// Casts an object of base class type to the derived class type.
/// \returns The object converted as if `static_cast<derived_ref>(obj)`,
/// where `derived_ref` is the type of `Derived` with matching qualifiers.
/// \requires `Base` must be a base class of `Derived`,
/// and the dynamic type of `obj` must be `Derived`.
/// \group downcast_tag
template <typename Derived, typename Base>
Derived& downcast(derived_type<Derived>, Base& obj) noexcept
{
    detail::validate_downcast<Derived>(obj);
    return static_cast<Derived&>(obj);
}

/// \group downcast_tag
template <typename Derived, typename Base>
const Derived& downcast(derived_type<Derived>, const Base& obj) noexcept
{
    detail::validate_downcast<Derived>(obj);
    return static_cast<const Derived&>(obj);
}
} // namespace collie::ts
