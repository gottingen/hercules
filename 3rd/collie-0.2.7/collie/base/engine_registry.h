// Copyright 2023 The Elastic-AI Authors.
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

#include <cstddef>
#include <type_traits>
#include <initializer_list>
#include <utility>

namespace collie {

    struct unavailable {
        static constexpr bool supported() noexcept { return false; }

        static constexpr bool available() noexcept { return false; }

        static constexpr unsigned version() noexcept { return 0; }

        static constexpr std::size_t alignment() noexcept { return 0; }

        static constexpr bool requires_alignment() noexcept { return false; }

        static constexpr char const *name() noexcept { return "<none>"; }
    };

    namespace detail {
        // Checks whether T appears in Tys.
        template<class T, class... Tys>
        struct contains;

        template<class T>
        struct contains<T> : std::false_type {
        };

        template<class T, class Ty, class... Tys>
        struct contains<T, Ty, Tys...>
                : std::conditional<std::is_same<Ty, T>::value, std::true_type,
                        contains<T, Tys...>>::type {
        };

        template<unsigned... Vals>
        struct is_sorted;

        template<>
        struct is_sorted<> : std::true_type {
        };

        template<unsigned Val>
        struct is_sorted<Val> : std::true_type {
        };

        template<unsigned V0, unsigned V1, unsigned... Vals>
        struct is_sorted<V0, V1, Vals...>
                : std::conditional<(V0 >= V1), is_sorted<V1, Vals...>,
                        std::false_type>::type {
        };

        template<typename T>
        inline constexpr T max_of(T value) noexcept {
            return value;
        }

        template<typename T, typename... Ts>
        inline constexpr T max_of(T head0, T head1, Ts... tail) noexcept {
            return max_of((head0 > head1 ? head0 : head1), tail...);
        }

        template<typename... Ts>
        struct head;

        template<typename T, typename... Ts>
        struct head<T, Ts...> {
            using type = T;
        };

        template<>
        struct head<> {
            using type = unavailable;
        };

    } // namespace detail

    // An engine_list is a list of engines, sorted by version number.
    template<class... Engines>
    struct engine_list {
#ifndef NDEBUG
        static_assert(detail::is_sorted<Engines::version()...>::value,
                      "engines list must be sorted by version");
#endif

        using best = typename detail::head<Engines...>::type;

        template<class Engine>
        using add = engine_list<Engines..., Engine>;

        template<class... OtherEngines>
        using extend = engine_list<Engines..., OtherEngines...>;

        template<class Engine>
        static constexpr bool contains() noexcept {
            return detail::contains<Engine, Engines...>::value;
        }

        template<class F>
        static inline void for_each(F &&f) noexcept {
            (void) std::initializer_list<bool>{(f(Engines{}), true)...};
        }

        static constexpr std::size_t alignment() noexcept {
            // all alignments are a power of two
            return detail::max_of(Engines::alignment()..., static_cast<size_t>(0));
        }
    };

    namespace detail {

        // Filter engine lists Engines, picking only supported engines and adding
        // them to L.
        template<class L, class... Engines>
        struct supported_helper;

        template<class L>
        struct supported_helper<L, engine_list<>> {
            using type = L;
        };

        template<class L, class Engine, class... Engines>
        struct supported_helper<L, engine_list<Engine, Engines...>>
                : supported_helper<
                        typename std::conditional<Engine::supported(),
                                typename L::template add<Engine>, L>::type,
                        engine_list<Engines...>> {
        };

    } // namespace detail

    // Joins all engine_list Engines in a single engine_list.
    template<class... Engines>
    struct engine_join;

    template<class Engine>
    struct engine_join<Engine> {
        using type = Engine;
    };

    template<class Engine, class... Engines, class... Args>
    struct engine_join<Engine, engine_list<Engines...>, Args...>
            : engine_join<typename Engine::template extend<Engines...>, Args...> {
    };

    template<class... Engines>
    struct engine_supported : detail::supported_helper<engine_list<>, Engines...> {
    };

    template<class F, class EngineList>
    class engine_dispatcher {

        const unsigned best_engine_found;
        F functor;

        template<class Engine, class... Tys>
        inline auto walk_engines(engine_list<Engine>, Tys &&... args) noexcept -> decltype(functor(Engine{},
                                                                                             std::forward<Tys>(
                                                                                                     args)...)) {
            assert(Engine::available() && "At least one engine must be supported during dispatch");
            return functor(Engine{}, std::forward<Tys>(args)...);
        }

        template<class Engine, class EngineNext, class... Engines, class... Tys>
        inline auto
        walk_engines(engine_list<Engine, EngineNext, Engines...>, Tys &&... args) noexcept -> decltype(functor(Engine{},
                                                                                                     std::forward<Tys>(
                                                                                                             args)...)) {
            if (Engine::version() <= best_engine_found)
                return functor(Engine{}, std::forward<Tys>(args)...);
            else
                return walk_engines(engine_list<EngineNext, Engines...>{}, std::forward<Tys>(args)...);
        }

    public:
        using default_engine = typename EngineList::best;
        inline engine_dispatcher(unsigned best_version, F f) noexcept
                : best_engine_found(best_version), functor(f) {
        }

        template<class... Tys>
        inline auto
        operator()(Tys &&... args) noexcept -> decltype(functor(default_engine{}, std::forward<Tys>(args)...)) {
            return walk_engines(EngineList{}, std::forward<Tys>(args)...);
        }
    };

    template<class EngineList, class F>
    inline engine_dispatcher<F, EngineList> dispatch(unsigned best_version, F &&f) noexcept {
        return {best_version, std::forward<F>(f)};
    }

}  // namespace collie
