//
// span for C++98 and later.
// Based on http://wg21.link/p0122r7
// For more information see https://github.com/martinmoene/span-lite
//
// Copyright 2018-2021 Martin Moene
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef NONSTD_SPAN_HPP_INCLUDED
#define NONSTD_SPAN_HPP_INCLUDED

#define span_lite_MAJOR  0
#define span_lite_MINOR  11
#define span_lite_PATCH  0

#define span_lite_VERSION  span_STRINGIFY(span_lite_MAJOR) "." span_STRINGIFY(span_lite_MINOR) "." span_STRINGIFY(span_lite_PATCH)

#define span_STRINGIFY(x)  span_STRINGIFY_( x )
#define span_STRINGIFY_(x)  #x

// span configuration:

#define span_SPAN_DEFAULT  0
#define span_SPAN_NONSTD   1
#define span_SPAN_STD      2

// tweak header support:

#ifdef __has_include
# if __has_include(<nonstd/span.tweak.hpp>)
#  include <nonstd/span.tweak.hpp>
# endif
#define span_HAVE_TWEAK_HEADER  1
#else
#define span_HAVE_TWEAK_HEADER  0
//# pragma message("span.hpp: Note: Tweak header not supported.")
#endif

// span selection and configuration:

#define span_HAVE(feature)  ( span_HAVE_##feature )

#ifndef  span_CONFIG_SELECT_SPAN
# define span_CONFIG_SELECT_SPAN  ( span_HAVE_STD_SPAN ? span_SPAN_STD : span_SPAN_NONSTD )
#endif

#ifndef  span_CONFIG_EXTENT_TYPE
# define span_CONFIG_EXTENT_TYPE  std::size_t
#endif

#ifndef  span_CONFIG_SIZE_TYPE
# define span_CONFIG_SIZE_TYPE  std::size_t
#endif

#ifdef span_CONFIG_INDEX_TYPE
# error `span_CONFIG_INDEX_TYPE` is deprecated since v0.7.0; it is replaced by `span_CONFIG_SIZE_TYPE`.
#endif

// span configuration (features):

#ifndef  span_FEATURE_WITH_INITIALIZER_LIST_P2447
# define span_FEATURE_WITH_INITIALIZER_LIST_P2447  0
#endif

#ifndef  span_FEATURE_WITH_CONTAINER
#ifdef   span_FEATURE_WITH_CONTAINER_TO_STD
# define span_FEATURE_WITH_CONTAINER  span_IN_STD( span_FEATURE_WITH_CONTAINER_TO_STD )
#else
# define span_FEATURE_WITH_CONTAINER  0
# define span_FEATURE_WITH_CONTAINER_TO_STD  0
#endif
#endif

#ifndef  span_FEATURE_CONSTRUCTION_FROM_STDARRAY_ELEMENT_TYPE
# define span_FEATURE_CONSTRUCTION_FROM_STDARRAY_ELEMENT_TYPE  0
#endif

#ifndef  span_FEATURE_MEMBER_AT
# define span_FEATURE_MEMBER_AT  0
#endif

#ifndef  span_FEATURE_MEMBER_BACK_FRONT
# define span_FEATURE_MEMBER_BACK_FRONT  1
#endif

#ifndef  span_FEATURE_MEMBER_SWAP
# define span_FEATURE_MEMBER_SWAP  0
#endif

#ifndef  span_FEATURE_NON_MEMBER_FIRST_LAST_SUB
# define span_FEATURE_NON_MEMBER_FIRST_LAST_SUB  0
#elif    span_FEATURE_NON_MEMBER_FIRST_LAST_SUB
# define span_FEATURE_NON_MEMBER_FIRST_LAST_SUB_SPAN       1
# define span_FEATURE_NON_MEMBER_FIRST_LAST_SUB_CONTAINER  1
#endif

#ifndef  span_FEATURE_NON_MEMBER_FIRST_LAST_SUB_SPAN
# define span_FEATURE_NON_MEMBER_FIRST_LAST_SUB_SPAN  0
#endif

#ifndef  span_FEATURE_NON_MEMBER_FIRST_LAST_SUB_CONTAINER
# define span_FEATURE_NON_MEMBER_FIRST_LAST_SUB_CONTAINER  0
#endif

#ifndef  span_FEATURE_COMPARISON
# define span_FEATURE_COMPARISON  0  // Note: C++20 does not provide comparison
#endif

#ifndef  span_FEATURE_SAME
# define span_FEATURE_SAME  0
#endif

#if span_FEATURE_SAME && !span_FEATURE_COMPARISON
# error `span_FEATURE_SAME` requires `span_FEATURE_COMPARISON`
#endif

#ifndef  span_FEATURE_MAKE_SPAN
#ifdef   span_FEATURE_MAKE_SPAN_TO_STD
# define span_FEATURE_MAKE_SPAN  span_IN_STD( span_FEATURE_MAKE_SPAN_TO_STD )
#else
# define span_FEATURE_MAKE_SPAN  0
# define span_FEATURE_MAKE_SPAN_TO_STD  0
#endif
#endif

#ifndef  span_FEATURE_BYTE_SPAN
# define span_FEATURE_BYTE_SPAN  0
#endif


#if    defined( span_CONFIG_CONTRACT_LEVEL_ON )
# define        span_CONFIG_CONTRACT_LEVEL_MASK  0x11
#elif  defined( span_CONFIG_CONTRACT_LEVEL_OFF )
# define        span_CONFIG_CONTRACT_LEVEL_MASK  0x00
#elif  defined( span_CONFIG_CONTRACT_LEVEL_EXPECTS_ONLY )
# define        span_CONFIG_CONTRACT_LEVEL_MASK  0x01
#elif  defined( span_CONFIG_CONTRACT_LEVEL_ENSURES_ONLY )
# define        span_CONFIG_CONTRACT_LEVEL_MASK  0x10
#else
# define        span_CONFIG_CONTRACT_LEVEL_MASK  0x11
#endif

#if    defined( span_CONFIG_CONTRACT_VIOLATION_THROWS )
# define        span_CONFIG_CONTRACT_VIOLATION_THROWS_V  span_CONFIG_CONTRACT_VIOLATION_THROWS
#else
# define        span_CONFIG_CONTRACT_VIOLATION_THROWS_V  0
#endif

#if    defined( span_CONFIG_CONTRACT_VIOLATION_THROWS     ) && span_CONFIG_CONTRACT_VIOLATION_THROWS && \
       defined( span_CONFIG_CONTRACT_VIOLATION_TERMINATES ) && span_CONFIG_CONTRACT_VIOLATION_TERMINATES
# error Please define none or one of span_CONFIG_CONTRACT_VIOLATION_THROWS and span_CONFIG_CONTRACT_VIOLATION_TERMINATES to 1, but not both.
#endif

// C++ language version detection (C++23 is speculative):
// Note: VC14.0/1900 (VS2015) lacks too much from C++14.

#ifndef   span_CPLUSPLUS
# if defined(_MSVC_LANG ) && !defined(__clang__)
#  define span_CPLUSPLUS  (_MSC_VER == 1900 ? 201103L : _MSVC_LANG )
# else
#  define span_CPLUSPLUS  __cplusplus
# endif
#endif

#define span_CPP20_OR_GREATER  ( span_CPLUSPLUS >= 202002L )
#define span_CPP23_OR_GREATER  ( span_CPLUSPLUS >= 202300L )

// C++ language version (represent 98 as 3):

#define span_CPLUSPLUS_V  ( span_CPLUSPLUS / 100 - (span_CPLUSPLUS > 200000 ? 2000 : 1994) )

#define span_IN_STD(v)  ( ((v) == 98 ? 3 : (v)) >= span_CPLUSPLUS_V )

#define span_CONFIG(feature)  ( span_CONFIG_##feature )
#define span_FEATURE(feature)  ( span_FEATURE_##feature )
#define span_FEATURE_TO_STD(feature)  ( span_IN_STD( span_FEATURE( feature##_TO_STD ) ) )

// Use C++20 std::span if available and requested:

#if span_CPP20_OR_GREATER && defined(__has_include )
# if __has_include( <span> )
#  define span_HAVE_STD_SPAN  1
# else
#  define span_HAVE_STD_SPAN  0
# endif
#else
# define  span_HAVE_STD_SPAN  0
#endif

#define  span_USES_STD_SPAN  ( (span_CONFIG_SELECT_SPAN == span_SPAN_STD) || ((span_CONFIG_SELECT_SPAN == span_SPAN_DEFAULT) && span_HAVE_STD_SPAN) )

//
// Use C++20 std::span:
//

#if span_USES_STD_SPAN

#include <span>

namespace collie {

using std::span;
using std::dynamic_extent;

// Note: C++20 does not provide comparison
// using std::operator==;
// using std::operator!=;
// using std::operator<;
// using std::operator<=;
// using std::operator>;
// using std::operator>=;
}  // namespace collie

#else  // span_USES_STD_SPAN

#include <algorithm>

#if defined(_MSC_VER ) && !defined(__clang__)
# define span_COMPILER_MSVC_VER      (_MSC_VER )
# define span_COMPILER_MSVC_VERSION  (_MSC_VER / 10 - 10 * ( 5 + (_MSC_VER < 1900 ) ) )
#else
# define span_COMPILER_MSVC_VER      0
# define span_COMPILER_MSVC_VERSION  0
#endif

#define span_COMPILER_VERSION(major, minor, patch)  ( 10 * ( 10 * (major) + (minor) ) + (patch) )

#if defined(__clang__)
# define span_COMPILER_CLANG_VERSION  span_COMPILER_VERSION(__clang_major__, __clang_minor__, __clang_patchlevel__)
#else
# define span_COMPILER_CLANG_VERSION  0
#endif

#if defined(__GNUC__) && !defined(__clang__)
# define span_COMPILER_GNUC_VERSION  span_COMPILER_VERSION(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)
#else
# define span_COMPILER_GNUC_VERSION  0
#endif

// half-open range [lo..hi):
#define span_BETWEEN(v, lo, hi)  ( (lo) <= (v) && (v) < (hi) )

// Compiler warning suppression:

#if defined(__clang__)
# pragma clang diagnostic push
# pragma clang diagnostic ignored "-Wundef"
# pragma clang diagnostic ignored "-Wmismatched-tags"
# define span_RESTORE_WARNINGS()   _Pragma( "clang diagnostic pop" )

#elif defined __GNUC__
# pragma GCC   diagnostic push
# pragma GCC   diagnostic ignored "-Wundef"
# define span_RESTORE_WARNINGS()   _Pragma( "GCC diagnostic pop" )

#elif span_COMPILER_MSVC_VER >= 1900
# define span_DISABLE_MSVC_WARNINGS(codes)  __pragma(warning(push))  __pragma(warning(disable: codes))
# define span_RESTORE_WARNINGS()            __pragma(warning(pop ))

// Suppress the following MSVC GSL warnings:
// - C26439, gsl::f.6 : special function 'function' can be declared 'noexcept'
// - C26440, gsl::f.6 : function 'function' can be declared 'noexcept'
// - C26472, gsl::t.1 : don't use a static_cast for arithmetic conversions;
//                      use brace initialization, gsl::narrow_cast or gsl::narrow
// - C26473: gsl::t.1 : don't cast between pointer types where the source type and the target type are the same
// - C26481: gsl::b.1 : don't use pointer arithmetic. Use span instead
// - C26490: gsl::t.1 : don't use reinterpret_cast

span_DISABLE_MSVC_WARNINGS( 26439 26440 26472 26473 26481 26490 )

#else
# define span_RESTORE_WARNINGS()  /*empty*/
#endif

#if defined(__cpp_deduction_guides)
# define span_HAVE_DEDUCTION_GUIDES         1
#else
# define span_HAVE_DEDUCTION_GUIDES         ( !span_BETWEEN( span_COMPILER_MSVC_VER, 1, 1913 ))
#endif

# include <memory>
# include <array>
# include <cstddef>
# include <iterator> // for std::data(), std::size()
# include <type_traits>
# include <cstdio>
# include <stdexcept>

// Contract violation

#define span_ELIDE_CONTRACT_EXPECTS  ( 0 == ( span_CONFIG_CONTRACT_LEVEL_MASK & 0x01 ) )
#define span_ELIDE_CONTRACT_ENSURES  ( 0 == ( span_CONFIG_CONTRACT_LEVEL_MASK & 0x10 ) )

#define span_CONTRACT_CHECK(type, cond) \
    cond ? static_cast< void >( 0 ) \
         : collie::span_lite::detail::report_contract_violation( span_LOCATION( __FILE__, __LINE__ ) ": " type " violation." )

#if span_ELIDE_CONTRACT_EXPECTS
# define span_EXPECTS( cond )  /* Expect elided */
#else
# define span_EXPECTS(cond)  span_CONTRACT_CHECK( "Precondition", cond )
#endif


#ifdef __GNUG__
# define span_LOCATION(file, line)  file ":" span_STRINGIFY( line )
#else
# define span_LOCATION( file, line )  file "(" span_STRINGIFY( line ) ")"
#endif

// Method enabling

#define span_REQUIRES_0(VA) \
    template< bool B = (VA), typename std::enable_if<B, int>::type = 0 >

# if span_BETWEEN(span_COMPILER_MSVC_VERSION, 1, 140)
// VS 2013 and earlier seem to have trouble with SFINAE for default non-type arguments
# define span_REQUIRES_T(VA) \
    , typename = typename std::enable_if< ( VA ), collie::span_lite::detail::enabler >::type
# else
# define span_REQUIRES_T(VA) \
    , typename std::enable_if< (VA), int >::type = 0
# endif


namespace collie {
    namespace span_lite {

// [views.constants], constants

        typedef span_CONFIG_EXTENT_TYPE extent_t;
        typedef span_CONFIG_SIZE_TYPE size_t;

        constexpr const extent_t dynamic_extent = static_cast<extent_t>( -1 );

        template<class T, extent_t Extent = dynamic_extent>
        class span;

        // Tag to select span constructor taking a container (prevent ms-gsl warning C26426):

        struct with_container_t {
            constexpr with_container_t() noexcept {}
        };

        const constexpr with_container_t with_container;


        // C++20 emulation:

        namespace std20 {

#if span_HAVE(DEDUCTION_GUIDES)
            template<class T>
            using iter_reference_t = decltype(*std::declval<T &>());
#endif

        } // namespace std20

// Implementation details:

        namespace detail {

/*enum*/ struct enabler {
            };

            template<typename T>
            constexpr bool is_positive(T x) {
                return std::is_signed<T>::value ? x >= 0 : true;
            }


            template<class Q>
            struct is_span_oracle : std::false_type {
            };

            template<class T, span_CONFIG_EXTENT_TYPE Extent>
            struct is_span_oracle<span<T, Extent> > : std::true_type {
            };

            template<class Q>
            struct is_span : is_span_oracle<typename std::remove_cv<Q>::type> {
            };

            template<class Q>
            struct is_std_array_oracle : std::false_type {
            };

            template<class T, std::size_t Extent>
            struct is_std_array_oracle<std::array<T, Extent> > : std::true_type {
            };

            template<class Q>
            struct is_std_array : is_std_array_oracle<typename std::remove_cv<Q>::type> {
            };

            template<class Q>
            struct is_array : std::false_type {
            };

            template<class T>
            struct is_array<T[]> : std::true_type {
            };

            template<class T, std::size_t N>
            struct is_array<T[N]> : std::true_type {
            };

            template<class, class = void>
            struct has_size_and_data : std::false_type {
            };

            template<class C>
            struct has_size_and_data
                    <
                            C, std::void_t<
                            decltype(std::size(std::declval<C>())),
                            decltype(std::data(std::declval<C>()))>
                    > : std::true_type {
            };

            template<class, class, class = void>
            struct is_compatible_element : std::false_type {
            };

            template<class C, class E>
            struct is_compatible_element
                    <
                            C, E, std::void_t<
                            decltype(std::data(std::declval<C>()))>
                    > : std::is_convertible<typename std::remove_pointer<decltype(std::data(
                    std::declval<C &>()))>::type(*)[], E(*)[]> {
            };

            template<class C>
            struct is_container : std::bool_constant
                                          <
                                                  !is_span<C>::value
                                                  && !is_array<C>::value
                                                  && !is_std_array<C>::value
                                                  && has_size_and_data<C>::value
                                          > {
            };

            template<class C, class E>
            struct is_compatible_container : std::bool_constant
                                                     <
                                                             is_container<C>::value
                                                             && is_compatible_element<C, E>::value
                                                     > {
            };


#if defined(__clang__)
# pragma clang diagnostic ignored "-Wlong-long"
#elif defined __GNUC__
# pragma GCC   diagnostic ignored "-Wformat=ll"
# pragma GCC   diagnostic ignored "-Wlong-long"
#endif

            [[noreturn]] inline void throw_out_of_range(size_t idx, size_t size) {
                const char fmt[] = "span::at(): index '%lli' is out of range [0..%lli)";
                char buffer[2 * 20 + sizeof fmt];
                sprintf(buffer, fmt, static_cast<long long>(idx), static_cast<long long>(size));

                throw std::out_of_range(buffer);
            }


            struct contract_violation : std::logic_error {
                explicit contract_violation(char const *const message)
                        : std::logic_error(message) {}
            };

            inline void report_contract_violation(char const *msg) {
                throw contract_violation(msg);
            }

        }  // namespace detail

        // Prevent signed-unsigned mismatch:

#define span_sizeof(T)  static_cast<extent_t>( sizeof(T) )

        template<class T>
        inline constexpr size_t to_size(T size) {
            return static_cast<size_t>( size );
        }

        //
        // [views.span] - A view over a contiguous, single-dimension sequence of objects
        //
        template<class T, extent_t Extent /*= dynamic_extent*/ >
        class span {
        public:
            // constants and types

            typedef T element_type;
            typedef typename std::remove_cv<T>::type value_type;

            typedef T &reference;
            typedef T *pointer;
            typedef T const *const_pointer;
            typedef T const &const_reference;

            typedef size_t size_type;
            typedef extent_t extent_type;

            typedef pointer iterator;
            typedef const_pointer const_iterator;

            typedef std::ptrdiff_t difference_type;

            typedef std::reverse_iterator<iterator> reverse_iterator;
            typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

            //    static constexpr extent_type extent = Extent;
            enum {
                extent = Extent
            };

            // 26.7.3.2 Constructors, copy, and assignment [span.cons]

            span_REQUIRES_0(
                    (Extent == 0) ||
                    (Extent == dynamic_extent)
            )
            constexpr span() noexcept
                    : data_(nullptr), size_(0) {
                // span_EXPECTS( data() == nullptr );
                // span_EXPECTS( size() == 0 );
            }

            // Didn't yet succeed in combining the next two constructors:

            constexpr span(std::nullptr_t, size_type count)
                    : data_(nullptr), size_(count) {
                span_EXPECTS(data_ == nullptr && count == 0);
            }

            template<typename It
                    span_REQUIRES_T((std::is_convertible<decltype(*std::declval<It &>()), element_type &>::value))>
            constexpr span(It first, size_type count)
                    : data_(to_address(first)), size_(count) {
                span_EXPECTS(
                        (data_ == nullptr && count == 0) ||
                        (data_ != nullptr && detail::is_positive(count))
                );
            }

            template<typename It,
                    typename End span_REQUIRES_T((std::is_convertible<decltype(&*std::declval<It &>()), element_type *>::value && !std::is_convertible<End, std::size_t>::value))
            >
            constexpr span(It first, End last)
                    : data_(to_address(first)), size_(to_size(last - first)) {
                span_EXPECTS(
                        last - first >= 0
                );
            }

            template<std::size_t N span_REQUIRES_T((
                                                           (Extent == dynamic_extent ||
                                                            Extent == static_cast<extent_t>(N))
                                                           &&
                                                           std::is_convertible<value_type(*)[], element_type(*)[]>::value
                                                   ))
            >
            constexpr span(element_type ( &arr )[N]) noexcept
                    : data_(std::addressof(arr[0])), size_(N) {}

            template<std::size_t N span_REQUIRES_T((
                                                           (Extent == dynamic_extent ||
                                                            Extent == static_cast<extent_t>(N))
                                                           &&
                                                           std::is_convertible<value_type(*)[], element_type(*)[]>::value
                                                   ))
            >
# if span_FEATURE(CONSTRUCTION_FROM_STDARRAY_ELEMENT_TYPE)
            constexpr span( std::array< element_type, N > & arr ) noexcept
# else
            constexpr span(std::array<value_type, N> &arr) noexcept
# endif
                    : data_(arr.data()), size_(to_size(arr.size())) {}

            template<std::size_t N span_REQUIRES_T((
                                                           (Extent == dynamic_extent ||
                                                            Extent == static_cast<extent_t>(N))
                                                           &&
                                                           std::is_convertible<value_type(*)[], element_type(*)[]>::value
                                                   ))
            >
            constexpr span(std::array<value_type, N> const &arr) noexcept
                    : data_(arr.data()), size_(to_size(arr.size())) {}


            template<class Container span_REQUIRES_T((
                                                             detail::is_compatible_container<Container, element_type>::value
                                                     ))
            >
            constexpr span(Container &cont)
                    : data_(std::data(cont)), size_(to_size(std::size(cont))) {}

            template<class Container span_REQUIRES_T((
                                                             std::is_const<element_type>::value
                                                             &&
                                                             detail::is_compatible_container<Container, element_type>::value
                                                     ))
            >
            constexpr span(Container const &cont)
                    : data_(std::data(cont)), size_(to_size(std::size(cont))) {}


#if span_FEATURE(WITH_CONTAINER)

            template< class Container >
            constexpr span( with_container_t, Container & cont )
                : data_( cont.size() == 0 ? nullptr : std::addressof( cont[0] ) )
                , size_( to_size( cont.size() ) )
            {}

            template< class Container >
            constexpr span( with_container_t, Container const & cont )
                : data_( cont.size() == 0 ? nullptr : const_cast<pointer>( std::addressof( cont[0] ) ) )
                , size_( to_size( cont.size() ) )
            {}
#endif

#if span_FEATURE(WITH_INITIALIZER_LIST_P2447)

            // constexpr explicit(extent != dynamic_extent) span(std::initializer_list<value_type> il) noexcept;

#if !span_BETWEEN( span_COMPILER_MSVC_VERSION, 120, 130 )

            template< extent_t U = Extent
                span_REQUIRES_T((
                    U != dynamic_extent
                ))
            >
#if span_COMPILER_GNUC_VERSION >= 900   // prevent GCC's "-Winit-list-lifetime"
            constexpr explicit span( std::initializer_list<value_type> il ) noexcept
            {
                data_ = il.begin();
                size_ = il.size();
            }
#else
            constexpr explicit span( std::initializer_list<value_type> il ) noexcept
                : data_( il.begin() )
                , size_( il.size()  )
            {}
#endif

#endif // MSVC 120 (VS2013)

            template< extent_t U = Extent
                span_REQUIRES_T((
                    U == dynamic_extent
                ))
            >
#if span_COMPILER_GNUC_VERSION >= 900   // prevent GCC's "-Winit-list-lifetime"
            constexpr /*explicit*/ span( std::initializer_list<value_type> il ) noexcept
            {
                data_ = il.begin();
                size_ = il.size();
            }
#else
            constexpr /*explicit*/ span( std::initializer_list<value_type> il ) noexcept
                : data_( il.begin() )
                , size_( il.size()  )
            {}
#endif

#endif // P2447

            constexpr span(span const &other) noexcept = default;

            ~span() noexcept = default;

            constexpr span &operator=(span const &other) noexcept = default;

            template<class OtherElementType, extent_type OtherExtent span_REQUIRES_T((
                                                                                             (Extent ==
                                                                                              dynamic_extent ||
                                                                                              OtherExtent ==
                                                                                              dynamic_extent ||
                                                                                              Extent == OtherExtent)
                                                                                             &&
                                                                                             std::is_convertible<OtherElementType(
                                                                                                     *)[], element_type(*)[]>::value
                                                                                     ))
            >
            constexpr span(span<OtherElementType, OtherExtent> const &other) noexcept
                    : data_(other.data()), size_(other.size()) {
                span_EXPECTS(OtherExtent == dynamic_extent || other.size() == to_size(OtherExtent));
            }

            // 26.7.3.3 Subviews [span.sub]

            template<extent_type Count>
            constexpr span<element_type, Count>
            first() const {
                span_EXPECTS(detail::is_positive(Count) && Count <= size());

                return span<element_type, Count>(data(), Count);
            }

            template<extent_type Count>
            constexpr span<element_type, Count>
            last() const {
                span_EXPECTS(detail::is_positive(Count) && Count <= size());

                return span<element_type, Count>(data() + (size() - Count), Count);
            }

            template<size_type Offset, extent_type Count = dynamic_extent>
            constexpr span<element_type, Count>
            subspan() const {
                span_EXPECTS(
                        (detail::is_positive(Offset) && Offset <= size()) &&
                        (Count == dynamic_extent || (detail::is_positive(Count) && Count + Offset <= size()))
                );

                return span<element_type, Count>(
                        data() + Offset,
                        Count != dynamic_extent ? Count : (Extent != dynamic_extent ? Extent - Offset : size() -
                                                                                                        Offset));
            }

            constexpr span<element_type, dynamic_extent>
            first(size_type count) const {
                span_EXPECTS(detail::is_positive(count) && count <= size());

                return span<element_type, dynamic_extent>(data(), count);
            }

            constexpr span<element_type, dynamic_extent>
            last(size_type count) const {
                span_EXPECTS(detail::is_positive(count) && count <= size());

                return span<element_type, dynamic_extent>(data() + (size() - count), count);
            }

            constexpr span<element_type, dynamic_extent>
            subspan(size_type offset, size_type count = static_cast<size_type>(dynamic_extent)) const {
                span_EXPECTS(
                        ((detail::is_positive(offset) && offset <= size())) &&
                        (count == static_cast<size_type>(dynamic_extent) ||
                         (detail::is_positive(count) && offset + count <= size()))
                );

                return span<element_type, dynamic_extent>(
                        data() + offset, count == static_cast<size_type>(dynamic_extent) ? size() - offset : count);
            }

            // 26.7.3.4 Observers [span.obs]

            constexpr size_type size() const noexcept {
                return size_;
            }

            constexpr std::ptrdiff_t ssize() const noexcept {
                return static_cast<std::ptrdiff_t>( size_ );
            }

            constexpr size_type size_bytes() const noexcept {
                return size() * to_size(sizeof(element_type));
            }

            [[nodiscard]] constexpr bool empty() const noexcept {
                return size() == 0;
            }

            // 26.7.3.5 Element access [span.elem]

            constexpr reference operator[](size_type idx) const {
                span_EXPECTS(detail::is_positive(idx) && idx < size());

                return *(data() + idx);
            }

            constexpr reference at(size_type idx) const {
                if (!detail::is_positive(idx) || size() <= idx) {
                    detail::throw_out_of_range(idx, size());
                }
                return *(data() + idx);
            }

            constexpr pointer data() const noexcept {
                return data_;
            }

#if span_FEATURE(MEMBER_BACK_FRONT)

            constexpr reference front() const noexcept {
                span_EXPECTS(!empty());

                return *data();
            }

            constexpr reference back() const noexcept {
                span_EXPECTS(!empty());

                return *(data() + size() - 1);
            }

#endif

            // xx.x.x.x Modifiers [span.modifiers]

#if span_FEATURE(MEMBER_SWAP)

            constexpr void swap( span & other ) noexcept
            {
                using std::swap;
                swap( data_, other.data_ );
                swap( size_, other.size_ );
            }
#endif

            // 26.7.3.6 Iterator support [span.iterators]

            constexpr iterator begin() const noexcept {
                return {data()};
            }

            constexpr iterator end() const noexcept {
                return {data() + size()};
            }

            constexpr const_iterator cbegin() const noexcept {
                return {data()};
            }

            constexpr const_iterator cend() const noexcept {
                return {data() + size()};
            }

            constexpr reverse_iterator rbegin() const noexcept {
                return reverse_iterator(end());
            }

            constexpr reverse_iterator rend() const noexcept {
                return reverse_iterator(begin());
            }

            constexpr const_reverse_iterator crbegin() const noexcept {
                return const_reverse_iterator(cend());
            }

            constexpr const_reverse_iterator crend() const noexcept {
                return const_reverse_iterator(cbegin());
            }

        private:

            // Note: C++20 has std::pointer_traits<Ptr>::to_address( it );

            static inline constexpr pointer to_address(std::nullptr_t) noexcept {
                return nullptr;
            }

            template<typename U>
            static inline constexpr U *to_address(U *p) noexcept {
                return p;
            }

            template<typename Ptr span_REQUIRES_T((!std::is_pointer<Ptr>::value))
            >
            static inline constexpr pointer to_address(Ptr const &it) noexcept {
                return to_address(it.operator->());
            }

        private:
            pointer data_;
            size_type size_;
        };

// class template argument deduction guides:

#if span_HAVE(DEDUCTION_GUIDES)

        template<class T, size_t N>
        span(T (&)[N]) -> span<T, static_cast<extent_t>(N)>;

        template<class T, size_t N>
        span(std::array<T, N> &) -> span<T, static_cast<extent_t>(N)>;

        template<class T, size_t N>
        span(std::array<T, N> const &) -> span<const T, static_cast<extent_t>(N)>;


        template<class Container>
        span(Container &) -> span<typename Container::value_type>;

        template<class Container>
        span(Container const &) -> span<const typename Container::value_type>;


// iterator: constraints: It satisfies contiguous_­iterator.

        template<class It, class EndOrSize>
        span(It, EndOrSize) -> span<typename std::remove_reference<typename std20::iter_reference_t<It> >::type>;

#endif // span_HAVE( DEDUCTION_GUIDES )

// 26.7.3.7 Comparison operators [span.comparison]

#if span_FEATURE(COMPARISON)
#if span_FEATURE( SAME )

        template< class T1, extent_t E1, class T2, extent_t E2  >
        inline constexpr bool same( span<T1,E1> const & l, span<T2,E2> const & r ) noexcept
        {
            return std::is_same<T1, T2>::value
                && l.size() == r.size()
                && static_cast<void const*>( l.data() ) == r.data();
        }

#endif

        template< class T1, extent_t E1, class T2, extent_t E2  >
        inline constexpr bool operator==( span<T1,E1> const & l, span<T2,E2> const & r )
        {
            return
#if span_FEATURE( SAME )
                same( l, r ) ||
#endif
                ( l.size() == r.size() && std::equal( l.begin(), l.end(), r.begin() ) );
        }

        template< class T1, extent_t E1, class T2, extent_t E2  >
        inline constexpr bool operator<( span<T1,E1> const & l, span<T2,E2> const & r )
        {
            return std::lexicographical_compare( l.begin(), l.end(), r.begin(), r.end() );
        }

        template< class T1, extent_t E1, class T2, extent_t E2  >
        inline constexpr bool operator!=( span<T1,E1> const & l, span<T2,E2> const & r )
        {
            return !( l == r );
        }

        template< class T1, extent_t E1, class T2, extent_t E2  >
        inline constexpr bool operator<=( span<T1,E1> const & l, span<T2,E2> const & r )
        {
            return !( r < l );
        }

        template< class T1, extent_t E1, class T2, extent_t E2  >
        inline constexpr bool operator>( span<T1,E1> const & l, span<T2,E2> const & r )
        {
            return ( r < l );
        }

        template< class T1, extent_t E1, class T2, extent_t E2  >
        inline constexpr bool operator>=( span<T1,E1> const & l, span<T2,E2> const & r )
        {
            return !( l < r );
        }

#endif // span_FEATURE( COMPARISON )

// 26.7.2.6 views of object representation [span.objectrep]

// Avoid MSVC 14.1 (1910), VS 2017: warning C4307: '*': integral constant overflow:

        template<typename T, extent_t Extent>
        struct BytesExtent {
            enum ET : extent_t {
                value = span_sizeof(T) * Extent
            };
        };

        template<typename T>
        struct BytesExtent<T, dynamic_extent> {
            enum ET : extent_t {
                value = dynamic_extent
            };
        };

        template<class T, extent_t Extent>
        inline constexpr span<const std::byte, BytesExtent<T, Extent>::value>
        as_bytes(span<T, Extent> spn) noexcept {
            return span<const std::byte, BytesExtent<T, Extent>::value>(
                    reinterpret_cast< std::byte const * >( spn.data()), spn.size_bytes());  // NOLINT
        }

        template<class T, extent_t Extent>
        inline constexpr span<std::byte, BytesExtent<T, Extent>::value>
        as_writable_bytes(span<T, Extent> spn) noexcept {
            return span<std::byte, BytesExtent<T, Extent>::value>(
                    reinterpret_cast< std::byte * >( spn.data()), spn.size_bytes());  // NOLINT
        }


// 27.8 Container and view access [iterator.container]

        template<class T, extent_t Extent /*= dynamic_extent*/ >
        constexpr std::size_t size(span<T, Extent> const &spn) {
            return static_cast<std::size_t>( spn.size());
        }

        template<class T, extent_t Extent /*= dynamic_extent*/ >
        constexpr std::ptrdiff_t ssize(span<T, Extent> const &spn) {
            return static_cast<std::ptrdiff_t>( spn.size());
        }

    }  // namespace span_lite
}  // namespace collie

// make available in nonstd:

namespace collie {

    using span_lite::dynamic_extent;

    using span_lite::span;

    using span_lite::with_container;

#if span_FEATURE(COMPARISON)
#if span_FEATURE( SAME )
    using span_lite::same;
#endif

    using span_lite::operator==;
    using span_lite::operator!=;
    using span_lite::operator<;
    using span_lite::operator<=;
    using span_lite::operator>;
    using span_lite::operator>=;
#endif

    using span_lite::as_bytes;
    using span_lite::as_writable_bytes;
    using span_lite::size;
    using span_lite::ssize;

}  // namespace collie

#endif  // span_USES_STD_SPAN

// make_span() [span-lite extension]:

#if span_FEATURE(MAKE_SPAN) || span_FEATURE(NON_MEMBER_FIRST_LAST_SUB_SPAN) || span_FEATURE(NON_MEMBER_FIRST_LAST_SUB_CONTAINER)

#if span_USES_STD_SPAN
# define  constexpr  constexpr
# define  noexcept   noexcept
# define  nullptr    nullptr
# ifndef  span_CONFIG_EXTENT_TYPE
#  define span_CONFIG_EXTENT_TYPE  std::size_t
# endif
using extent_t = span_CONFIG_EXTENT_TYPE;
#endif  // span_USES_STD_SPAN

namespace collie {
namespace span_lite {

template< class T >
inline constexpr span<T>
make_span( T * ptr, size_t count ) noexcept
{
    return span<T>( ptr, count );
}

template< class T >
inline constexpr span<T>
make_span( T * first, T * last ) noexcept
{
    return span<T>( first, last );
}

template< class T, std::size_t N >
inline constexpr span<T, static_cast<extent_t>(N)>
make_span( T ( &arr )[ N ] ) noexcept
{
    return span<T, static_cast<extent_t>(N)>( &arr[ 0 ], N );
}

template< class T, std::size_t N >
inline constexpr span<T, static_cast<extent_t>(N)>
make_span( std::array< T, N > & arr ) noexcept
{
    return span<T, static_cast<extent_t>(N)>( arr );
}

template< class T, std::size_t N >
inline constexpr span< const T, static_cast<extent_t>(N) >
make_span( std::array< T, N > const & arr ) noexcept
{
    return span<const T, static_cast<extent_t>(N)>( arr );
}


template< class T >
inline constexpr span< const T >
make_span( std::initializer_list<T> il ) noexcept
{
    return span<const T>( il.begin(), il.size() );
}

#if span_USES_STD_SPAN

template< class Container, class EP = decltype( std::data(std::declval<Container&>())) >
inline constexpr auto
make_span( Container & cont ) noexcept -> span< typename std::remove_pointer<EP>::type >
{
    return span< typename std::remove_pointer<EP>::type >( cont );
}

template< class Container, class EP = decltype( std::data(std::declval<Container&>())) >
inline constexpr auto
make_span( Container const & cont ) noexcept -> span< const typename std::remove_pointer<EP>::type >
{
    return span< const typename std::remove_pointer<EP>::type >( cont );
}

#else

template< class Container, class EP = decltype( std::data(std::declval<Container&>())) >
inline constexpr auto
make_span( Container & cont ) noexcept -> span< typename std::remove_pointer<EP>::type >
{
    return span< typename std::remove_pointer<EP>::type >( cont );
}

template< class Container, class EP = decltype( std::data(std::declval<Container&>())) >
inline constexpr auto
make_span( Container const & cont ) noexcept -> span< const typename std::remove_pointer<EP>::type >
{
    return span< const typename std::remove_pointer<EP>::type >( cont );
}

#endif // span_USES_STD_SPAN || ( ... )

#if ! span_USES_STD_SPAN && span_FEATURE( WITH_CONTAINER )

template< class Container >
inline constexpr span<typename Container::value_type>
make_span( with_container_t, Container & cont ) noexcept
{
    return span< typename Container::value_type >( with_container, cont );
}

template< class Container >
inline constexpr span<const typename Container::value_type>
make_span( with_container_t, Container const & cont ) noexcept
{
    return span< const typename Container::value_type >( with_container, cont );
}

#endif // ! span_USES_STD_SPAN && span_FEATURE( WITH_CONTAINER )

// extensions: non-member views:
// this feature implies the presence of make_span()

#if span_FEATURE( NON_MEMBER_FIRST_LAST_SUB_SPAN )

template< extent_t Count, class T, extent_t Extent >
constexpr span<T, Count>
first( span<T, Extent> spn )
{
    return spn.template first<Count>();
}

template< class T, extent_t Extent >
constexpr span<T>
first( span<T, Extent> spn, size_t count )
{
    return spn.first( count );
}

template< extent_t Count, class T, extent_t Extent >
constexpr span<T, Count>
last( span<T, Extent> spn )
{
    return spn.template last<Count>();
}

template< class T, extent_t Extent >
constexpr span<T>
last( span<T, Extent> spn, size_t count )
{
    return spn.last( count );
}

template< size_t Offset, extent_t Count, class T, extent_t Extent >
constexpr span<T, Count>
subspan( span<T, Extent> spn )
{
    return spn.template subspan<Offset, Count>();
}

template< class T, extent_t Extent >
constexpr span<T>
subspan( span<T, Extent> spn, size_t offset, extent_t count = dynamic_extent )
{
    return spn.subspan( offset, count );
}

#endif // span_FEATURE( NON_MEMBER_FIRST_LAST_SUB_SPAN )

#if span_FEATURE( NON_MEMBER_FIRST_LAST_SUB_CONTAINER )

template< extent_t Count, class T >
constexpr auto
first( T & t ) -> decltype( make_span(t).template first<Count>() )
{
    return make_span( t ).template first<Count>();
}

template< class T >
constexpr auto
first( T & t, size_t count ) -> decltype( make_span(t).first(count) )
{
    return make_span( t ).first( count );
}

template< extent_t Count, class T >
constexpr auto
last( T & t ) -> decltype( make_span(t).template last<Count>() )
{
    return make_span(t).template last<Count>();
}

template< class T >
constexpr auto
last( T & t, extent_t count ) -> decltype( make_span(t).last(count) )
{
    return make_span( t ).last( count );
}

template< size_t Offset, extent_t Count = dynamic_extent, class T >
constexpr auto
subspan( T & t ) -> decltype( make_span(t).template subspan<Offset, Count>() )
{
    return make_span( t ).template subspan<Offset, Count>();
}

template< class T >
constexpr auto
subspan( T & t, size_t offset, extent_t count = dynamic_extent ) -> decltype( make_span(t).subspan(offset, count) )
{
    return make_span( t ).subspan( offset, count );
}

#endif // span_FEATURE( NON_MEMBER_FIRST_LAST_SUB_CONTAINER )

}  // namespace span_lite
}  // namespace collie

// make available in nonstd:

namespace collie {
using span_lite::make_span;

#if span_FEATURE( NON_MEMBER_FIRST_LAST_SUB_SPAN ) || ( span_FEATURE( NON_MEMBER_FIRST_LAST_SUB_CONTAINER ) )

using span_lite::first;
using span_lite::last;
using span_lite::subspan;

#endif // span_FEATURE( NON_MEMBER_FIRST_LAST_SUB_[SPAN|CONTAINER] )

}  // namespace collie

#endif // #if span_FEATURE_TO_STD( MAKE_SPAN )

#if span_FEATURE(BYTE_SPAN)

namespace collie {
namespace span_lite {

template< class T >
inline constexpr auto
byte_span( T & t ) noexcept -> span< std::byte, span_sizeof(T) >
{
    return span< std::byte, span_sizeof(t) >( reinterpret_cast< std::byte * >( &t ), span_sizeof(T) );
}

template< class T >
inline constexpr auto
byte_span( T const & t ) noexcept -> span< const std::byte, span_sizeof(T) >
{
    return span< const std::byte, span_sizeof(t) >( reinterpret_cast< std::byte const * >( &t ), span_sizeof(T) );
}

}  // namespace span_lite
}  // namespace collie

// make available in nonstd:

namespace collie {
using span_lite::byte_span;
}  // namespace collie

#endif // span_FEATURE( BYTE_SPAN )

# include <tuple>


namespace std {

// 26.7.X Tuple interface

// std::tuple_size<>:

    template<typename ElementType, collie::span_lite::extent_t Extent>
    class tuple_size<collie::span < ElementType, Extent>

    > : public integral_constant<size_t, static_cast<size_t>(Extent)> {
};

// std::tuple_size<>: Leave undefined for dynamic extent:

template<typename ElementType>
class tuple_size<collie::span<ElementType, collie::dynamic_extent> >;

// std::tuple_element<>:

template<size_t I, typename ElementType, collie::span_lite::extent_t Extent>
class tuple_element<I, collie::span<ElementType, Extent> > {
public:
    static_assert(Extent != collie::dynamic_extent && I < Extent,
                  "tuple_element<I,span>: dynamic extent or index out of range");
    using type = ElementType;
};

// std::get<>(), 2 variants:

template<size_t I, typename ElementType, collie::span_lite::extent_t Extent>
constexpr ElementType &get(collie::span<ElementType, Extent> &spn) noexcept {
    static_assert(Extent != collie::dynamic_extent && I < Extent, "get<>(span): dynamic extent or index out of range");
    return spn[I];
}

template<size_t I, typename ElementType, collie::span_lite::extent_t Extent>
constexpr ElementType const &get(collie::span<ElementType, Extent> const &spn) noexcept {
    static_assert(Extent != collie::dynamic_extent && I < Extent, "get<>(span): dynamic extent or index out of range");
    return spn[I];
}

} // end namespace std


#if !span_USES_STD_SPAN
span_RESTORE_WARNINGS()
#endif  // span_USES_STD_SPAN

#endif  // NONSTD_SPAN_HPP_INCLUDED
