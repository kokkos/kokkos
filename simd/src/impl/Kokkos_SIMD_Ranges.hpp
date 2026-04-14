// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_SIMD_RANGES_HPP
#define KOKKOS_SIMD_RANGES_HPP

#include <Kokkos_Macros.hpp>

// FIXME: Some of the compiler versions we support are compatible with standard
// library implementations which don't fully support C++20 ranges
#if defined(__cpp_lib_ranges) && (__cpp_lib_ranges >= 201911L)
#define KOKKOS_IMPL_COMPILER_SUPPORTS_CXX20_RANGES
#endif

#if defined(KOKKOS_IMPL_COMPILER_SUPPORTS_CXX20_RANGES)
#include <ranges>

namespace Kokkos::Experimental::Impl::Ranges {
using std::ranges::contiguous_range;
using std::ranges::data;
using std::ranges::range_value_t;
using std::ranges::sized_range;
}  // namespace Kokkos::Experimental::Impl::Ranges
#else
#include <iterator>

namespace Kokkos::Experimental::Impl::Ranges {

namespace {
// We need to rely on ADL but "using" declarations cannot be used inside a
// requires clause, we use an immediately-invoked lambda returning the requires
// clause as an alternative.
template <class R>
concept range = []() {
  using std::begin;
  using std::end;
  return requires(R& r) {
    begin(r);
    end(r);
  };
}();

inline constexpr auto begin = []<range R>(R&& r) {
  using std::begin;
  return begin(r);
};

template <class R>
using iterator_t = decltype(begin(std::declval<R&>()));

template <class R>
using range_reference_t = decltype(*std::declval<iterator_t<R>&>());
}  // namespace

inline constexpr auto data = []<range R>(R&& r) {
  using std::data;
  return data(r);
};

template <class R>
concept sized_range = range<R> && []() {
  using std::size;
  return requires(R& r) { size(r); };
}();

template <class R>
concept contiguous_range =
    range<R> &&
#if defined(__cpp_lib_concepts) && (__cpp_lib_concepts >= 202002L)
    std::contiguous_iterator<iterator_t<R> > && requires(R& r) {
      { data(r) } -> std::same_as<std::add_pointer_t<range_reference_t<R> > >;
    };
#else
    requires(R& r, iterator_t<R>& it) {
      ++it;
      --it;
      it += 2;
      it -= 2;
      *it;
      it[0];
      requires std::is_same_v<decltype(data(r)),
                              std::add_pointer_t<range_reference_t<R> > >;
    };
#endif

template <range R>
using range_value_t = typename std::iterator_traits<
    std::remove_cvref_t<iterator_t<R> > >::value_type;

}  // namespace Kokkos::Experimental::Impl::Ranges
#endif

#endif
