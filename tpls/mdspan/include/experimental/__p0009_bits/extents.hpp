//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#pragma once
#include "dynamic_extent.hpp"
#include "utility.hpp"

#ifdef __cpp_lib_span
#include <span>
#endif
#include <array>
#include <type_traits>

#include <cassert>
#include <cinttypes>

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace detail {

// Function used to check compatibility of extents in converting constructor
// can't be a private member function for some reason.
template <size_t... Extents, size_t... OtherExtents>
MDSPAN_INLINE_FUNCTION
constexpr std::integral_constant<bool, false> impl_check_compatible_extents(
    std::integral_constant<bool, false>,
    std::integer_sequence<size_t, Extents...>,
    std::integer_sequence<size_t, OtherExtents...>) noexcept {
  return {};
}

// This helper prevents ICE's on MSVC.
template <size_t Lhs, size_t Rhs>
struct impl_compare_extent_compatible : std::integral_constant<bool,
     Lhs == dynamic_extent ||
     Rhs == dynamic_extent ||
     Lhs == Rhs>
{};

template <size_t... Extents, size_t... OtherExtents>
MDSPAN_INLINE_FUNCTION
constexpr std::integral_constant<
    bool, MDSPAN_IMPL_FOLD_AND(impl_compare_extent_compatible<Extents, OtherExtents>::value)>
impl_check_compatible_extents(
    std::integral_constant<bool, true>,
    std::integer_sequence<size_t, Extents...>,
    std::integer_sequence<size_t, OtherExtents...>) noexcept {
  return {};
}

template<class IndexType, class ... Arguments>
MDSPAN_INLINE_FUNCTION
constexpr bool are_valid_indices() {
    return
      MDSPAN_IMPL_FOLD_AND(std::is_convertible<Arguments, IndexType>::value) &&
      MDSPAN_IMPL_FOLD_AND(std::is_nothrow_constructible<IndexType, Arguments>::value);
}

// ------------------------------------------------------------------
// ------------ static_array ----------------------------------------
// ------------------------------------------------------------------

// array like class which provides an array of static values with get
// function and operator [].

// Implementation of Static Array with recursive implementation of get.
template <size_t R, class T, T... Extents> struct static_array_impl;

template <size_t R, class T, T FirstExt, T... Extents>
struct static_array_impl<R, T, FirstExt, Extents...> {
  MDSPAN_INLINE_FUNCTION
  constexpr static T get(size_t r) {
    if (r == R)
      return FirstExt;
    else
      return static_array_impl<R + 1, T, Extents...>::get(r);
  }
  template <size_t r> MDSPAN_INLINE_FUNCTION constexpr static T get() {
#if MDSPAN_HAS_CXX_17
    if constexpr (r == R)
      return FirstExt;
    else
      return static_array_impl<R + 1, T, Extents...>::template get<r>();
#else
    get(r);
#endif
  }
};

// End the recursion
template <size_t R, class T, T FirstExt>
struct static_array_impl<R, T, FirstExt> {
  MDSPAN_INLINE_FUNCTION
  constexpr static T get(size_t) { return FirstExt; }
  template <size_t> MDSPAN_INLINE_FUNCTION constexpr static T get() {
    return FirstExt;
  }
};

// Don't start recursion if size 0
template <class T> struct static_array_impl<0, T> {
  MDSPAN_INLINE_FUNCTION
  constexpr static T get(size_t) { return T(); }
  template <size_t> MDSPAN_INLINE_FUNCTION constexpr static T get() {
    return T();
  }
};

// Static array, provides get<r>(), get(r) and operator[r]
template <class T, T... Values> struct static_array:
  public static_array_impl<0, T, Values...>  {

public:
  using value_type = T;

  MDSPAN_INLINE_FUNCTION
  constexpr static size_t size() { return sizeof...(Values); }
};


// ------------------------------------------------------------------
// ------------ index_sequence_scan ---------------------------------
// ------------------------------------------------------------------

// index_sequence_scan takes compile time values and provides get(r)
// and get<r>() which return the sum of the first r-1 values.

// Recursive implementation for get
template <size_t R, size_t... Values> struct index_sequence_scan_impl;

template <size_t R, size_t FirstVal, size_t... Values>
struct index_sequence_scan_impl<R, FirstVal, Values...> {
  MDSPAN_INLINE_FUNCTION
  constexpr static size_t get(size_t r) {
    if (r > R)
      return FirstVal + index_sequence_scan_impl<R + 1, Values...>::get(r);
    else
      return 0;
  }
};

template <size_t R, size_t FirstVal>
struct index_sequence_scan_impl<R, FirstVal> {
#if defined(__NVCC__) || defined(__NVCOMPILER) ||                              \
    defined(MDSPAN_IMPL_COMPILER_INTEL)
  // NVCC warns about pointless comparison with 0 for R==0 and r being const
  // evaluatable and also 0.
  MDSPAN_INLINE_FUNCTION
  constexpr static size_t get(size_t r) {
    return static_cast<int64_t>(R) > static_cast<int64_t>(r) ? FirstVal : 0;
  }
#else
  MDSPAN_INLINE_FUNCTION
  constexpr static size_t get(size_t r) { return R > r ? FirstVal : 0; }
#endif
};
template <> struct index_sequence_scan_impl<0> {
  MDSPAN_INLINE_FUNCTION
  constexpr static size_t get(size_t) { return 0; }
};

// ------------------------------------------------------------------
// ------------ possibly_empty_array  -------------------------------
// ------------------------------------------------------------------

// array like class which provides get function and operator [], and
// has a specialization for the size 0 case.
// This is needed to make the maybe_static_array be truly empty, for
// all static values.

template <class T, size_t N> struct possibly_empty_array {
  T vals[N]{};
  MDSPAN_INLINE_FUNCTION
  constexpr T &operator[](size_t r) { return vals[r]; }
  MDSPAN_INLINE_FUNCTION
  constexpr const T &operator[](size_t r) const { return vals[r]; }
};

template <class T> struct possibly_empty_array<T, 0> {
  MDSPAN_INLINE_FUNCTION
  constexpr T operator[](size_t) { return T(); }
  MDSPAN_INLINE_FUNCTION
  constexpr const T operator[](size_t) const { return T(); }
};

// ------------------------------------------------------------------
// ------------ maybe_static_array ----------------------------------
// ------------------------------------------------------------------

// array like class which has a mix of static and runtime values but
// only stores the runtime values.
// The type of the static and the runtime values can be different.
// The position of a dynamic value is indicated through a tag value.
template <class TDynamic, class TStatic, TStatic dyn_tag, TStatic... Values>
struct maybe_static_array {

  static_assert(std::is_convertible<TStatic, TDynamic>::value, "maybe_static_array: TStatic must be convertible to TDynamic");
  static_assert(std::is_convertible<TDynamic, TStatic>::value, "maybe_static_array: TDynamic must be convertible to TStatic");

private:
  // Static values member
  using static_vals_t = static_array<TStatic, Values...>;
  constexpr static size_t m_size = sizeof...(Values);
  constexpr static size_t m_size_dynamic =
      MDSPAN_IMPL_FOLD_PLUS_RIGHT((Values == dyn_tag), 0);

  // Dynamic values member
  MDSPAN_IMPL_NO_UNIQUE_ADDRESS possibly_empty_array<TDynamic, m_size_dynamic>
      m_dyn_vals;

  // static mapping of indices to the position in the dynamic values array
  using dyn_map_t = index_sequence_scan_impl<0, static_cast<size_t>(Values == dyn_tag)...>;
public:

  // two types for static and dynamic values
  using value_type = TDynamic;
  using static_value_type = TStatic;
  // tag value indicating dynamic value
  constexpr static static_value_type tag_value = dyn_tag;

  constexpr maybe_static_array() = default;

  // constructor for all static values
  // TODO: add precondition check?
  MDSPAN_TEMPLATE_REQUIRES(class... Vals,
                           /* requires */ ((m_size_dynamic == 0) &&
                                           (sizeof...(Vals) > 0)))
  MDSPAN_INLINE_FUNCTION
  constexpr maybe_static_array(Vals...) : m_dyn_vals{} {}

  // constructors from dynamic values only
  MDSPAN_TEMPLATE_REQUIRES(class... DynVals,
                           /* requires */ (sizeof...(DynVals) ==
                                               m_size_dynamic &&
                                           m_size_dynamic > 0))
  MDSPAN_INLINE_FUNCTION
  constexpr maybe_static_array(DynVals... vals)
      : m_dyn_vals{static_cast<TDynamic>(vals)...} {}


  MDSPAN_TEMPLATE_REQUIRES(class T, size_t N,
                           /* requires */ (N == m_size_dynamic && N > 0))
  MDSPAN_INLINE_FUNCTION
  constexpr maybe_static_array(const std::array<T, N> &vals) {
    for (size_t r = 0; r < N; r++)
      m_dyn_vals[r] = static_cast<TDynamic>(vals[r]);
  }

  MDSPAN_TEMPLATE_REQUIRES(class T, size_t N,
                           /* requires */ (N == m_size_dynamic && N == 0))
  MDSPAN_INLINE_FUNCTION
  constexpr maybe_static_array(const std::array<T, N> &) : m_dyn_vals{} {}

#ifdef __cpp_lib_span
  MDSPAN_TEMPLATE_REQUIRES(class T, size_t N,
                           /* requires */ (N == m_size_dynamic && N > 0))
  MDSPAN_INLINE_FUNCTION
  constexpr maybe_static_array(const std::span<T, N> &vals) {
    for (size_t r = 0; r < N; r++)
      m_dyn_vals[r] = static_cast<TDynamic>(vals[r]);
  }

  MDSPAN_TEMPLATE_REQUIRES(class T, size_t N,
                           /* requires */ (N == m_size_dynamic && N == 0))
  MDSPAN_INLINE_FUNCTION
  constexpr maybe_static_array(const std::span<T, N> &) : m_dyn_vals{} {}
#endif

  // constructors from all values
  MDSPAN_TEMPLATE_REQUIRES(class... DynVals,
                           /* requires */ (sizeof...(DynVals) !=
                                               m_size_dynamic &&
                                           m_size_dynamic > 0))
  MDSPAN_INLINE_FUNCTION
  constexpr maybe_static_array(DynVals... vals)
    : m_dyn_vals{} {
    static_assert((sizeof...(DynVals) == m_size), "Invalid number of values.");
    TDynamic values[m_size]{static_cast<TDynamic>(vals)...};
    for (size_t r = 0; r < m_size; r++) {
      TStatic static_val = static_vals_t::get(r);
      if (static_val == dyn_tag) {
        m_dyn_vals[dyn_map_t::get(r)] = values[r];
      }
// Precondition check
#ifdef MDSPAN_DEBUG
      else {
        assert(values[r] == static_cast<TDynamic>(static_val));
      }
#endif
    }
  }

  MDSPAN_TEMPLATE_REQUIRES(
      class T, size_t N,
      /* requires */ (N != m_size_dynamic && m_size_dynamic > 0))
  MDSPAN_INLINE_FUNCTION
  constexpr maybe_static_array(const std::array<T, N> &vals) {
    static_assert((N == m_size), "Invalid number of values.");
// Precondition check
#ifdef MDSPAN_DEBUG
    assert(N == m_size);
#endif
    for (size_t r = 0; r < m_size; r++) {
      TStatic static_val = static_vals_t::get(r);
      if (static_val == dyn_tag) {
        m_dyn_vals[dyn_map_t::get(r)] = static_cast<TDynamic>(vals[r]);
      }
// Precondition check
#ifdef MDSPAN_DEBUG
      else {
        assert(static_cast<TDynamic>(vals[r]) ==
               static_cast<TDynamic>(static_val));
      }
#endif
    }
  }

#ifdef __cpp_lib_span
  MDSPAN_TEMPLATE_REQUIRES(
      class T, size_t N,
      /* requires */ (N != m_size_dynamic && m_size_dynamic > 0))
  MDSPAN_INLINE_FUNCTION
  constexpr maybe_static_array(const std::span<T, N> &vals) {
    static_assert((N == m_size) || (m_size == dynamic_extent));
#ifdef MDSPAN_DEBUG
    assert(N == m_size);
#endif
    for (size_t r = 0; r < m_size; r++) {
      TStatic static_val = static_vals_t::get(r);
      if (static_val == dyn_tag) {
        m_dyn_vals[dyn_map_t::get(r)] = static_cast<TDynamic>(vals[r]);
      }
#ifdef MDSPAN_DEBUG
      else {
        assert(static_cast<TDynamic>(vals[r]) ==
               static_cast<TDynamic>(static_val));
      }
#endif
    }
  }
#endif

  // access functions
  MDSPAN_INLINE_FUNCTION
  constexpr static TStatic static_value(size_t r) { return static_vals_t::get(r); }

  MDSPAN_INLINE_FUNCTION
  constexpr TDynamic value(size_t r) const {
    TStatic static_val = static_vals_t::get(r);

    // FIXME: workaround for nvhpc OpenACC compiler bug
    TStatic dyn_tag_copy = dyn_tag;
    return static_val == dyn_tag_copy ? m_dyn_vals[dyn_map_t::get(r)]
                                     : static_cast<TDynamic>(static_val);
  }
  MDSPAN_INLINE_FUNCTION
  constexpr TDynamic operator[](size_t r) const { return value(r); }


  // observers
  MDSPAN_INLINE_FUNCTION
  constexpr static size_t size() { return m_size; }
  MDSPAN_INLINE_FUNCTION
  constexpr static size_t size_dynamic() { return m_size_dynamic; }
};

} // namespace detail
} // namespace MDSPAN_IMPL_STANDARD_NAMESPACE

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {

// ------------------------------------------------------------------
// ------------ extents ---------------------------------------------
// ------------------------------------------------------------------

// Class to describe the extents of a multi dimensional array.
// Used by mdspan, mdarray and layout mappings.
// See ISO C++ standard [mdspan.extents]

template <class IndexType, size_t... Extents> class extents {
public:
  // typedefs for integral types used
  using index_type = IndexType;
  using size_type = std::make_unsigned_t<index_type>;
  using rank_type = size_t;

  static_assert(std::is_integral<index_type>::value && !std::is_same<index_type, bool>::value,
                MDSPAN_IMPL_STANDARD_NAMESPACE_STRING "::extents::index_type must be a signed or unsigned integer type");
private:
  constexpr static rank_type m_rank = sizeof...(Extents);
  constexpr static rank_type m_rank_dynamic =
      MDSPAN_IMPL_FOLD_PLUS_RIGHT((Extents == dynamic_extent), /* + ... + */ 0);

  // internal storage type using maybe_static_array
  using vals_t =
      detail::maybe_static_array<IndexType, size_t, dynamic_extent, Extents...>;
  MDSPAN_IMPL_NO_UNIQUE_ADDRESS vals_t m_vals;

public:
  // [mdspan.extents.obs], observers of multidimensional index space
  MDSPAN_INLINE_FUNCTION
  constexpr static rank_type rank() noexcept { return m_rank; }
  MDSPAN_INLINE_FUNCTION
  constexpr static rank_type rank_dynamic() noexcept { return m_rank_dynamic; }

  MDSPAN_INLINE_FUNCTION
  constexpr index_type extent(rank_type r) const noexcept { return m_vals.value(r); }
  MDSPAN_INLINE_FUNCTION
  constexpr static size_t static_extent(rank_type r) noexcept {
    return vals_t::static_value(r);
  }

  // [mdspan.extents.cons], constructors
  MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr extents() noexcept = default;

  // Construction from just dynamic or all values.
  // Precondition check is deferred to maybe_static_array constructor
  MDSPAN_TEMPLATE_REQUIRES(
      class... OtherIndexTypes,
      /* requires */ (
          MDSPAN_IMPL_FOLD_AND(MDSPAN_IMPL_TRAIT(std::is_convertible, OtherIndexTypes,
                                         index_type) /* && ... */) &&
          MDSPAN_IMPL_FOLD_AND(MDSPAN_IMPL_TRAIT(std::is_nothrow_constructible, index_type,
                                         OtherIndexTypes) /* && ... */) &&
          (sizeof...(OtherIndexTypes) == m_rank ||
           sizeof...(OtherIndexTypes) == m_rank_dynamic)))
  MDSPAN_INLINE_FUNCTION
  constexpr explicit extents(OtherIndexTypes... dynvals) noexcept
      : m_vals(static_cast<index_type>(dynvals)...) {}

  MDSPAN_TEMPLATE_REQUIRES(
      class OtherIndexType, size_t N,
      /* requires */
      (
          MDSPAN_IMPL_TRAIT(std::is_convertible, const OtherIndexType&, index_type) &&
          MDSPAN_IMPL_TRAIT(std::is_nothrow_constructible, index_type,
              const OtherIndexType&) &&
          (N == m_rank || N == m_rank_dynamic)))
  MDSPAN_INLINE_FUNCTION
  MDSPAN_CONDITIONAL_EXPLICIT(N != m_rank_dynamic)
  constexpr extents(const std::array<OtherIndexType, N> &exts) noexcept
      : m_vals(std::move(exts)) {}

#ifdef __cpp_lib_span
  MDSPAN_TEMPLATE_REQUIRES(
      class OtherIndexType, size_t N,
      /* requires */
      (MDSPAN_IMPL_TRAIT(std::is_convertible, const OtherIndexType&, index_type) &&
       MDSPAN_IMPL_TRAIT(std::is_nothrow_constructible, index_type, const OtherIndexType&) &&
       (N == m_rank || N == m_rank_dynamic)))
  MDSPAN_INLINE_FUNCTION
  MDSPAN_CONDITIONAL_EXPLICIT(N != m_rank_dynamic)
  constexpr extents(const std::span<OtherIndexType, N> &exts) noexcept
      : m_vals(std::move(exts)) {}
#endif

private:
  // Function to construct extents storage from other extents.
  // With C++ 17 the first two variants could be collapsed using if constexpr
  // in which case you don't need all the requires clauses.
  // in C++ 14 mode that doesn't work due to infinite recursion
  MDSPAN_TEMPLATE_REQUIRES(
      size_t DynCount, size_t R, class OtherExtents, class... DynamicValues,
      /* requires */ ((R < m_rank) && (static_extent(R) == dynamic_extent)))
  MDSPAN_INLINE_FUNCTION
  constexpr
  vals_t impl_construct_vals_from_extents(std::integral_constant<size_t, DynCount>,
                                       std::integral_constant<size_t, R>,
                                       const OtherExtents &exts,
                                       DynamicValues... dynamic_values) noexcept {
    return impl_construct_vals_from_extents(
        std::integral_constant<size_t, DynCount + 1>(),
        std::integral_constant<size_t, R + 1>(), exts, dynamic_values...,
        exts.extent(R));
  }

  MDSPAN_TEMPLATE_REQUIRES(
      size_t DynCount, size_t R, class OtherExtents, class... DynamicValues,
      /* requires */ ((R < m_rank) && (static_extent(R) != dynamic_extent)))
  MDSPAN_INLINE_FUNCTION
  constexpr
  vals_t impl_construct_vals_from_extents(std::integral_constant<size_t, DynCount>,
                                       std::integral_constant<size_t, R>,
                                       const OtherExtents &exts,
                                       DynamicValues... dynamic_values) noexcept {
    return impl_construct_vals_from_extents(
        std::integral_constant<size_t, DynCount>(),
        std::integral_constant<size_t, R + 1>(), exts, dynamic_values...);
  }

  MDSPAN_TEMPLATE_REQUIRES(
      size_t DynCount, size_t R, class OtherExtents, class... DynamicValues,
      /* requires */ ((R == m_rank) && (DynCount == m_rank_dynamic)))
  MDSPAN_INLINE_FUNCTION
  constexpr
  vals_t impl_construct_vals_from_extents(std::integral_constant<size_t, DynCount>,
                                       std::integral_constant<size_t, R>,
                                       const OtherExtents &,
                                       DynamicValues... dynamic_values) noexcept {
    return vals_t{static_cast<index_type>(dynamic_values)...};
  }

public:

  // Converting constructor from other extents specializations
    MDSPAN_TEMPLATE_REQUIRES(
        class OtherIndexType, size_t... OtherExtents,
        /* requires */
        (
            /* multi-stage check to protect from invalid pack expansion when sizes
            don't match? */
            decltype(detail::impl_check_compatible_extents(
              // using: sizeof...(Extents) == sizeof...(OtherExtents) as the second argument fails with MSVC+NVCC with some obscure expansion error
              // MSVC: 19.38.33133 NVCC: 12.0
              std::integral_constant<bool, extents<int, Extents...>::rank() == extents<int, OtherExtents...>::rank()>{},
              std::integer_sequence<size_t, Extents...>{},
              std::integer_sequence<size_t, OtherExtents...>{}))::value
      )
  )
  MDSPAN_INLINE_FUNCTION
  MDSPAN_CONDITIONAL_EXPLICIT((((Extents != dynamic_extent) &&
                                (OtherExtents == dynamic_extent)) ||
                               ...) ||
                              (std::numeric_limits<index_type>::max() <
                               std::numeric_limits<OtherIndexType>::max()))
  constexpr extents(const extents<OtherIndexType, OtherExtents...> &other) noexcept
      : m_vals(impl_construct_vals_from_extents(
            std::integral_constant<size_t, 0>(),
            std::integral_constant<size_t, 0>(), other)) {}

  // Comparison operator
  template <class OtherIndexType, size_t... OtherExtents>
  MDSPAN_INLINE_FUNCTION friend constexpr bool
  operator==(const extents &lhs,
             const extents<OtherIndexType, OtherExtents...> &rhs) noexcept {
    return
      rank() == extents<OtherIndexType, OtherExtents...>::rank() &&
      detail::rankwise_equal(detail::with_rank<rank()>{}, rhs, lhs, detail::extent);
  }

#if !(MDSPAN_HAS_CXX_20)
  template <class OtherIndexType, size_t... OtherExtents>
  MDSPAN_INLINE_FUNCTION friend constexpr bool
  operator!=(extents const &lhs,
             extents<OtherIndexType, OtherExtents...> const &rhs) noexcept {
    return !(lhs == rhs);
  }
#endif
};

// Recursive helper classes to implement dextents alias for extents
namespace detail {

template <class IndexType, size_t Rank,
          class Extents = ::MDSPAN_IMPL_STANDARD_NAMESPACE::extents<IndexType>>
struct impl_make_dextents;

template <class IndexType, size_t Rank, size_t... ExtentsPack>
struct impl_make_dextents<
    IndexType, Rank, ::MDSPAN_IMPL_STANDARD_NAMESPACE::extents<IndexType, ExtentsPack...>>
{
  using type = typename impl_make_dextents<
      IndexType, Rank - 1,
      ::MDSPAN_IMPL_STANDARD_NAMESPACE::extents<IndexType,
                                                ::MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent,
                                                ExtentsPack...>>::type;
};

template <class IndexType, size_t... ExtentsPack>
struct impl_make_dextents<
    IndexType, 0, ::MDSPAN_IMPL_STANDARD_NAMESPACE::extents<IndexType, ExtentsPack...>>
{
  using type = ::MDSPAN_IMPL_STANDARD_NAMESPACE::extents<IndexType, ExtentsPack...>;
};

} // end namespace detail

// [mdspan.extents.dextents], alias template
template <class IndexType, size_t Rank>
using dextents = typename detail::impl_make_dextents<IndexType, Rank>::type;

// Deduction guide for extents
#if defined(MDSPAN_IMPL_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION)
template <class... IndexTypes>
extents(IndexTypes...)
    -> extents<size_t,
               ((void) sizeof(IndexTypes), ::MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent)...>;
#endif

// Helper type traits for identifying a class as extents.
namespace detail {

template <class T> struct impl_is_extents : ::std::false_type {};

template <class IndexType, size_t... ExtentsPack>
struct impl_is_extents<::MDSPAN_IMPL_STANDARD_NAMESPACE::extents<IndexType, ExtentsPack...>>
    : ::std::true_type {};

template <class T>
#if MDSPAN_HAS_CXX_17
inline
#else
static
#endif
constexpr bool impl_is_extents_v = impl_is_extents<T>::value;

template<class InputIndexType, class ExtentsIndexType>
MDSPAN_INLINE_FUNCTION
constexpr void
check_lower_bound(InputIndexType user_index,
                  ExtentsIndexType /* current_extent */,
                  std::true_type /* is_signed */)
{
  (void) user_index; // prevent unused variable warning
#ifdef MDSPAN_DEBUG
  assert(static_cast<ExtentsIndexType>(user_index) >= 0);
#endif
}

template<class InputIndexType, class ExtentsIndexType>
MDSPAN_INLINE_FUNCTION
constexpr void
check_lower_bound(InputIndexType /* user_index */,
                  ExtentsIndexType /* current_extent */,
                  std::false_type /* is_signed */)
{}

template<class InputIndexType, class ExtentsIndexType>
MDSPAN_INLINE_FUNCTION
constexpr void
check_upper_bound(InputIndexType user_index,
                  ExtentsIndexType current_extent)
{
  (void) user_index; // prevent unused variable warnings
  (void) current_extent;
#ifdef MDSPAN_DEBUG
  assert(static_cast<ExtentsIndexType>(user_index) < current_extent);
#endif
}

// Returning true to use AND fold instead of comma
// CPP14 mode doesn't like the use of void expressions
// with the way the MDSPAN_IMPL_FOLD_AND is set up
template<class InputIndex, class ExtentsIndexType>
MDSPAN_INLINE_FUNCTION
constexpr bool
check_one_index(InputIndex user_index,
                ExtentsIndexType current_extent)
{
  check_lower_bound(user_index, current_extent,
    std::integral_constant<bool, std::is_signed<ExtentsIndexType>::value>{});
  check_upper_bound(user_index, current_extent);
  return true;
}

template<size_t ... RankIndices,
         class ExtentsIndexType, size_t ... Exts,
         class ... Indices>
MDSPAN_INLINE_FUNCTION
constexpr void
check_all_indices_helper(std::index_sequence<RankIndices...>,
                         const extents<ExtentsIndexType, Exts...>& exts,
                         Indices... indices)
{
  // Suppress warning about statement has no effect
  (void) MDSPAN_IMPL_FOLD_AND(
    (check_one_index(indices, exts.extent(RankIndices)))
  );
}

template<class ExtentsIndexType, size_t ... Exts,
         class ... Indices>
MDSPAN_INLINE_FUNCTION
constexpr void
check_all_indices(const extents<ExtentsIndexType, Exts...>& exts,
                  Indices... indices)
{
  check_all_indices_helper(std::make_index_sequence<sizeof...(Indices)>(),
                           exts, indices...);
}

} // namespace detail
} // namespace MDSPAN_IMPL_STANDARD_NAMESPACE
