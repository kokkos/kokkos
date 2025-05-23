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

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#include <Kokkos_Macros.hpp>
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif
#ifndef KOKKOS_CORE_EXP_MD_RANGE_POLICY_HPP
#define KOKKOS_CORE_EXP_MD_RANGE_POLICY_HPP

#include <initializer_list>

#include <Kokkos_Layout.hpp>
#include <Kokkos_Rank.hpp>
#include <Kokkos_Array.hpp>
#include <impl/KokkosExp_Host_IterateTile.hpp>
#include <Kokkos_ExecPolicy.hpp>
#include <type_traits>
#include <cmath>

namespace Kokkos {

// ------------------------------------------------------------------ //
// Moved to Kokkos_Layout.hpp for more general accessibility
/*
enum class Iterate
{
  Default, // Default for the device
  Left,    // Left indices stride fastest
  Right,   // Right indices stride fastest
};
*/

template <typename ExecSpace>
struct default_outer_direction {
  using type                     = Iterate;
  static constexpr Iterate value = Iterate::Right;
};

template <typename ExecSpace>
struct default_inner_direction {
  using type                     = Iterate;
  static constexpr Iterate value = Iterate::Right;
};

namespace Impl {
// NOTE the comparison below is encapsulated to silent warnings about pointless
// comparison of unsigned integer with zero
template <class T>
constexpr std::enable_if_t<!std::is_signed_v<T>, bool>
is_less_than_value_initialized_variable(T) {
  return false;
}

template <class T>
constexpr std::enable_if_t<std::is_signed_v<T>, bool>
is_less_than_value_initialized_variable(T arg) {
  return arg < T{};
}

// Checked narrowing conversion that calls abort if the cast changes the value
template <class To, class From>
constexpr To checked_narrow_cast(From arg, std::size_t idx) {
  constexpr const bool is_different_signedness =
      (std::is_signed_v<To> != std::is_signed_v<From>);
  auto const ret = static_cast<To>(arg);  // NOLINT(bugprone-signed-char-misuse)
  if (static_cast<From>(ret) != arg ||
      (is_different_signedness &&
       is_less_than_value_initialized_variable(arg) !=
           is_less_than_value_initialized_variable(ret))) {
    auto msg =
        "Kokkos::MDRangePolicy bound type error: an unsafe implicit conversion "
        "is performed on a bound (" +
        std::to_string(arg) + ") in dimension (" + std::to_string(idx) +
        "), which may not preserve its original value.\n";
    Kokkos::abort(msg.c_str());
  }
  return ret;
}
// NOTE prefer C array U[M] to std::initalizer_list<U> so that the number of
// elements can be deduced (https://stackoverflow.com/q/40241370)
// NOTE for some unfortunate reason the policy bounds are stored as signed
// integer arrays (point_type which is Kokkos::Array<std::int64_t>) so we
// specify the index type (actual policy index_type from the traits) and check
// ahead of time that narrowing conversions will be safe.
template <class IndexType, class Array, class U, std::size_t M>
constexpr Array to_array_potentially_narrowing(const U (&init)[M]) {
  using T = typename Array::value_type;
  Array a{};
  constexpr std::size_t N = a.size();
  static_assert(M <= N);
  auto* ptr = a.data();
  // NOTE equivalent to
  // std::transform(std::begin(init), std::end(init), a.data(),
  //                [](U x) { return static_cast<T>(x); });
  // except that std::transform is not constexpr.
  for (std::size_t i = 0; i < M; ++i) {
    *ptr++ = checked_narrow_cast<T>(init[i], i);
    (void)checked_narrow_cast<IndexType>(init[i], i);  // see note above
  }
  return a;
}

// NOTE Making a copy even when std::is_same<Array, Kokkos::Array<U, M>>::value
// is true to reduce code complexity.  You may change this if you have a good
// reason to.  Intentionally not enabling std::array at this time but this may
// change too.
template <class IndexType, class NVCC_WONT_LET_ME_CALL_YOU_Array, class U,
          std::size_t M>
constexpr NVCC_WONT_LET_ME_CALL_YOU_Array to_array_potentially_narrowing(
    Kokkos::Array<U, M> const& other) {
  using T = typename NVCC_WONT_LET_ME_CALL_YOU_Array::value_type;
  NVCC_WONT_LET_ME_CALL_YOU_Array a{};
  constexpr std::size_t N = a.size();
  static_assert(M <= N);
  for (std::size_t i = 0; i < M; ++i) {
    a[i] = checked_narrow_cast<T>(other[i], i);
    (void)checked_narrow_cast<IndexType>(other[i], i);  // see note above
  }
  return a;
}

struct TileSizeProperties {
  int max_threads;
  int default_largest_tile_size;
  int default_tile_size;
  int max_total_tile_size;
};

template <typename ExecutionSpace>
TileSizeProperties get_tile_size_properties(const ExecutionSpace&) {
  // Host settings
  TileSizeProperties properties;
  properties.max_threads               = std::numeric_limits<int>::max();
  properties.default_largest_tile_size = 0;
  properties.default_tile_size         = 2;
  properties.max_total_tile_size       = std::numeric_limits<int>::max();
  return properties;
}

}  // namespace Impl

// multi-dimensional iteration pattern
template <typename... Properties>
struct MDRangePolicy;

// Note: If MDRangePolicy has a primary template, implicit CTAD (deduction
// guides) are generated -> MDRangePolicy<> by some compilers, which is
// incorrect.  By making it a template specialization instead, no implicit CTAD
// is generated.  This works because there has to be at least one property
// specified (which is Rank<...>); otherwise, we'd get the static_assert
// "Kokkos::Error: MD iteration pattern not defined".  This template
// specialization uses <P, Properties...> in all places for correctness.
template <typename P, typename... Properties>
struct MDRangePolicy<P, Properties...>
    : public Kokkos::Impl::PolicyTraits<P, Properties...> {
  using traits       = Kokkos::Impl::PolicyTraits<P, Properties...>;
  using range_policy = RangePolicy<P, Properties...>;

  typename traits::execution_space m_space;

  using impl_range_policy =
      RangePolicy<typename traits::execution_space,
                  typename traits::schedule_type, typename traits::index_type>;

  using execution_policy =
      MDRangePolicy<P, Properties...>;  // needed for is_execution_policy
                                        // interrogation

  template <class... OtherProperties>
  friend struct MDRangePolicy;

  static_assert(!std::is_void_v<typename traits::iteration_pattern>,
                "Kokkos Error: MD iteration pattern not defined");

  using iteration_pattern = typename traits::iteration_pattern;
  using work_tag          = typename traits::work_tag;
  using launch_bounds     = typename traits::launch_bounds;
  using member_type       = typename range_policy::member_type;

  static constexpr int rank = iteration_pattern::rank;
  static_assert(rank < 7, "Kokkos MDRangePolicy Error: Unsupported rank...");

  using index_type       = typename traits::index_type;
  using array_index_type = std::int64_t;
  using point_type = Kokkos::Array<array_index_type, rank>;  // was index_type
  using tile_type  = Kokkos::Array<array_index_type, rank>;
  // If point_type or tile_type is not templated on a signed integral type (if
  // it is unsigned), then if user passes in intializer_list of
  // runtime-determined values of signed integral type that are not const will
  // receive a compiler error due to an invalid case for implicit conversion -
  // "conversion from integer or unscoped enumeration type to integer type that
  // cannot represent all values of the original, except where source is a
  // constant expression whose value can be stored exactly in the target type"
  // This would require the user to either pass a matching index_type parameter
  // as template parameter to the MDRangePolicy or static_cast the individual
  // values

  point_type m_lower          = {};
  point_type m_upper          = {};
  tile_type m_tile            = {};
  point_type m_tile_end       = {};
  index_type m_num_tiles      = 1;
  index_type m_prod_tile_dims = 1;
  bool m_tune_tile_size       = false;

  static constexpr auto outer_direction =
      (iteration_pattern::outer_direction != Iterate::Default)
          ? iteration_pattern::outer_direction
          : default_outer_direction<typename traits::execution_space>::value;

  static constexpr auto inner_direction =
      iteration_pattern::inner_direction != Iterate::Default
          ? iteration_pattern::inner_direction
          : default_inner_direction<typename traits::execution_space>::value;

  static constexpr auto Right = Iterate::Right;
  static constexpr auto Left  = Iterate::Left;

  KOKKOS_INLINE_FUNCTION const typename traits::execution_space& space() const {
    return m_space;
  }

  MDRangePolicy() = default;

  template <typename LT, std::size_t LN, typename UT, std::size_t UN,
            typename TT = array_index_type, std::size_t TN = rank,
            typename = std::enable_if_t<std::is_integral_v<LT> &&
                                        std::is_integral_v<UT> &&
                                        std::is_integral_v<TT>>>
  MDRangePolicy(const LT (&lower)[LN], const UT (&upper)[UN],
                const TT (&tile)[TN] = {})
      : MDRangePolicy(
            Impl::to_array_potentially_narrowing<index_type, decltype(m_lower)>(
                lower),
            Impl::to_array_potentially_narrowing<index_type, decltype(m_upper)>(
                upper),
            Impl::to_array_potentially_narrowing<index_type, decltype(m_tile)>(
                tile)) {
    static_assert(
        LN == rank && UN == rank && TN <= rank,
        "MDRangePolicy: Constructor initializer lists have wrong size");
  }

  template <typename LT, std::size_t LN, typename UT, std::size_t UN,
            typename TT = array_index_type, std::size_t TN = rank,
            typename = std::enable_if_t<std::is_integral_v<LT> &&
                                        std::is_integral_v<UT> &&
                                        std::is_integral_v<TT>>>
  MDRangePolicy(const typename traits::execution_space& work_space,
                const LT (&lower)[LN], const UT (&upper)[UN],
                const TT (&tile)[TN] = {})
      : MDRangePolicy(
            work_space,
            Impl::to_array_potentially_narrowing<index_type, decltype(m_lower)>(
                lower),
            Impl::to_array_potentially_narrowing<index_type, decltype(m_upper)>(
                upper),
            Impl::to_array_potentially_narrowing<index_type, decltype(m_tile)>(
                tile)) {
    static_assert(
        LN == rank && UN == rank && TN <= rank,
        "MDRangePolicy: Constructor initializer lists have wrong size");
  }

  // NOTE: Keeping these two constructor despite the templated constructors
  // from Kokkos arrays for backwards compability to allow construction from
  // double-braced initializer lists.
  MDRangePolicy(point_type const& lower, point_type const& upper,
                tile_type const& tile = tile_type{})
      : MDRangePolicy(typename traits::execution_space(), lower, upper, tile) {}

  MDRangePolicy(const typename traits::execution_space& work_space,
                point_type const& lower, point_type const& upper,
                tile_type const& tile = tile_type{})
      : m_space(work_space), m_lower(lower), m_upper(upper), m_tile(tile) {
    init_helper(Impl::get_tile_size_properties(work_space));
  }

  template <typename T, std::size_t NT = rank,
            typename = std::enable_if_t<std::is_integral_v<T>>>
  MDRangePolicy(Kokkos::Array<T, rank> const& lower,
                Kokkos::Array<T, rank> const& upper,
                Kokkos::Array<T, NT> const& tile = Kokkos::Array<T, NT>{})
      : MDRangePolicy(typename traits::execution_space(), lower, upper, tile) {}

  template <typename T, std::size_t NT = rank,
            typename = std::enable_if_t<std::is_integral_v<T>>>
  MDRangePolicy(const typename traits::execution_space& work_space,
                Kokkos::Array<T, rank> const& lower,
                Kokkos::Array<T, rank> const& upper,
                Kokkos::Array<T, NT> const& tile = Kokkos::Array<T, NT>{})
      : MDRangePolicy(
            work_space,
            Impl::to_array_potentially_narrowing<index_type, decltype(m_lower)>(
                lower),
            Impl::to_array_potentially_narrowing<index_type, decltype(m_upper)>(
                upper),
            Impl::to_array_potentially_narrowing<index_type, decltype(m_tile)>(
                tile)) {}

  template <class... OtherProperties>
  MDRangePolicy(const MDRangePolicy<OtherProperties...> p)
      : traits(p),  // base class may contain data such as desired occupancy
        m_space(p.m_space),
        m_lower(p.m_lower),
        m_upper(p.m_upper),
        m_tile(p.m_tile),
        m_tile_end(p.m_tile_end),
        m_num_tiles(p.m_num_tiles),
        m_prod_tile_dims(p.m_prod_tile_dims),
        m_tune_tile_size(p.m_tune_tile_size) {}

  void impl_change_tile_size(const point_type& tile) {
    m_tile = tile;
    init_helper(Impl::get_tile_size_properties(m_space));
  }
  bool impl_tune_tile_size() const { return m_tune_tile_size; }

  tile_type tile_size_recommended() const {
    tile_type rec_tile_sizes = {};

    for (std::size_t i = 0; i < rec_tile_sizes.size(); ++i) {
      rec_tile_sizes[i] = tile_size_recommended(i);
    }
    return rec_tile_sizes;
  }

  int max_total_tile_size() const {
    return Impl::get_tile_size_properties(m_space).max_total_tile_size;
  }

 private:
  int tile_size_recommended(const int tile_rank) const {
    auto properties = Impl::get_tile_size_properties(m_space);
    int last_rank   = (inner_direction == Iterate::Right) ? rank - 1 : 0;
    int rank_acc =
        (inner_direction == Iterate::Right) ? tile_rank + 1 : tile_rank - 1;
    int rec_tile_size = (std::pow(properties.default_tile_size, rank_acc) <
                         properties.max_total_tile_size)
                            ? properties.default_tile_size
                            : 1;

    if (tile_rank == last_rank) {
      rec_tile_size = tile_size_last_rank(
          properties, m_upper[last_rank] - m_lower[last_rank]);
    }
    return rec_tile_size;
  }

  int tile_size_last_rank(const Impl::TileSizeProperties properties,
                          const index_type length) const {
    return properties.default_largest_tile_size == 0
               ? std::max<int>(length, 1)
               : properties.default_largest_tile_size;
  }

  void init_helper(Impl::TileSizeProperties properties) {
    m_prod_tile_dims = 1;
    int increment    = 1;
    int rank_start   = 0;
    int rank_end     = rank;
    if (inner_direction == Iterate::Right) {
      increment  = -1;
      rank_start = rank - 1;
      rank_end   = -1;
    }

    for (int i = rank_start; i != rank_end; i += increment) {
      const index_type length = m_upper[i] - m_lower[i];

      if (m_upper[i] < m_lower[i]) {
        std::string msg =
            "Kokkos::MDRangePolicy bounds error: The lower bound (" +
            std::to_string(m_lower[i]) + ") is greater than its upper bound (" +
            std::to_string(m_upper[i]) + ") in dimension " + std::to_string(i) +
            ".\n";
#if !defined(KOKKOS_ENABLE_DEPRECATED_CODE_4)
        Kokkos::abort(msg.c_str());
#elif defined(KOKKOS_ENABLE_DEPRECATION_WARNINGS)
        Kokkos::Impl::log_warning(msg);
#endif
      }

      if (m_tile[i] <= 0) {
        m_tune_tile_size = true;
        if ((inner_direction == Iterate::Right && (i < rank - 1)) ||
            (inner_direction == Iterate::Left && (i > 0))) {
          if (m_prod_tile_dims * properties.default_tile_size <
              static_cast<index_type>(properties.max_total_tile_size)) {
            m_tile[i] = properties.default_tile_size;
          } else {
            m_tile[i] = 1;
          }
        } else {
          m_tile[i] = tile_size_last_rank(properties, length);
        }
      }
      m_tile_end[i] =
          static_cast<index_type>((length + m_tile[i] - 1) / m_tile[i]);
      m_num_tiles *= m_tile_end[i];
      m_prod_tile_dims *= m_tile[i];
    }
    if (m_prod_tile_dims > static_cast<index_type>(properties.max_threads)) {
      printf(" Product of tile dimensions exceed maximum limit: %d\n",
             static_cast<int>(properties.max_threads));
      Kokkos::abort(
          "ExecSpace Error: MDRange tile dims exceed maximum number "
          "of threads per block - choose smaller tile dims");
    }
  }
};

template <typename LT, size_t N, typename UT>
MDRangePolicy(const LT (&)[N], const UT (&)[N]) -> MDRangePolicy<Rank<N>>;

template <typename LT, size_t N, typename UT, typename TT, size_t TN>
MDRangePolicy(const LT (&)[N], const UT (&)[N], const TT (&)[TN])
    -> MDRangePolicy<Rank<N>>;

template <typename LT, size_t N, typename UT>
MDRangePolicy(DefaultExecutionSpace const&, const LT (&)[N], const UT (&)[N])
    -> MDRangePolicy<Rank<N>>;

template <typename LT, size_t N, typename UT, typename TT, size_t TN>
MDRangePolicy(DefaultExecutionSpace const&, const LT (&)[N], const UT (&)[N],
              const TT (&)[TN]) -> MDRangePolicy<Rank<N>>;

template <typename ES, typename LT, size_t N, typename UT,
          typename = std::enable_if_t<is_execution_space_v<ES>>>
MDRangePolicy(ES const&, const LT (&)[N], const UT (&)[N])
    -> MDRangePolicy<ES, Rank<N>>;

template <typename ES, typename LT, size_t N, typename UT, typename TT,
          size_t TN, typename = std::enable_if_t<is_execution_space_v<ES>>>
MDRangePolicy(ES const&, const LT (&)[N], const UT (&)[N], const TT (&)[TN])
    -> MDRangePolicy<ES, Rank<N>>;

template <typename T, size_t N>
MDRangePolicy(Array<T, N> const&, Array<T, N> const&) -> MDRangePolicy<Rank<N>>;

template <typename T, size_t N, size_t NT>
MDRangePolicy(Array<T, N> const&, Array<T, N> const&, Array<T, NT> const&)
    -> MDRangePolicy<Rank<N>>;

template <typename T, size_t N>
MDRangePolicy(DefaultExecutionSpace const&, Array<T, N> const&,
              Array<T, N> const&) -> MDRangePolicy<Rank<N>>;

template <typename T, size_t N, size_t NT>
MDRangePolicy(DefaultExecutionSpace const&, Array<T, N> const&,
              Array<T, N> const&, Array<T, NT> const&)
    -> MDRangePolicy<Rank<N>>;

template <typename ES, typename T, size_t N,
          typename = std::enable_if_t<is_execution_space_v<ES>>>
MDRangePolicy(ES const&, Array<T, N> const&, Array<T, N> const&)
    -> MDRangePolicy<ES, Rank<N>>;

template <typename ES, typename T, size_t N, size_t NT,
          typename = std::enable_if_t<is_execution_space_v<ES>>>
MDRangePolicy(ES const&, Array<T, N> const&, Array<T, N> const&,
              Array<T, NT> const&) -> MDRangePolicy<ES, Rank<N>>;

}  // namespace Kokkos

#endif  // KOKKOS_CORE_EXP_MD_RANGE_POLICY_HPP
