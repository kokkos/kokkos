// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

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
#include <array>
#include <cmath>

namespace Kokkos {

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

template <typename... Properties>
struct MDRangePolicyInternal;

template <typename ExecSpace, typename P, typename... Properties>
struct MDRangePolicyInternal<ExecSpace, P, Properties...>
    : public PolicyTraits<P, Properties...> {
 public:
  using traits          = Impl::PolicyTraits<P, Properties...>;
  using execution_space = ExecSpace;
  using range_policy    = RangePolicy<Properties...>;

  using iteration_pattern = typename traits::iteration_pattern;
  using work_tag          = typename traits::work_tag;
  using launch_bounds     = typename traits::launch_bounds;
  using member_type       = typename range_policy::member_type;

  template <typename... OtherProperties>
  friend struct MDRangePolicyInternal;

  static constexpr int rank = iteration_pattern::rank;

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
  using index_type       = typename traits::index_type;
  using array_index_type = std::make_signed_t<index_type>;
  using point_type       = Kokkos::Array<array_index_type, rank>;
  using tile_type        = Kokkos::Array<array_index_type, rank>;

  execution_space m_space;

 public:
  int m_max_total_tile_size = std::numeric_limits<int>::max();
  Kokkos::Array<int, 3> m_max_threads_dimensions = {
      std::numeric_limits<int>::max(), std::numeric_limits<int>::max(),
      std::numeric_limits<int>::max()};

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

 public:
  template <typename OtherExecSpace, typename OtherP,
            typename... OtherProperties>
  MDRangePolicyInternal(const MDRangePolicyInternal<OtherExecSpace, OtherP,
                                                    OtherProperties...>& p)
      : traits(p),  // base class may contain data such as desired occupancy
        m_space(p.m_space),
        m_max_total_tile_size(p.m_max_total_tile_size),
        m_max_threads_dimensions(p.m_max_threads_dimensions),
        m_lower(p.m_lower),
        m_upper(p.m_upper),
        m_tile(p.m_tile),
        m_tile_end(p.m_tile_end),
        m_num_tiles(p.m_num_tiles),
        m_prod_tile_dims(p.m_prod_tile_dims),
        m_tune_tile_size(p.m_tune_tile_size) {}

  // Default constructor and assignment operators
  MDRangePolicyInternal()                                        = default;
  MDRangePolicyInternal(const MDRangePolicyInternal&)            = default;
  MDRangePolicyInternal(MDRangePolicyInternal&&)                 = default;
  MDRangePolicyInternal& operator=(const MDRangePolicyInternal&) = default;
  MDRangePolicyInternal& operator=(MDRangePolicyInternal&&)      = default;
  ~MDRangePolicyInternal()                                       = default;

  tile_type tile_size_recommended() const {
    tile_type recommended_tile_sizes{};
    int m_default_tile_size = 2;

    int rank_start = (inner_direction == Iterate::Right) ? rank - 1 : 0;
    int rank_end   = (inner_direction == Iterate::Right) ? -1 : rank;
    int iter_step  = (inner_direction == Iterate::Right) ? -1 : 1;
    array_index_type last_rank_length =
        m_upper[rank_start] - m_lower[rank_start];

    int prod_tile_size = 1;
    for (int i = rank_start; i != rank_end; i += iter_step) {
      int rank_tile_size = 1;
      if (prod_tile_size * m_default_tile_size <= m_max_total_tile_size) {
        rank_tile_size = m_default_tile_size;
      } else {
        rank_tile_size = 1;
      }
      if (i == rank_start) {
        rank_tile_size = std::max<int>(last_rank_length, 1);
      }
      prod_tile_size *= rank_tile_size;
      recommended_tile_sizes[i] = rank_tile_size;
    }
    return recommended_tile_sizes;
  }
};

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
    : public Impl::MDRangePolicyInternal<
          typename Impl::PolicyTraits<P, Properties...>::execution_space, P,
          Properties...> {
  using traits          = Kokkos::Impl::PolicyTraits<P, Properties...>;
  using internal_policy = Impl::MDRangePolicyInternal<
      typename Impl::PolicyTraits<P, Properties...>::execution_space, P,
      Properties...>;

  using range_policy = RangePolicy<Properties...>;

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

 public:
  using typename internal_policy::iteration_pattern;
  using typename internal_policy::launch_bounds;
  using typename internal_policy::member_type;
  using typename internal_policy::work_tag;

  static constexpr int rank = iteration_pattern::rank;
  static_assert(rank < 7, "Kokkos MDRangePolicy Error: Unsupported rank...");

  static constexpr auto outer_direction = internal_policy::outer_direction;
  static constexpr auto inner_direction = internal_policy::inner_direction;

  using typename internal_policy::array_index_type;
  using typename internal_policy::index_type;
  using typename internal_policy::point_type;
  using typename internal_policy::tile_type;

  KOKKOS_INLINE_FUNCTION const typename traits::execution_space& space() const {
    return this->m_space;
  }

  int max_total_tile_size() const { return this->m_max_total_tile_size; }

  void impl_change_tile_size(const point_type& tile) {
    this->m_tile = tile;
    init_helper();
  }

  bool impl_tune_tile_size() const { return this->m_tune_tile_size; }

 private:
  void init_helper() {
    tile_type default_tile = this->tile_size_recommended();
    this->m_num_tiles      = 1;
    this->m_prod_tile_dims = 1;

    int increment  = 1;
    int rank_start = 0;
    int rank_end   = rank;

    if constexpr (inner_direction == Iterate::Right) {
      increment  = -1;
      rank_start = rank - 1;
      rank_end   = -1;
    }

    for (int i = rank_start; i != rank_end; i += increment) {
      const index_type length = this->m_upper[i] - this->m_lower[i];

      if (this->m_upper[i] < this->m_lower[i]) {
        std::string msg =
            "Kokkos::MDRangePolicy bounds error: The lower bound (" +
            std::to_string(this->m_lower[i]) +
            ") is greater than its upper bound (" +
            std::to_string(this->m_upper[i]) + ") in dimension " +
            std::to_string(i) + ".\n";
        Kokkos::abort(msg.c_str());
      }

      // If tile size is not specified (default tile)
      if (this->m_tile[i] <= 0) {
        this->m_tune_tile_size = true;
        // Check if it fits within the limitation
        if (this->m_prod_tile_dims * default_tile[i] <=
            static_cast<index_type>(this->m_max_total_tile_size)) {
          this->m_tile[i] = default_tile[i];
        } else {
          // Try to fit within limitation by reducing tile size
          while (default_tile[i] > 1 &&
                 this->m_prod_tile_dims * default_tile[i] >
                     static_cast<index_type>(this->m_max_total_tile_size)) {
            default_tile[i] >>= 1;
          }
          if (default_tile[i] > 1) {
            this->m_tile[i] = default_tile[i];
          } else {
            this->m_tile[i] = 1;
          }
        }
      }

      this->m_tile_end[i] = static_cast<index_type>(
          (length + this->m_tile[i] - 1) / this->m_tile[i]);
      this->m_num_tiles *= this->m_tile_end[i];
      this->m_prod_tile_dims *= this->m_tile[i];
    }

    if (this->m_prod_tile_dims >
        static_cast<index_type>(this->m_max_total_tile_size)) {
      std::string msg =
          "Kokkos::MDRangePolicy tile dimensions error: Product of tile "
          "dimensions (" +
          std::to_string(static_cast<int>(this->m_prod_tile_dims)) +
          ") is greater than the maximum total tile size (" +
          std::to_string(static_cast<int>(this->m_max_total_tile_size)) +
          ") - choose smaller tile dims\n";
      Kokkos::abort(msg.c_str());
    }

    if (launch_bounds::maxTperB != 0 &&
        static_cast<index_type>(launch_bounds::maxTperB) <
            this->m_prod_tile_dims) {
      std::string msg =
          "Kokkos::MDRangePolicy tile dimensions error: Product of tile "
          "dimensions (" +
          std::to_string(static_cast<int>(this->m_prod_tile_dims)) +
          ") is greater than the maximum specified via LaunchBounds (" +
          std::to_string(launch_bounds::maxTperB) +
          ") - choose smaller tile dims\n";
      Kokkos::abort(msg.c_str());
    }
  }

 public:
  template <typename LT, std::size_t LN, typename UT, std::size_t UN,
            typename TT = array_index_type, std::size_t TN = rank,
            typename = std::enable_if_t<std::is_integral_v<LT> &&
                                        std::is_integral_v<UT> &&
                                        std::is_integral_v<TT>>>
  MDRangePolicy(const LT (&lower)[LN], const UT (&upper)[UN],
                const TT (&tile)[TN] = {})
      : MDRangePolicy(
            Impl::to_array_potentially_narrowing<
                index_type, decltype(internal_policy::m_lower)>(lower),
            Impl::to_array_potentially_narrowing<
                index_type, decltype(internal_policy::m_upper)>(upper),
            Impl::to_array_potentially_narrowing<
                index_type, decltype(internal_policy::m_tile)>(tile)) {
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
            Impl::to_array_potentially_narrowing<
                index_type, decltype(internal_policy::m_lower)>(lower),
            Impl::to_array_potentially_narrowing<
                index_type, decltype(internal_policy::m_upper)>(upper),
            Impl::to_array_potentially_narrowing<
                index_type, decltype(internal_policy::m_tile)>(tile)) {
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
      : internal_policy() {
    this->m_space = work_space;
    this->m_lower = lower;
    this->m_upper = upper;
    this->m_tile  = tile;
    init_helper();
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
            Impl::to_array_potentially_narrowing<
                index_type, decltype(internal_policy::m_lower)>(lower),
            Impl::to_array_potentially_narrowing<
                index_type, decltype(internal_policy::m_upper)>(upper),
            Impl::to_array_potentially_narrowing<
                index_type, decltype(internal_policy::m_tile)>(tile)) {}

  template <typename OtherP, typename... OtherProperties>
  MDRangePolicy(const MDRangePolicy<OtherP, OtherProperties...>& other)
      : internal_policy(other) {}

  MDRangePolicy(const internal_policy& p) : internal_policy(p) {}

  MDRangePolicy() = default;

  MDRangePolicy(const Impl::PolicyUpdate, const MDRangePolicy& other,
                typename traits::execution_space space)
      : MDRangePolicy(other) {
    this->m_space = std::move(space);
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
