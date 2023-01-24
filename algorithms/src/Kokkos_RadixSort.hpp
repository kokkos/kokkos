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

#ifndef KOKKOS_RADIXSORT_HPP_
#define KOKKOS_RADIXSORT_HPP_
#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_RADIXSORT
#endif

#include <Kokkos_Core.hpp>

#ifdef KOKKOS_IMPL_HALF_TYPE_DEFINED
#include <Kokkos_Half.hpp>
#endif

namespace Kokkos {
namespace Experimental {

// Roughly based on the algorithm described at
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda

namespace Impl {
template <typename SubjectType>
struct equivalent_size_integral_type {
  using type = std::make_unsigned_t<SubjectType>;
};

#if KOKKOS_HAS_8_BIT_FLOAT
template <>
struct equivalent_size_integral_type<quarter_t> {
  using type = uint8_t;
};
#endif

#if !KOKKOS_HALF_T_IS_FLOAT
template <>
struct equivalent_size_integral_type<::Kokkos::Experimental::half_t> {
  using type = uint16_t;
};
template <>
struct equivalent_size_integral_type<::Kokkos::Impl::half_impl_t::type> {
  using type = uint16_t;
};
#endif

#if !KOKKOS_BHALF_T_IS_FLOAT
template <>
struct equivalent_size_integral_type<::Kokkos::Experimental::bhalf_t> {
  using type = uint16_t;
};
template <>
struct equivalent_size_integral_type<::Kokkos::Impl::bhalf_impl_t::type> {
  using type = uint16_t;
};
#endif

template <>
struct equivalent_size_integral_type<float> {
  using type = uint32_t;
};

template <>
struct equivalent_size_integral_type<double> {
  using type = uint64_t;
};

// long double will be trouble, because we'll need to account for 8
// vs 10 vs 16 byte representations and storage

}  // namespace Impl

template <typename FloatingPointType>
using equivalent_size_integral_type =
    typename Impl::equivalent_size_integral_type<FloatingPointType>::type;

/// Gets key values from elements of a View. Handles any integral type
/// and floating point types of 16-64 bits.
///
/// Can limit consideration and hence sorting effort to just the
/// least-significant BitWidth bits of each element. Such a limitation
/// is non-sensical for floating-point numbers.
template <int BitWidth, typename KeyView>
struct KeyFromView {
  static constexpr int num_bits = BitWidth;

  using key_value_type = typename KeyView::value_type;

  // Always load and operate on unsigned integer types to avoid
  // - Logic overhead in loading as floating point and then reinterpreting as
  // integral
  // - Potential UB issues in shifting signed-integer sign bits
  using key_integral_type = equivalent_size_integral_type<key_value_type>;

  static_assert(std::is_integral_v<key_integral_type>,
                "Only integral or integral-interpretable FP key types are "
                "presently supported");

  // Keep a reference to the View passed, even if we're using an
  // integral alias, to ensure correct lifetime behavior
  KeyView const keys;

  KOKKOS_FUNCTION
  KeyFromView(KeyView const& k) : keys(k) {}
  KOKKOS_FUNCTION
  KeyFromView(KeyView const& k, std::integral_constant<int, BitWidth>)
      : keys(k) {}

  // i: index of the key to get
  // bit: which bit, with 0 indicating the least-significant
  KOKKOS_INLINE_FUNCTION
  auto operator()(int i, int bit) const {
    // I'd rather use a fully aliasing view here ...
    auto key = *reinterpret_cast<key_integral_type*>(&keys(i));
    auto h   = key >> bit;

    // Handle the sign bit of signed 2's-complement indicating low values
    if constexpr (std::is_integral_v<key_value_type> &&
                  std::is_signed_v<key_value_type>) {
      if (bit == 8 * sizeof(key_value_type) - 1) {
        return h & 0b1u;
      }
    }

    // Handle the sign-magnitude representation of FP
    if constexpr (std::is_floating_point_v<key_value_type>) {
      auto sign_bit_pos = 8 * sizeof(key_value_type) - 1;
      auto sign_bit     = key >> sign_bit_pos;
      if (bit == sign_bit_pos) {
        return h & 0b1u;
      } else {
        return ~(h ^ sign_bit) & 0b1u;
      }
    }

    return ~h & 0b1u;
  }

  KOKKOS_FUNCTION
  int getNumBits() { return num_bits; }
};

template <typename KeyView>
KeyFromView(KeyView const&)
    ->KeyFromView<8 * sizeof(typename KeyView::value_type), KeyView>;

template <typename KeyView, int BitWidth>
KeyFromView(KeyView const&, std::integral_constant<int, BitWidth>)
    ->KeyFromView<BitWidth, KeyView>;

template <typename KeyType, typename IndexType = ::std::uint32_t>
class RadixSorter {
 public:
  RadixSorter() = default;
  explicit RadixSorter(std::size_t n)
      : m_key_scratch("radix_sort_key_scratch", n),
        m_index_old("radix_sort_index", n),
        m_index_new("radix_sort_index_scratch", n),
        m_scan("radix_sort_scan", n),
        m_bits("radix_sort_bits", n) {}

  // Generate and store the permutation induced by the keys, without
  // modifying their initial order
  template <typename ExecutionSpace>
  void create_indirection_vector(ExecutionSpace const& exec,
                                 View<KeyType*> keys) {
    auto key_functor = KeyFromView{keys};
    const auto n     = keys.extent(0);

    create_indirection_vector(exec, key_functor, n);
  }

  template <typename ExecutionSpace, typename KeyFunctor>
  void create_indirection_vector(ExecutionSpace const& exec,
                                 KeyFunctor key_functor, size_t n) {
    RangePolicy<ExecutionSpace, Kokkos::IndexType<IndexType>> policy(exec, 0,
                                                                     n);

    // Initialize m_index_old, since it will be read from in the first
    // iteration's call to step()
    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(IndexType i) { m_index_old(i) = i; });

    int num_bits = key_functor.getNumBits();
    for (int i = 0; i < num_bits; ++i) {
      step(policy, key_functor, i, m_index_old);
      permute_by_scan<IndexType>(policy, {m_index_new, m_index_old});
    }
  }

  // Directly re-arrange the entries of keys, optionally storing the permutation
  template <bool store_permutation = false, typename ExecutionSpace>
  void sort(ExecutionSpace const& exec, View<KeyType*> keys) {
    // Almost identical to create_indirection_array, except actually permute the
    // input
    const auto n = keys.extent(0);
    RangePolicy<ExecutionSpace, Kokkos::IndexType<IndexType>> policy(exec, 0,
                                                                     n);

    if constexpr (store_permutation) {
      Kokkos::parallel_for(
          policy, KOKKOS_LAMBDA(IndexType i) { m_index_old(i) = i; });
    }

    using KeyFunctor = decltype(KeyFromView{keys});

    for (int i = 0; i < KeyFunctor::num_bits; ++i) {
      KeyFunctor key_functor = KeyFromView{keys};

      step(
          policy, key_functor, i, KOKKOS_LAMBDA(size_t id) { return id; });
      if constexpr (store_permutation) {
        permute_by_scan<KeyType, IndexType>(policy, {m_key_scratch, keys},
                                            {m_index_new, m_index_old});
      } else {
        permute_by_scan<KeyType>(policy, {m_key_scratch, keys});
      }

      // Number of bits is always even, and we know on odd numbered
      // iterations we are reading from m_key_scratch and writing to keys
      // So when this loop ends, keys will contain the results
    }
  }

  // Directly re-arrange the entries of keys, optionally storing the permutation
  template <bool store_permutation = false, typename U, typename ExecutionSpace>
  void sortByKeys(ExecutionSpace const& exec, View<KeyType*> keys,
                  View<U*> values) {
    // Almost identical to create_indirection_array, except actually permute the
    // input
    const auto n = keys.extent(0);
    RangePolicy<ExecutionSpace, Kokkos::IndexType<IndexType>> policy(exec, 0,
                                                                     n);

    if constexpr (store_permutation) {
      Kokkos::parallel_for(
          policy, KOKKOS_LAMBDA(IndexType i) { m_index_old(i) = i; });
    }

    auto values_scratch =
        View<U*>(view_alloc(exec, "radix_sorter_values_scratch",
                            Kokkos::WithoutInitializing),
                 n);

    using KeyFunctor = decltype(KeyFromView{keys});

    for (int i = 0; i < KeyFunctor::num_bits; ++i) {
      KeyFunctor key_functor = KeyFromView{keys};

      step(
          policy, key_functor, i, KOKKOS_LAMBDA(size_t id) { return id; });
      if constexpr (store_permutation) {
        permute_by_scan<KeyType, U, IndexType>(policy, {m_key_scratch, keys},
                                               {values_scratch, values},
                                               {m_index_new, m_index_old});
      } else {
        permute_by_scan<KeyType, U>(policy, {m_key_scratch, keys},
                                    {values_scratch, values});
      }

      // Number of bits is always even, and we know on odd numbered
      // iterations we are reading from m_key_scratch/values_scratch and writing
      // to keys/values. So, when this loop ends, keys/values will contain the
      // results
    }
  }

  template <typename ExecutionSpace>
  void apply_permutation(ExecutionSpace const& exec, View<KeyType*> v) {
    parallel_for(
        RangePolicy<ExecutionSpace, Kokkos::IndexType<IndexType>>(exec, 0,
                                                                  v.extent(0)),
        KOKKOS_LAMBDA(IndexType i) { m_key_scratch(i) = v(m_index_old(i)); });
    deep_copy(exec, v, m_key_scratch);
  }

 private:
  template <typename... U, typename Policy>
  void permute_by_scan(Policy policy,
                       Kokkos::pair<View<U*>&, View<U*>&>... views) {
    parallel_for(
        policy, KOKKOS_LAMBDA(IndexType i) {
          auto n           = m_scan.extent(0);
          const auto total = m_scan(n - 1) + m_bits(n - 1);
          auto t           = i - m_scan(i) + total;
          auto new_idx     = m_bits(i) ? m_scan(i) : t;
          [[maybe_unused]] int dummy[sizeof...(U)] = {
              (views.first(new_idx) = views.second(i), 0)...};
        });
    using std::swap;
    [[maybe_unused]] int dummy[sizeof...(U)] = {
        (swap(views.first, views.second), 0)...};
  }

  template <typename Policy, typename KeyFunctor, typename Permutation>
  void step(Policy policy, KeyFunctor getKeyBit, std::uint32_t shift,
            const Permutation& permutation) {
    parallel_for(
        policy, KOKKOS_LAMBDA(IndexType i) {
          auto key_bit = getKeyBit(permutation(i), shift);

          m_bits(i) = key_bit;
          m_scan(i) = m_bits(i);
        });

    parallel_scan(
        policy, KOKKOS_LAMBDA(IndexType i, size_t & _x, bool _final) {
          auto val = m_scan(i);

          if (_final) m_scan(i) = _x;

          _x += val;
        });
  }

  View<KeyType*> m_key_scratch;
  View<IndexType*> m_index_old;
  View<IndexType*> m_index_new;
  View<size_t*> m_scan;
  View<unsigned char*> m_bits;
};

}  // namespace Experimental
}  // namespace Kokkos

#ifdef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_RADIXSORT
#undef KOKKOS_IMPL_PUBLIC_INCLUDE
#undef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_RADIXSORT
#endif
#endif
