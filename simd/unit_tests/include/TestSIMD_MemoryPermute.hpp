// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_TEST_SIMD_MEMORY_PERMUTE_HPP
#define KOKKOS_TEST_SIMD_MEMORY_PERMUTE_HPP

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.simd;
import kokkos.simd_impl;
#else
#include <Kokkos_SIMD.hpp>
#endif
#include <SIMDTesting_Utilities.hpp>

template <typename T>
using simd_index_type =
    std::conditional_t<(sizeof(T) == 8), std::int64_t, std::int32_t>;

template <typename Abi, typename DataType, typename Flag>
inline void host_test_scatter_to(
    Kokkos::Experimental::basic_simd<DataType, Abi> init,
    Kokkos::Experimental::basic_simd_mask<simd_index_type<DataType>, Abi> mask,
    Kokkos::Experimental::basic_simd<simd_index_type<DataType>, Abi> indices,
    Flag flag) {
  using simd_type = Kokkos::Experimental::basic_simd<DataType, Abi>;
  using size_type = Kokkos::Experimental::Impl::simd_size_t;

  constexpr size_type size     = simd_type::size();
  constexpr DataType untouched = 7;

  alignas(size * sizeof(DataType)) DataType arr[size];

  DataType expected[size];
  for (size_type i = 0; i < size; ++i) expected[i] = untouched;

  for (size_type i = 0; i < size; ++i) {
    if (mask[i] &&
        Kokkos::Experimental::Impl::is_index_in_range(indices[i], size))
      expected[indices[i]] = init[i];
  }

  auto check_scattered = [&]() {
    gtest_checker checker;
    for (size_type i = 0; i < size; ++i) {
      checker.equality(expected[i], arr[i]);
    }
  };

  bool test_unchecked = true;
  bool test_unmasked  = true;
  for (size_type i = 0; i < size; ++i) {
    if (mask[i] &&
        !Kokkos::Experimental::Impl::is_index_in_range(indices[i], size))
      test_unchecked = false;
    if (!mask[i]) test_unmasked = false;
  }

  if (test_unchecked) {
    std::fill(std::begin(arr), std::end(arr), untouched);
    Kokkos::Experimental::unchecked_scatter_to(init, arr, mask, indices, flag);
    check_scattered();
  }
  if (test_unchecked && test_unmasked) {
    std::fill(std::begin(arr), std::end(arr), untouched);
    Kokkos::Experimental::unchecked_scatter_to(init, arr, indices, flag);
    check_scattered();
  }
  {
    std::fill(std::begin(arr), std::end(arr), untouched);
    Kokkos::Experimental::partial_scatter_to(init, arr, mask, indices, flag);
    check_scattered();
  }
  if (test_unmasked) {
    std::fill(std::begin(arr), std::end(arr), untouched);
    Kokkos::Experimental::partial_scatter_to(init, arr, indices, flag);
    check_scattered();
  }
}

template <typename Abi, typename DataType, typename Flag>
inline void host_test_gather_from(
    Kokkos::Experimental::basic_simd<DataType, Abi> init,
    Kokkos::Experimental::basic_simd_mask<simd_index_type<DataType>, Abi> mask,
    Kokkos::Experimental::basic_simd<simd_index_type<DataType>, Abi> indices,
    Flag flag) {
  using simd_type = Kokkos::Experimental::basic_simd<DataType, Abi>;
  using size_type = Kokkos::Experimental::Impl::simd_size_t;

  constexpr size_type size            = simd_type::size();
  constexpr bool is_host_fixed_native = std::is_same_v<
      Kokkos::Experimental::simd_abi::Impl::host_fixed_native<DataType>, Abi>;

  alignas(size * sizeof(DataType)) DataType arr[size];
  Kokkos::Experimental::simd_unchecked_store(init, arr, flag);

  auto check_gathered = [&](simd_type const& result) {
    gtest_checker checker;
    for (size_type i = 0; i < size; ++i) {
      auto expected = (mask[i] && Kokkos::Experimental::Impl::is_index_in_range(
                                      indices[i], size))
                          ? arr[indices[i]]
                          : DataType{};
      checker.equality(expected, result[i]);
    }
  };

  bool test_unchecked = true;
  bool test_unmasked  = true;
  for (size_type i = 0; i < size; ++i) {
    if (mask[i] &&
        !Kokkos::Experimental::Impl::is_index_in_range(indices[i], size))
      test_unchecked = false;
    if (!mask[i]) test_unmasked = false;
  }

  simd_type result;
  if (test_unchecked) {
    if constexpr (is_host_fixed_native) {
      result =
          Kokkos::Experimental::unchecked_gather_from(arr, mask, indices, flag);
    } else {
      result = Kokkos::Experimental::unchecked_gather_from<simd_type>(
          arr, mask, indices, flag);
    }
    check_gathered(result);
  }
  if (test_unchecked && test_unmasked) {
    if constexpr (is_host_fixed_native) {
      result = Kokkos::Experimental::unchecked_gather_from(arr, indices, flag);
    } else {
      result = Kokkos::Experimental::unchecked_gather_from<simd_type>(
          arr, indices, flag);
    }
    check_gathered(result);
  }
  {
    if constexpr (is_host_fixed_native) {
      result =
          Kokkos::Experimental::partial_gather_from(arr, mask, indices, flag);
    } else {
      result = Kokkos::Experimental::partial_gather_from<simd_type>(
          arr, mask, indices, flag);
    }
    check_gathered(result);
  }
  if (test_unmasked) {
    if constexpr (is_host_fixed_native) {
      result = Kokkos::Experimental::partial_gather_from(arr, indices, flag);
    } else {
      result = Kokkos::Experimental::partial_gather_from<simd_type>(
          arr, indices, flag);
    }
    check_gathered(result);
  }
}

template <class Abi, typename DataType>
inline void host_check_gather_scatter() {
  if constexpr (is_simd_avail_v<DataType, Abi>) {
    using simd_type  = Kokkos::Experimental::basic_simd<DataType, Abi>;
    using index_type = simd_index_type<DataType>;
    using indices_type =
        Kokkos::Experimental::basic_simd<simd_index_type<DataType>, Abi>;
    using mask_type = typename indices_type::mask_type;

    simd_type init(KOKKOS_LAMBDA(std::size_t i) { return (i + 1) * 11; });

    constexpr auto size = simd_type::size();

    indices_type reverse(
        KOKKOS_LAMBDA(std::size_t i) { return (size - 1) - i; });
    indices_type past_end(static_cast<index_type>(size));
    indices_type before_begin(static_cast<index_type>(-1));
    indices_type mixed(KOKKOS_LAMBDA(std::size_t i) {
      return (i % 2 == 0) ? static_cast<index_type>((size - 1) - i)
                          : static_cast<index_type>(size + i);
    });

    mask_type all(true);
    mask_type none(false);
    mask_type even(KOKKOS_LAMBDA(std::size_t i) { return i % 2 == 0; });

    auto check_all_cases = [&](auto flag) {
      host_test_gather_from(init, all, reverse, flag);
      host_test_gather_from(init, all, past_end, flag);
      host_test_gather_from(init, all, before_begin, flag);
      host_test_gather_from(init, all, mixed, flag);
      host_test_gather_from(init, even, reverse, flag);
      host_test_gather_from(init, none, past_end, flag);
      host_test_gather_from(init, even, mixed, flag);

      host_test_scatter_to(init, all, reverse, flag);
      host_test_scatter_to(init, all, past_end, flag);
      host_test_scatter_to(init, all, before_begin, flag);
      host_test_scatter_to(init, all, mixed, flag);
      host_test_scatter_to(init, even, reverse, flag);
      host_test_scatter_to(init, none, past_end, flag);
      host_test_scatter_to(init, even, mixed, flag);
    };

    check_all_cases(Kokkos::Experimental::simd_flag_default);
    check_all_cases(Kokkos::Experimental::simd_flag_aligned);
  }
}

template <typename Abi, typename... DataTypes>
inline void host_check_memory_permute_all_types(
    Kokkos::Experimental::Impl::data_types<DataTypes...>) {
  (host_check_gather_scatter<Abi, DataTypes>(), ...);
}

template <typename... Abis>
inline void host_check_memory_permute_all_abis(
    Kokkos::Experimental::Impl::abi_set<Abis...>) {
  using DataTypes = Kokkos::Experimental::Impl::data_type_set;
  (host_check_memory_permute_all_types<Abis>(DataTypes()), ...);
}

template <typename Abi, typename DataType, typename Flag>
KOKKOS_INLINE_FUNCTION void device_test_scatter_to(
    Kokkos::Experimental::basic_simd<DataType, Abi> init,
    Kokkos::Experimental::basic_simd_mask<simd_index_type<DataType>, Abi> mask,
    Kokkos::Experimental::basic_simd<simd_index_type<DataType>, Abi> indices,
    Flag flag) {
  using simd_type = Kokkos::Experimental::basic_simd<DataType, Abi>;
  using size_type = Kokkos::Experimental::Impl::simd_size_t;

  constexpr size_type size     = simd_type::size();
  constexpr DataType untouched = 7;

  alignas(size * sizeof(DataType)) DataType arr[size];

  DataType expected[size];
  for (size_type i = 0; i < size; ++i) expected[i] = untouched;

  for (size_type i = 0; i < size; ++i) {
    if (mask[i] &&
        Kokkos::Experimental::Impl::is_index_in_range(indices[i], size))
      expected[indices[i]] = init[i];
  }

  auto reset_array = KOKKOS_LAMBDA(DataType * a) {
    for (size_type i = 0; i < size; ++i) a[i] = untouched;
  };
  auto check_scattered = KOKKOS_LAMBDA(DataType * a, DataType * e) {
    kokkos_checker checker;
    for (size_type i = 0; i < size; ++i) checker.equality(e[i], a[i]);
  };

  bool test_unchecked = true;
  bool test_unmasked  = true;
  for (size_type i = 0; i < size; ++i) {
    if (mask[i] &&
        !Kokkos::Experimental::Impl::is_index_in_range(indices[i], size))
      test_unchecked = false;
    if (!mask[i]) test_unmasked = false;
  }

  if (test_unchecked) {
    reset_array(arr);
    Kokkos::Experimental::unchecked_scatter_to(init, arr, mask, indices, flag);
    check_scattered(arr, expected);
  }
  if (test_unchecked && test_unmasked) {
    reset_array(arr);
    Kokkos::Experimental::unchecked_scatter_to(init, arr, indices, flag);
    check_scattered(arr, expected);
  }
  {
    reset_array(arr);
    Kokkos::Experimental::partial_scatter_to(init, arr, mask, indices, flag);
    check_scattered(arr, expected);
  }
  if (test_unmasked) {
    reset_array(arr);
    Kokkos::Experimental::partial_scatter_to(init, arr, indices, flag);
    check_scattered(arr, expected);
  }
}

template <typename Abi, typename DataType, typename Flag>
KOKKOS_INLINE_FUNCTION void device_test_gather_from(
    Kokkos::Experimental::basic_simd<DataType, Abi> init,
    Kokkos::Experimental::basic_simd_mask<simd_index_type<DataType>, Abi> mask,
    Kokkos::Experimental::basic_simd<simd_index_type<DataType>, Abi> indices,
    Flag flag) {
  using simd_type = Kokkos::Experimental::basic_simd<DataType, Abi>;
  using size_type = Kokkos::Experimental::Impl::simd_size_t;

  constexpr size_type size            = simd_type::size();
  constexpr bool is_host_fixed_native = std::is_same_v<
      Kokkos::Experimental::simd_abi::Impl::host_fixed_native<DataType>, Abi>;

  alignas(size * sizeof(DataType)) DataType arr[size];
  Kokkos::Experimental::simd_unchecked_store(init, arr, flag);

  auto check_gathered = KOKKOS_LAMBDA(DataType * a, simd_type const& result) {
    kokkos_checker checker;
    for (size_type i = 0; i < size; ++i) {
      auto expected = (mask[i] && Kokkos::Experimental::Impl::is_index_in_range(
                                      indices[i], size))
                          ? a[indices[i]]
                          : DataType{};
      checker.equality(expected, result[i]);
    }
  };

  bool test_unchecked = true;
  bool test_unmasked  = true;
  for (size_type i = 0; i < size; ++i) {
    if (mask[i] &&
        !Kokkos::Experimental::Impl::is_index_in_range(indices[i], size))
      test_unchecked = false;
    if (!mask[i]) test_unmasked = false;
  }

  simd_type result;
  if (test_unchecked) {
    if constexpr (is_host_fixed_native) {
      result =
          Kokkos::Experimental::unchecked_gather_from(arr, mask, indices, flag);
    } else {
      result = Kokkos::Experimental::unchecked_gather_from<simd_type>(
          arr, mask, indices, flag);
    }
    check_gathered(arr, result);
  }
  if (test_unchecked && test_unmasked) {
    if constexpr (is_host_fixed_native) {
      result = Kokkos::Experimental::unchecked_gather_from(arr, indices, flag);
    } else {
      result = Kokkos::Experimental::unchecked_gather_from<simd_type>(
          arr, indices, flag);
    }
    check_gathered(arr, result);
  }
  {
    if constexpr (is_host_fixed_native) {
      result =
          Kokkos::Experimental::partial_gather_from(arr, mask, indices, flag);
    } else {
      result = Kokkos::Experimental::partial_gather_from<simd_type>(
          arr, mask, indices, flag);
    }
    check_gathered(arr, result);
  }
  if (test_unmasked) {
    if constexpr (is_host_fixed_native) {
      result = Kokkos::Experimental::partial_gather_from(arr, indices, flag);
    } else {
      result = Kokkos::Experimental::partial_gather_from<simd_type>(
          arr, indices, flag);
    }
    check_gathered(arr, result);
  }
}

template <class Abi, typename DataType>
KOKKOS_INLINE_FUNCTION void device_check_memory_permute() {
  if constexpr (is_simd_avail_v<DataType, Abi>) {
    using simd_type  = Kokkos::Experimental::basic_simd<DataType, Abi>;
    using index_type = simd_index_type<DataType>;
    using indices_type =
        Kokkos::Experimental::basic_simd<simd_index_type<DataType>, Abi>;
    using mask_type = typename indices_type::mask_type;

    simd_type init(KOKKOS_LAMBDA(std::size_t i) { return (i + 1) * 11; });

    constexpr auto size = simd_type::size();

    indices_type reverse(
        KOKKOS_LAMBDA(std::size_t i) { return (size - 1) - i; });
    indices_type past_end(static_cast<index_type>(size));
    indices_type before_begin(static_cast<index_type>(-1));
    indices_type mixed(KOKKOS_LAMBDA(std::size_t i) {
      return (i % 2 == 0) ? static_cast<index_type>((size - 1) - i)
                          : static_cast<index_type>(size + i);
    });

    mask_type all(true);
    mask_type none(false);
    mask_type even(KOKKOS_LAMBDA(std::size_t i) { return i % 2 == 0; });

    auto check_all_cases = [&](auto flag) {
      device_test_gather_from(init, all, reverse, flag);
      device_test_gather_from(init, all, past_end, flag);
      device_test_gather_from(init, all, before_begin, flag);
      device_test_gather_from(init, all, mixed, flag);
      device_test_gather_from(init, even, reverse, flag);
      device_test_gather_from(init, none, past_end, flag);
      device_test_gather_from(init, even, mixed, flag);

      device_test_scatter_to(init, all, reverse, flag);
      device_test_scatter_to(init, all, past_end, flag);
      device_test_scatter_to(init, all, before_begin, flag);
      device_test_scatter_to(init, all, mixed, flag);
      device_test_scatter_to(init, even, reverse, flag);
      device_test_scatter_to(init, none, past_end, flag);
      device_test_scatter_to(init, even, mixed, flag);
    };

    check_all_cases(Kokkos::Experimental::simd_flag_default);
    check_all_cases(Kokkos::Experimental::simd_flag_aligned);
  }
}

template <typename Abi, typename... DataTypes>
KOKKOS_INLINE_FUNCTION void device_check_memory_permute_all_types(
    Kokkos::Experimental::Impl::data_types<DataTypes...>) {
  (device_check_memory_permute<Abi, DataTypes>(), ...);
}

template <typename... Abis>
KOKKOS_INLINE_FUNCTION void device_check_memory_permute_all_abis(
    Kokkos::Experimental::Impl::abi_set<Abis...>) {
  using DataTypes = Kokkos::Experimental::Impl::data_type_set;
  (device_check_memory_permute_all_types<Abis>(DataTypes()), ...);
}

class simd_device_memory_permute_functor {
 public:
  KOKKOS_INLINE_FUNCTION void operator()(int) const {
    device_check_memory_permute_all_abis(
        Kokkos::Experimental::Impl::device_abi_set{});
  }
};

TEST(simd, host_memory_permute) {
  host_check_memory_permute_all_abis(
      Kokkos::Experimental::Impl::host_abi_set{});
}

TEST(simd, device_memory_permute) {
  Kokkos::parallel_for(1, simd_device_memory_permute_functor{});
  Kokkos::fence();
}

#endif
