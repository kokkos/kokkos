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

template <typename Abi, typename DataType, typename Flag>
inline void host_test_scatter_to(
    Kokkos::Experimental::basic_simd<DataType, Abi> init,
    Kokkos::Experimental::basic_simd_mask<int, Abi> mask,
    Kokkos::Experimental::basic_simd<int, Abi> indices, Flag flag) {
  using size_type = Kokkos::Experimental::Impl::simd_size_t;
  using mask_type = decltype(mask);

  auto check_scattered = [&](DataType(&arr)[init.size()], mask_type m = mask_type{true}) {
    gtest_checker checker;
    for (size_type i = 0; i < init.size(); ++i) {
      auto expected = (m[i]) ? init[i] : arr[indices[i]];
      auto found    = arr[indices[i]];
      checker.equality(expected, found);
    }
  };

  DataType result[init.size()];
  {
    Kokkos::Experimental::unchecked_scatter_to(init, result, indices, flag);
    check_scattered(result);
  }
  {
    Kokkos::Experimental::unchecked_scatter_to(init, result, mask, indices,
                                               flag);
    check_scattered(result, mask);
  }
  {
    Kokkos::Experimental::partial_scatter_to(init, result, indices, flag);
    check_scattered(result);
  }
  {
    Kokkos::Experimental::partial_scatter_to(init, result, mask, indices, flag);
    check_scattered(result, mask);
  }
}

template <typename Abi, typename DataType, typename Flag>
inline void host_test_gather_from(
    Kokkos::Experimental::basic_simd<DataType, Abi> init,
    Kokkos::Experimental::basic_simd_mask<int, Abi> mask,
    Kokkos::Experimental::basic_simd<int, Abi> indices, Flag flag) {
  using simd_type = Kokkos::Experimental::basic_simd<DataType, Abi>;
  using size_type = Kokkos::Experimental::Impl::simd_size_t;
  using mask_type = decltype(mask);

  auto check_gathered = [&](DataType(&arr)[init.size()], simd_type& result, mask_type m = mask_type{true}) {
    gtest_checker checker;
    for (size_type i = 0; i < result.size(); ++i) {
      auto expected = (m[i]) ? arr[indices[i]] : DataType{};
      auto found    = result[i];
      checker.equality(expected, found);
    }
  };

  simd_type result;
  DataType arr[init.size()];
  Kokkos::Experimental::simd_unchecked_store(init, arr, flag);
  {
    if constexpr (std::is_same_v<Kokkos::Experimental::simd_abi::Impl::
                                     host_fixed_native<DataType>,
                                 Abi>) {
      result =
          Kokkos::Experimental::unchecked_gather_from(arr, indices, flag);
    } else {
      result =
          Kokkos::Experimental::unchecked_gather_from<simd_type>(arr, indices, flag);
    }
    check_gathered(arr, result);
  }
  {
    if constexpr (std::is_same_v<Kokkos::Experimental::simd_abi::Impl::
                                     host_fixed_native<DataType>,
                                 Abi>) {
      result =
          Kokkos::Experimental::unchecked_gather_from(arr, mask, indices, flag);
    } else {
      result =
          Kokkos::Experimental::unchecked_gather_from<simd_type>(arr, mask, indices, flag);
    }
    check_gathered(arr, result, mask);
  }
  {
    if constexpr (std::is_same_v<Kokkos::Experimental::simd_abi::Impl::
                                     host_fixed_native<DataType>,
                                 Abi>) {
      result =
          Kokkos::Experimental::partial_gather_from(arr, indices, flag);
    } else {
      result =
          Kokkos::Experimental::partial_gather_from<simd_type>(arr, indices, flag);
    }
    check_gathered(arr, result);
  }
  {
    if constexpr (std::is_same_v<Kokkos::Experimental::simd_abi::Impl::
                                     host_fixed_native<DataType>,
                                 Abi>) {
      result =
          Kokkos::Experimental::partial_gather_from(arr, mask, indices, flag);
    } else {
      result =
          Kokkos::Experimental::partial_gather_from<simd_type>(arr, mask, indices, flag);
    }
    check_gathered(arr, result, mask);
  }
}

template <class Abi, typename DataType>
inline void host_check_gather_scatter() {
  if constexpr (is_simd_avail_v<DataType, Abi>) {
    using simd_type  = Kokkos::Experimental::basic_simd<DataType, Abi>;
    using index_type = Kokkos::Experimental::basic_simd<int, Abi>;
    using mask_type  = typename index_type::mask_type;

    simd_type init([=](std::size_t i) { return (i + 1) * 11; });
    mask_type mask([=](std::size_t i) { return i % 2 == 0; });
    index_type reverse(
        [=](std::size_t i) { return (simd_type::size() - 1) - i; });

    host_test_scatter_to(init, mask, reverse,
                        Kokkos::Experimental::simd_flag_default);
    host_test_scatter_to(init, mask, reverse,
                        Kokkos::Experimental::simd_flag_aligned);

    host_test_gather_from(init, mask, reverse,
                          Kokkos::Experimental::simd_flag_default);
    host_test_gather_from(init, mask, reverse,
                          Kokkos::Experimental::simd_flag_aligned);
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
KOKKOS_INLINE_FUNCTION
void device_test_scatter_to(
    Kokkos::Experimental::basic_simd<DataType, Abi> init,
    Kokkos::Experimental::basic_simd_mask<int, Abi> mask,
    Kokkos::Experimental::basic_simd<int, Abi> indices, Flag flag) {
  using size_type = Kokkos::Experimental::Impl::simd_size_t;
  using mask_type = decltype(mask);

  auto check_scattered = KOKKOS_LAMBDA(DataType(&arr)[init.size()], mask_type m = mask_type{true}) {
    kokkos_checker checker;
    for (size_type i = 0; i < init.size(); ++i) {
      auto expected = (m[i]) ? init[i] : arr[indices[i]];
      auto found    = arr[indices[i]];
      checker.equality(expected, found);
    }
  };

  DataType result[init.size()];
  {
    Kokkos::Experimental::unchecked_scatter_to(init, result, indices, flag);
    check_scattered(result);
  }
  {
    Kokkos::Experimental::unchecked_scatter_to(init, result, mask, indices,
                                               flag);
    check_scattered(result, mask);
  }
  {
    Kokkos::Experimental::partial_scatter_to(init, result, indices, flag);
    check_scattered(result);
  }
  {
    Kokkos::Experimental::partial_scatter_to(init, result, mask, indices, flag);
    check_scattered(result, mask);
  }
}

template <typename Abi, typename DataType, typename Flag>
KOKKOS_INLINE_FUNCTION
void device_test_gather_from(
    Kokkos::Experimental::basic_simd<DataType, Abi> init,
    Kokkos::Experimental::basic_simd_mask<int, Abi> mask,
    Kokkos::Experimental::basic_simd<int, Abi> indices, Flag flag) {
  using simd_type = Kokkos::Experimental::basic_simd<DataType, Abi>;
  using size_type = Kokkos::Experimental::Impl::simd_size_t;
  using mask_type = decltype(mask);

  auto check_gathered = KOKKOS_LAMBDA(DataType(&arr)[init.size()], simd_type& result, mask_type m = mask_type{true}) {
    kokkos_checker checker;
    for (size_type i = 0; i < result.size(); ++i) {
      auto expected = (m[i]) ? arr[indices[i]] : DataType{};
      auto found    = result[i];
      checker.equality(expected, found);
    }
  };

  simd_type result;
  DataType arr[init.size()];
  Kokkos::Experimental::simd_unchecked_store(init, arr, flag);
  {
    if constexpr (std::is_same_v<Kokkos::Experimental::simd_abi::Impl::
                                     host_fixed_native<DataType>,
                                 Abi>) {
    result =
        Kokkos::Experimental::unchecked_gather_from(arr, indices, flag);
    } else {
    result =
        Kokkos::Experimental::unchecked_gather_from<simd_type>(arr, indices, flag);
    }
    check_gathered(arr, result);
  }
  {
    if constexpr (std::is_same_v<Kokkos::Experimental::simd_abi::Impl::
                                     host_fixed_native<DataType>,
                                 Abi>) {
      result =
        Kokkos::Experimental::unchecked_gather_from(arr, mask, indices, flag);
    } else {
      result =
        Kokkos::Experimental::unchecked_gather_from<simd_type>(arr, mask, indices, flag);
    }
    check_gathered(arr, result, mask);
  }
  {
    if constexpr (std::is_same_v<Kokkos::Experimental::simd_abi::Impl::
                                     host_fixed_native<DataType>,
                                 Abi>) {
      result =
        Kokkos::Experimental::partial_gather_from(arr, indices, flag);
    } else {
      result =
        Kokkos::Experimental::partial_gather_from<simd_type>(arr, indices, flag);
    }
    check_gathered(arr, result);
  }
  {
    if constexpr (std::is_same_v<Kokkos::Experimental::simd_abi::Impl::
                                     host_fixed_native<DataType>,
                                 Abi>) {
      result =
          Kokkos::Experimental::partial_gather_from(arr, mask, indices, flag);
    } else {
      result =
          Kokkos::Experimental::partial_gather_from<simd_type>(arr, mask, indices, flag);
    }
    check_gathered(arr, result, mask);
  }
}

template <class Abi, typename DataType>
KOKKOS_INLINE_FUNCTION void device_check_memory_permute() {
  if constexpr (is_simd_avail_v<DataType, Abi>) {
    using simd_type  = Kokkos::Experimental::basic_simd<DataType, Abi>;
    using index_type = Kokkos::Experimental::basic_simd<int, Abi>;
    using mask_type  = typename index_type::mask_type;

    simd_type init(KOKKOS_LAMBDA(std::size_t i) { return (i + 1) * 11; });
    mask_type mask(KOKKOS_LAMBDA(std::size_t i) { return i % 2 == 0; });
    index_type reverse(
        KOKKOS_LAMBDA(std::size_t i) { return (simd_type::size() - 1) - i; });

    device_test_scatter_to(init, mask, reverse,
                          Kokkos::Experimental::simd_flag_default);
    device_test_scatter_to(init, mask, reverse,
                          Kokkos::Experimental::simd_flag_aligned);
    device_test_gather_from(init, mask, reverse,
                          Kokkos::Experimental::simd_flag_default);
    device_test_gather_from(init, mask, reverse,
                          Kokkos::Experimental::simd_flag_aligned);
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

TEST(simd, host_memory_permuete) {
  host_check_memory_permute_all_abis(
      Kokkos::Experimental::Impl::host_abi_set{});
}

TEST(simd, device_memory_permute) {
  Kokkos::parallel_for(1, simd_device_memory_permute_functor{});
  Kokkos::fence();
}

#endif