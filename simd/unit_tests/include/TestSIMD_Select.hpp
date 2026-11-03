// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_TEST_SIMD_SELECT_HPP
#define KOKKOS_TEST_SIMD_SELECT_HPP

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.simd;
import kokkos.simd_impl;
#else
#include <Kokkos_SIMD.hpp>
#endif
#include <SIMDTesting_Utilities.hpp>

template <typename Abi, typename DataType>
inline void host_check_select() {
  if constexpr (is_simd_avail_v<DataType, Abi>) {
    using simd_type = typename Kokkos::Experimental::basic_simd<DataType, Abi>;
    using mask_type = typename simd_type::mask_type;

    auto select_op = [](mask_type const& mask, simd_type const& a,
                        simd_type const& b) {
      return Kokkos::Experimental::select(mask, a, b);
    };

    simd_type value_a(16);
    simd_type value_b(20);

    auto select_result = select_op(mask_type(false), value_a, value_b);
    EXPECT_TRUE(all_of(select_result == value_b));
    select_result = select_op(mask_type(true), value_a, value_b);
    EXPECT_TRUE(all_of(select_result == value_a));

    mask_type mask([](std::size_t i) { return i % 2 == 0; });
    select_result = select_op(mask, value_a, value_b);
    for (std::size_t i = 0; i < simd_type::size(); ++i) {
      EXPECT_EQ(select_result[i], mask[i] ? value_a[i] : value_b[i]);
    }

    mask_type mask_a([](std::size_t i) { return i % 3 == 0; });
    mask_type mask_b([](std::size_t i) { return i % 3 != 1; });
    mask_type mask_result = Kokkos::Experimental::select(mask, mask_a, mask_b);

    for (std::size_t i = 0; i < simd_type::size(); ++i) {
      EXPECT_EQ(mask_result[i], mask[i] ? mask_a[i] : mask_b[i]);
    }
  }
}

template <typename Abi, typename... DataTypes>
inline void host_check_select_all_types(
    Kokkos::Experimental::Impl::data_types<DataTypes...>) {
  (host_check_select<Abi, DataTypes>(), ...);
}

template <typename... Abis>
inline void host_check_select_all_abis(
    Kokkos::Experimental::Impl::abi_set<Abis...>) {
  using DataTypes = Kokkos::Experimental::Impl::data_type_set;
  (host_check_select_all_types<Abis>(DataTypes()), ...);
}

template <typename Abi, typename DataType>
KOKKOS_INLINE_FUNCTION void device_check_select() {
  if constexpr (is_type_v<Kokkos::Experimental::basic_simd<DataType, Abi>>) {
    using simd_type = typename Kokkos::Experimental::basic_simd<DataType, Abi>;
    using mask_type = typename simd_type::mask_type;
    kokkos_checker checker;

    auto select_op = [](mask_type const& mask, simd_type const& a,
                        simd_type const& b) {
      return Kokkos::Experimental::select(mask, a, b);
    };

    simd_type value_a(16);
    simd_type value_b(20);

    auto select_result = select_op(mask_type(false), value_a, value_b);
    checker.truth(all_of(select_result == value_b));
    select_result = select_op(mask_type(true), value_a, value_b);
    checker.truth(all_of(select_result == value_a));

    mask_type mask([](std::size_t i) { return i % 2 == 0; });
    select_result = select_op(mask, value_a, value_b);
    for (std::size_t i = 0; i < simd_type::size(); ++i) {
      checker.equality(select_result[i], mask[i] ? value_a[i] : value_b[i]);
    }

    mask_type mask_a([](std::size_t i) { return i % 3 == 0; });
    mask_type mask_b([](std::size_t i) { return i % 3 != 1; });
    mask_type mask_result = Kokkos::Experimental::select(mask, mask_a, mask_b);

    for (std::size_t i = 0; i < simd_type::size(); ++i) {
      checker.equality(mask_result[i], mask[i] ? mask_a[i] : mask_b[i]);
    }
  }
}

template <typename Abi, typename... DataTypes>
KOKKOS_INLINE_FUNCTION void device_check_select_all_types(
    Kokkos::Experimental::Impl::data_types<DataTypes...>) {
  (device_check_select<Abi, DataTypes>(), ...);
}

template <typename... Abis>
KOKKOS_INLINE_FUNCTION void device_check_select_all_abis(
    Kokkos::Experimental::Impl::abi_set<Abis...>) {
  using DataTypes = Kokkos::Experimental::Impl::data_type_set;
  (device_check_select_all_types<Abis>(DataTypes()), ...);
}

class simd_device_select_functor {
 public:
  KOKKOS_INLINE_FUNCTION void operator()(int) const {
    device_check_select_all_abis(Kokkos::Experimental::Impl::device_abi_set());
  }
};

TEST(simd, host_select) {
  host_check_select_all_abis(Kokkos::Experimental::Impl::host_abi_set());
}

TEST(simd, device_select) {
  Kokkos::parallel_for(1, simd_device_select_functor());
  Kokkos::fence();
}

#endif
