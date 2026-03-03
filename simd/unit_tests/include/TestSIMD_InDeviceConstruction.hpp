// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_TEST_SIMD_IN_DEVICE_CONSTRUCTION_HPP
#define KOKKOS_TEST_SIMD_IN_DEVICE_CONSTRUCTION_HPP

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.simd;
#else
#include <Kokkos_SIMD.hpp>
#endif
#include <SIMDTesting_Utilities.hpp>

template <typename T>
KOKKOS_INLINE_FUNCTION void test_simd_constructions() {
  using simd_type = T;

  [[maybe_unused]] simd_type a{0};
  [[maybe_unused]] simd_type gen(
      KOKKOS_LAMBDA(Kokkos::Experimental::Impl::simd_size_t i) { return i; });
  [[maybe_unused]] simd_type copy(a);
  [[maybe_unused]] simd_type move(std::move(gen));
}

template <typename T>
KOKKOS_INLINE_FUNCTION void test_mask_constructions() {
  using mask_type = T;

  [[maybe_unused]] mask_type a{false};
  [[maybe_unused]] mask_type gen(KOKKOS_LAMBDA(
      Kokkos::Experimental::Impl::simd_size_t i) { return (i == 0); });
  [[maybe_unused]] mask_type copy(a);
  [[maybe_unused]] mask_type move(std::move(gen));
}

template <typename T>
KOKKOS_INLINE_FUNCTION void test_basic_simd_operators() {
  T a{0}, b{1};
  (void)(a[0]);
  if constexpr (std::is_signed_v<typename T::value_type>) {
    (void)(-a);
  }
  (void)(a + b);
  (void)(a - b);
  (void)(a * b);
  (void)(a / b);
  (void)(a += b);
  (void)(a -= b);
  (void)(a *= b);
  (void)(a /= b);
  if constexpr (std::is_integral_v<typename T::value_type>) {
    (void)(~a);
    (void)(a & b);
    (void)(a | b);
    (void)(a ^ b);
    (void)(a << b);
    (void)(a >> b);
    (void)(a << 0);
    (void)(a >> 0);
    (void)(a &= b);
    (void)(a |= b);
    (void)(a ^= b);
    (void)(a <<= b);
    (void)(a >>= b);
    (void)(a <<= 0);
    (void)(a >>= 0);
  }
  (void)(a == b);
  (void)(a != b);
  (void)(a >= b);
  (void)(a <= b);
  (void)(a > b);
  (void)(a < b);
}

template <typename T>
KOKKOS_INLINE_FUNCTION void test_basic_mask_operators() {
  T a{false}, b{true};
  (void)!a;
  (void)(~a);
  (void)(a && b);
  (void)(a || b);
  (void)(a & b);
  (void)(a | b);
  (void)(a ^ b);
  (void)(a &= b);
  (void)(a |= b);
  (void)(a ^= b);
  (void)(a == b);
  (void)(a != b);

  // FIXME fallback impl needed
  // (void) (a>=b);
  // (void) (a<=b);
  // (void) (a>b);
  // (void) (a<b);
}

template <typename T>
KOKKOS_INLINE_FUNCTION void test_basic_math_fns() {
  T a{};
  Kokkos::abs(a);
  Kokkos::floor(a);
  Kokkos::ceil(a);
  Kokkos::round(a);
  Kokkos::trunc(a);
}

template <typename Abi, typename DataType>
KOKKOS_INLINE_FUNCTION void test_host_simd_construction() {
  using simd_type = Kokkos::Experimental::basic_simd<DataType, Abi>;
  using mask_type = Kokkos::Experimental::basic_simd_mask<DataType, Abi>;

  if constexpr (is_simd_avail_v<DataType, Abi>) {
    test_simd_constructions<simd_type>();
    test_mask_constructions<mask_type>();
    test_basic_simd_operators<simd_type>();
    test_basic_mask_operators<mask_type>();
    test_basic_math_fns<simd_type>();
  }
}

template <typename Abi, typename... DataTypes>
KOKKOS_INLINE_FUNCTION void check_host_simd_construction_all_types(
    Kokkos::Experimental::Impl::data_types<DataTypes...>) {
  (test_host_simd_construction<Abi, DataTypes>(), ...);
}

template <typename... Abis>
KOKKOS_INLINE_FUNCTION void check_host_simd_construction_all_abis(
    Kokkos::Experimental::Impl::abi_set<Abis...>) {
  using DataTypes = Kokkos::Experimental::Impl::data_type_set;
  (check_host_simd_construction_all_types<Abis>(DataTypes()), ...);
}

struct test_host_simd_construction_in_device_functor {
  KOKKOS_INLINE_FUNCTION void operator()(int) const {
    check_host_simd_construction_all_abis(
        Kokkos::Experimental::Impl::host_abi_set());
  }
};

// FIXME This test should eventually be integrated into other simd unit tests
TEST(simd, host_simd_construction_in_device_build) {
  using scalar_simd_abi = Kokkos::Experimental::simd_abi::scalar;
  using host_simd_abi =
      Kokkos::Experimental::simd_abi::Impl::host_fixed_native<double>;

  if constexpr (std::same_as<host_simd_abi, scalar_simd_abi>) {
    GTEST_SKIP();
  }

  test_host_simd_construction_in_device_functor{}(0);
  Kokkos::parallel_for(1, test_host_simd_construction_in_device_functor{});
}

#endif
