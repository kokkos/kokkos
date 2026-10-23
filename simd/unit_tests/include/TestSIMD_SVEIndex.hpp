// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_TEST_SIMD_SVE_INDEX_HPP
#define KOKKOS_TEST_SIMD_SVE_INDEX_HPP

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.simd;
import kokkos.simd_impl;
#else
#include <Kokkos_SIMD.hpp>
#endif
#include <SIMDTesting_Utilities.hpp>

// The half-size int32 index type of the SVE backend (same number of lanes as a
// vector of 64-bit data) is not covered by the generic tests, which use an
// index type with the same ABI as the data. Check it on its own for SVE-512.
#if defined(KOKKOS_ARCH_ARM_SVE) && defined(__ARM_FEATURE_SVE_BITS) && \
    (__ARM_FEATURE_SVE_BITS == 512)

namespace {

using sve_abi =
    Kokkos::Experimental::simd_abi::sve_fixed_size<__ARM_FEATURE_SVE_BITS / 64>;
using sve_data_simd  = Kokkos::Experimental::basic_simd<double, sve_abi>;
using sve_index_simd = Kokkos::Experimental::basic_simd<std::int32_t, sve_abi>;
using sve_index_mask = sve_index_simd::mask_type;
using sve_size_type  = Kokkos::Experimental::Impl::simd_size_t;
constexpr sve_size_type sve_size = sve_data_simd::size();

static_assert(std::is_same_v<
              sve_abi,
              Kokkos::Experimental::simd_abi::Impl::host_fixed_native<double>>);
static_assert(sve_index_simd::size() == sve_size);

inline void host_test_sve_index_construction() {
  alignas(64) std::int32_t src[sve_size];
  for (sve_size_type i = 0; i < sve_size; ++i)
    src[i] = ((i % 2) ? -1 : 1) * (10 + i);

  sve_index_simd broadcast(-5);
  sve_index_simd generated(
      [](sve_size_type i) { return static_cast<std::int32_t>(3 * i - 7); });
  sve_index_simd loaded(src);
  sve_index_simd loaded_aligned(src, Kokkos::Experimental::simd_flag_aligned);
  sve_index_mask first_half([](sve_size_type i) { return i < sve_size / 2; });
  sve_index_simd masked(src, first_half);
  sve_index_mask all(true);
  sve_index_mask none(false);
  sve_index_simd selected =
      Kokkos::Experimental::condition(first_half, loaded, broadcast);

  // widening keeps the sign, narrowing gives the original value back
  Kokkos::Experimental::basic_simd<std::int64_t, sve_abi> as_i64(loaded);
  Kokkos::Experimental::basic_simd<std::uint64_t, sve_abi> as_u64(loaded);
  sve_index_simd round_trip(as_u64);

  for (sve_size_type i = 0; i < sve_size; ++i) {
    EXPECT_EQ(broadcast[i], -5);
    EXPECT_EQ(generated[i], 3 * i - 7);
    EXPECT_EQ(loaded[i], src[i]);
    EXPECT_EQ(loaded_aligned[i], src[i]);
    EXPECT_EQ(first_half[i], i < sve_size / 2);
    EXPECT_EQ(masked[i], first_half[i] ? src[i] : 0);
    EXPECT_TRUE(all[i]);
    EXPECT_FALSE(none[i]);
    EXPECT_EQ(selected[i], first_half[i] ? src[i] : -5);
    EXPECT_EQ(as_i64[i], static_cast<std::int64_t>(src[i]));
    EXPECT_EQ(as_u64[i],
              static_cast<std::uint64_t>(static_cast<std::int64_t>(src[i])));
    EXPECT_EQ(round_trip[i], src[i]);
  }
}

template <typename Flag>
inline void host_test_sve_index_gather_scatter(Flag flag) {
  sve_data_simd init(
      [](sve_size_type i) { return static_cast<double>((i + 1) * 11); });
  sve_index_simd reverse([](sve_size_type i) {
    return static_cast<std::int32_t>(sve_size - 1 - i);
  });
  sve_index_mask even([](sve_size_type i) { return i % 2 == 0; });

  alignas(64) double arr[sve_size];
  for (sve_size_type i = 0; i < sve_size; ++i) arr[i] = 100.0 + i;

  auto check_gathered = [&](sve_data_simd const& result, bool use_mask) {
    for (sve_size_type i = 0; i < sve_size; ++i) {
      EXPECT_DOUBLE_EQ(result[i],
                       (!use_mask || even[i]) ? arr[sve_size - 1 - i] : 0.0);
    }
  };
  check_gathered(
      Kokkos::Experimental::unchecked_gather_from(arr, reverse, flag), false);
  check_gathered(
      Kokkos::Experimental::unchecked_gather_from(arr, even, reverse, flag),
      true);
  check_gathered(Kokkos::Experimental::partial_gather_from(arr, reverse, flag),
                 false);
  check_gathered(
      Kokkos::Experimental::partial_gather_from(arr, even, reverse, flag),
      true);

  constexpr double untouched = 7.0;
  alignas(64) double out[sve_size];
  auto reset = [&]() {
    for (sve_size_type i = 0; i < sve_size; ++i) out[i] = untouched;
  };
  auto check_scattered = [&](bool use_mask) {
    for (sve_size_type i = 0; i < sve_size; ++i) {
      EXPECT_DOUBLE_EQ(out[sve_size - 1 - i],
                       (!use_mask || even[i]) ? init[i] : untouched);
    }
  };
  reset();
  Kokkos::Experimental::unchecked_scatter_to(init, out, reverse, flag);
  check_scattered(false);
  reset();
  Kokkos::Experimental::unchecked_scatter_to(init, out, even, reverse, flag);
  check_scattered(true);
  reset();
  Kokkos::Experimental::partial_scatter_to(init, out, reverse, flag);
  check_scattered(false);
  reset();
  Kokkos::Experimental::partial_scatter_to(init, out, even, reverse, flag);
  check_scattered(true);

  // a fully masked-off scatter must not write anything
  reset();
  Kokkos::Experimental::partial_scatter_to(init, out, sve_index_mask(false),
                                           reverse, flag);
  for (sve_size_type i = 0; i < sve_size; ++i)
    EXPECT_DOUBLE_EQ(out[i], untouched);
}

}  // namespace

TEST(simd, host_sve_index_type) {
  host_test_sve_index_construction();
  host_test_sve_index_gather_scatter(Kokkos::Experimental::simd_flag_default);
  host_test_sve_index_gather_scatter(Kokkos::Experimental::simd_flag_aligned);
}

#endif  // SVE-512

#endif
