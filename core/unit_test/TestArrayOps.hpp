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

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wchar-subscripts"
#endif

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wchar-subscripts"
#endif

#include <Kokkos_Array.hpp>

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <numeric>

namespace {

TEST(TEST_CATEGORY, array_capacity) {
  using A = Kokkos::Array<int, 2>;
  A a{{3, 5}};

  ASSERT_FALSE(a.empty());
  ASSERT_EQ(a.size(), 2u);
  ASSERT_EQ(a.max_size(), 2u);
}

enum Enum { EZero, EOne };
enum EnumBool : bool { EBFalse, EBTrue };

TEST(TEST_CATEGORY, array_element_access) {
  using A = Kokkos::Array<int, 2>;
  A a{{3, 5}};
  A const& ca = a;

  size_t index = 1;
  ASSERT_EQ(a[index], 5);

  auto b = static_cast<bool>(index);
  ASSERT_EQ(a[b], a[index]);
  ASSERT_EQ(ca[b], a[index]);

  auto c = static_cast<char>(index);
  ASSERT_EQ(a[c], a[index]);
  ASSERT_EQ(ca[c], a[index]);

  auto sc = static_cast<signed char>(index);
  ASSERT_EQ(a[sc], a[index]);
  ASSERT_EQ(ca[sc], a[index]);

  auto uc = static_cast<unsigned char>(index);
  ASSERT_EQ(a[uc], a[index]);
  ASSERT_EQ(ca[uc], a[index]);

#if defined(__cpp_char8_t)
  auto c8 = static_cast<char8_t>(index);
  ASSERT_EQ(a[ch], a[index]);
  ASSERT_EQ(a[ch], a[index]);
#endif

  auto c16 = static_cast<char16_t>(index);
  ASSERT_EQ(a[c16], a[index]);
  ASSERT_EQ(ca[c16], a[index]);

  auto c32 = static_cast<char32_t>(index);
  ASSERT_EQ(a[c32], a[index]);
  ASSERT_EQ(ca[c32], a[index]);

  auto wc = static_cast<wchar_t>(index);
  ASSERT_EQ(a[wc], a[index]);
  ASSERT_EQ(ca[wc], a[index]);

  auto s = static_cast<short>(index);
  ASSERT_EQ(a[s], a[index]);
  ASSERT_EQ(ca[s], a[index]);

  auto us = static_cast<unsigned short>(index);
  ASSERT_EQ(a[us], a[index]);
  ASSERT_EQ(ca[us], a[index]);

  auto i = static_cast<int>(index);
  ASSERT_EQ(a[i], a[index]);
  ASSERT_EQ(ca[i], a[index]);

  auto ui = static_cast<unsigned int>(index);
  ASSERT_EQ(a[ui], a[index]);
  ASSERT_EQ(ca[ui], a[index]);

  auto l = static_cast<long>(index);
  ASSERT_EQ(a[l], a[index]);
  ASSERT_EQ(ca[l], a[index]);

  auto ul = static_cast<unsigned long>(index);
  ASSERT_EQ(a[ul], a[index]);
  ASSERT_EQ(ca[ul], a[index]);

  auto ll = static_cast<long long>(index);
  ASSERT_EQ(a[ll], a[index]);
  ASSERT_EQ(ca[ll], a[index]);

  auto ull = static_cast<unsigned long long>(index);
  ASSERT_EQ(a[ull], a[index]);
  ASSERT_EQ(ca[ull], a[index]);

  auto e = static_cast<Enum>(index);
  ASSERT_EQ(a[e], a[index]);
  ASSERT_EQ(ca[e], a[index]);

  auto eb = static_cast<EnumBool>(index);
  ASSERT_EQ(a[eb], a[index]);
  ASSERT_EQ(ca[eb], a[index]);

#if defined(__clang__)
  auto i128 = static_cast<__int128>(index);
  ASSERT_EQ(a[i128], a[index]);
  ASSERT_EQ(ca[i128], a[index]);

  auto u128 = static_cast<unsigned __int128>(index);
  ASSERT_EQ(a[u128], a[index]);
  ASSERT_EQ(ca[u128], a[index]);
#endif

  ASSERT_EQ(a.data()[index], a[index]);
  ASSERT_EQ(ca.data()[index], a[index]);
}

TEST(TEST_CATEGORY, array_zero_capacity) {
  using A = Kokkos::Array<int, 0>;
  A e;

  ASSERT_TRUE(e.empty());
  ASSERT_EQ(e.size(), 0u);
  ASSERT_EQ(e.max_size(), 0u);
}

TEST(TEST_CATEGORY, array_zero_data_nullptr) {
  using A = Kokkos::Array<int, 0>;

  A e;
  ASSERT_EQ(e.data(), nullptr);

  const A& ce = e;
  ASSERT_EQ(ce.data(), nullptr);
}

TEST(TEST_CATEGORY, array_contiguous_capacity) {
  using A =
      Kokkos::Array<int, KOKKOS_INVALID_INDEX, Kokkos::Array<>::contiguous>;

  A e(nullptr, 0);

  ASSERT_TRUE(e.empty());
  ASSERT_EQ(e.size(), 0u);
  ASSERT_EQ(e.max_size(), 0u);

  int aa[] = {3, 5};
  A a(aa, std::size(aa));

  ASSERT_EQ(a.empty(), 0 == std::size(aa));
  ASSERT_EQ(a.size(), std::size(aa));
  ASSERT_EQ(a.max_size(), std::size(aa));
}

TEST(TEST_CATEGORY, array_contiguous_element_access) {
  int aa[] = {3, 5};
  using A =
      Kokkos::Array<int, KOKKOS_INVALID_INDEX, Kokkos::Array<>::contiguous>;
  A a(aa, std::size(aa));
  A const& ca = a;

  size_t index = 1;
  ASSERT_EQ(std::addressof(a[index]), std::addressof(aa[index]));

  auto b = static_cast<bool>(index);
  ASSERT_EQ(a[b], aa[index]);
  ASSERT_EQ(ca[b], aa[index]);

  auto c = static_cast<char>(index);
  ASSERT_EQ(a[c], aa[index]);
  ASSERT_EQ(ca[c], aa[index]);

  auto sc = static_cast<signed char>(index);
  ASSERT_EQ(a[sc], aa[index]);
  ASSERT_EQ(ca[sc], aa[index]);

  auto uc = static_cast<unsigned char>(index);
  ASSERT_EQ(a[uc], aa[index]);
  ASSERT_EQ(ca[uc], aa[index]);

#if defined(__cpp_char8_t)
  auto c8 = static_cast<char8_t>(index);
  ASSERT_EQ(a[ch], aa[index]);
  ASSERT_EQ(a[ch], aa[index]);
#endif

  auto c16 = static_cast<char16_t>(index);
  ASSERT_EQ(a[c16], aa[index]);
  ASSERT_EQ(ca[c16], aa[index]);

  auto c32 = static_cast<char32_t>(index);
  ASSERT_EQ(a[c32], aa[index]);
  ASSERT_EQ(ca[c32], aa[index]);

  auto wc = static_cast<wchar_t>(index);
  ASSERT_EQ(a[wc], aa[index]);
  ASSERT_EQ(ca[wc], aa[index]);

  auto s = static_cast<short>(index);
  ASSERT_EQ(a[s], aa[index]);
  ASSERT_EQ(ca[s], aa[index]);

  auto us = static_cast<unsigned short>(index);
  ASSERT_EQ(a[us], aa[index]);
  ASSERT_EQ(ca[us], aa[index]);

  auto i = static_cast<int>(index);
  ASSERT_EQ(a[i], aa[index]);
  ASSERT_EQ(ca[i], aa[index]);

  auto ui = static_cast<unsigned int>(index);
  ASSERT_EQ(a[ui], aa[index]);
  ASSERT_EQ(ca[ui], aa[index]);

  auto l = static_cast<long>(index);
  ASSERT_EQ(a[l], aa[index]);
  ASSERT_EQ(ca[l], aa[index]);

  auto ul = static_cast<unsigned long>(index);
  ASSERT_EQ(a[ul], aa[index]);
  ASSERT_EQ(ca[ul], aa[index]);

  auto ll = static_cast<long long>(index);
  ASSERT_EQ(a[ll], aa[index]);
  ASSERT_EQ(ca[ll], aa[index]);

  auto ull = static_cast<unsigned long long>(index);
  ASSERT_EQ(a[ull], aa[index]);
  ASSERT_EQ(ca[ull], aa[index]);

  auto e = static_cast<Enum>(index);
  ASSERT_EQ(a[e], aa[index]);
  ASSERT_EQ(ca[e], aa[index]);

  auto eb = static_cast<EnumBool>(index);
  ASSERT_EQ(a[eb], aa[index]);
  ASSERT_EQ(ca[eb], aa[index]);

#if defined(__clang__)
  auto i128 = static_cast<__int128>(index);
  ASSERT_EQ(a[i128], aa[index]);
  ASSERT_EQ(ca[i128], aa[index]);

  auto u128 = static_cast<unsigned __int128>(index);
  ASSERT_EQ(a[u128], aa[index]);
  ASSERT_EQ(ca[u128], aa[index]);
#endif

  ASSERT_EQ(a.data(), aa);
  ASSERT_EQ(ca.data(), aa);
}

TEST(TEST_CATEGORY, array_contiguous_assignment) {
  using A =
      Kokkos::Array<int, KOKKOS_INVALID_INDEX, Kokkos::Array<>::contiguous>;

  int aa[] = {3, 5};
  A a(aa, std::size(aa));

  // operator=(Array<T, N, P> const&) semantics when lhs size a > rhs size b
  using B = Kokkos::Array<int, 1>;
  static_assert(std::size(aa) > B::size());
  B b{{7}};

  ASSERT_GT(std::size(a), std::size(b));
  a = b;
  ASSERT_GT(std::size(a), std::size(b));

  ASSERT_EQ(a.size(), std::size(aa));
  ASSERT_EQ(a.max_size(), std::size(aa));
  ASSERT_EQ(a[0], 7);
  ASSERT_EQ(a[1], 5);

  // operator=(Array<T, N, P> const&) semantics when lhs size a < rhs size d
  using D = Kokkos::Array<int, 4>;
  static_assert(std::size(aa) < D::size());
  D d{{11, 13, 17, 19}};

  ASSERT_LT(std::size(a), std::size(d));
  a = d;
  ASSERT_LT(std::size(a), std::size(d));

  ASSERT_EQ(a.size(), std::size(aa));
  ASSERT_EQ(a.max_size(), std::size(aa));
  ASSERT_EQ(a[0], 11);
  ASSERT_EQ(a[1], 13);

  // Copy assignment operator semantics when lhs size a > rhs size e
  int ee[] = {23};
  A e(ee, std::size(ee));

  ASSERT_GT(a.size(), e.size());
  a = e;
  ASSERT_GT(a.size(), e.size());

  ASSERT_EQ(a.size(), std::size(aa));
  ASSERT_EQ(a.max_size(), std::size(aa));
  ASSERT_EQ(a[0], 23);
  ASSERT_EQ(a[1], 13);

  // Copy assignment operator semantics when lhs size e < rhs size a
  ASSERT_LT(e.size(), a.size());
  e[0] = 29;  // To check that e[0] is overwritten by e = a
  e    = a;
  ASSERT_LT(e.size(), a.size());

  ASSERT_EQ(e.size(), std::size(ee));
  ASSERT_EQ(e.max_size(), std::size(ee));
  ASSERT_EQ(e[0], 23);
}

TEST(TEST_CATEGORY, array_strided_capacity) {
  using A = Kokkos::Array<int, KOKKOS_INVALID_INDEX, Kokkos::Array<>::strided>;

  A e(nullptr, 0, 0);

  ASSERT_TRUE(e.empty());
  ASSERT_EQ(e.size(), 0u);
  ASSERT_EQ(e.max_size(), 0u);

  int aa[]                 = {5, 7, 11, 13, 17, 19};
  constexpr size_t aStride = 2;
  A a(aa, std::size(aa) / aStride, aStride);

  ASSERT_EQ(a.empty(), 0 == std::size(aa) / aStride);
  ASSERT_EQ(a.size(), std::size(aa) / aStride);
  ASSERT_EQ(a.max_size(), std::size(aa) / aStride);
}

TEST(TEST_CATEGORY, array_strided_element_access) {
  using A = Kokkos::Array<int, KOKKOS_INVALID_INDEX, Kokkos::Array<>::strided>;

  int aa[]                 = {5, 7, 11, 13, 17, 19};
  constexpr size_t aStride = 2;

  A a(aa, std::size(aa) / aStride, aStride);
  A const& ca = a;

  size_t index = 1;
  ASSERT_EQ(std::addressof(a[index]), std::addressof(aa[index * aStride]));

  auto b = static_cast<bool>(index);
  ASSERT_EQ(a[b], aa[index * aStride]);
  ASSERT_EQ(ca[b], aa[index * aStride]);

  auto c = static_cast<char>(index);
  ASSERT_EQ(a[c], aa[index * aStride]);
  ASSERT_EQ(ca[c], aa[index * aStride]);

  auto sc = static_cast<signed char>(index);
  ASSERT_EQ(a[sc], aa[index * aStride]);
  ASSERT_EQ(ca[sc], aa[index * aStride]);

  auto uc = static_cast<unsigned char>(index);
  ASSERT_EQ(a[uc], aa[index * aStride]);
  ASSERT_EQ(ca[uc], aa[index * aStride]);

#if defined(__cpp_char8_t)
  auto c8 = static_cast<char8_t>(index);
  ASSERT_EQ(a[ch], aa[index * aStride]);
  ASSERT_EQ(a[ch], aa[index * aStride]);
#endif

  auto c16 = static_cast<char16_t>(index);
  ASSERT_EQ(a[c16], aa[index * aStride]);
  ASSERT_EQ(ca[c16], aa[index * aStride]);

  auto c32 = static_cast<char32_t>(index);
  ASSERT_EQ(a[c32], aa[index * aStride]);
  ASSERT_EQ(ca[c32], aa[index * aStride]);

  auto wc = static_cast<wchar_t>(index);
  ASSERT_EQ(a[wc], aa[index * aStride]);
  ASSERT_EQ(ca[wc], aa[index * aStride]);

  auto s = static_cast<short>(index);
  ASSERT_EQ(a[s], aa[index * aStride]);
  ASSERT_EQ(ca[s], aa[index * aStride]);

  auto us = static_cast<unsigned short>(index);
  ASSERT_EQ(a[us], aa[index * aStride]);
  ASSERT_EQ(ca[us], aa[index * aStride]);

  auto i = static_cast<int>(index);
  ASSERT_EQ(a[i], aa[index * aStride]);
  ASSERT_EQ(ca[i], aa[index * aStride]);

  auto ui = static_cast<unsigned int>(index);
  ASSERT_EQ(a[ui], aa[index * aStride]);
  ASSERT_EQ(ca[ui], aa[index * aStride]);

  auto l = static_cast<long>(index);
  ASSERT_EQ(a[l], aa[index * aStride]);
  ASSERT_EQ(ca[l], aa[index * aStride]);

  auto ul = static_cast<unsigned long>(index);
  ASSERT_EQ(a[ul], aa[index * aStride]);
  ASSERT_EQ(ca[ul], aa[index * aStride]);

  auto ll = static_cast<long long>(index);
  ASSERT_EQ(a[ll], aa[index * aStride]);
  ASSERT_EQ(ca[ll], aa[index * aStride]);

  auto ull = static_cast<unsigned long long>(index);
  ASSERT_EQ(a[ull], aa[index * aStride]);
  ASSERT_EQ(ca[ull], aa[index * aStride]);

  auto e = static_cast<Enum>(index);
  ASSERT_EQ(a[e], aa[index * aStride]);
  ASSERT_EQ(ca[e], aa[index * aStride]);

  auto eb = static_cast<EnumBool>(index);
  ASSERT_EQ(a[eb], aa[index * aStride]);
  ASSERT_EQ(ca[eb], aa[index * aStride]);

#if defined(__clang__)
  auto i128 = static_cast<__int128>(index);
  ASSERT_EQ(a[i128], aa[index * aStride]);
  ASSERT_EQ(ca[i128], aa[index * aStride]);

  auto u128 = static_cast<unsigned __int128>(index);
  ASSERT_EQ(a[u128], aa[index * aStride]);
  ASSERT_EQ(ca[u128], aa[index * aStride]);
#endif

  ASSERT_EQ(a.data(), aa);
  ASSERT_EQ(ca.data(), aa);
}

TEST(TEST_CATEGORY, array_strided_assignment) {
  using A  = Kokkos::Array<int, KOKKOS_INVALID_INDEX, Kokkos::Array<>::strided>;
  int aa[] = {5, 7, 11, 13, 17, 19};
  constexpr size_t aStride = 2;
  A a(aa, std::size(aa) / aStride, aStride);

  // operator=(Array<T, N, P> const&) semantics when lhs size a > rhs size b
  using B = Kokkos::Array<int, 1>;
  static_assert(std::size(aa) / aStride > B::size());
  B b{{23}};

  ASSERT_GT(std::size(a), std::size(b));
  a = b;
  ASSERT_GT(std::size(a), std::size(b));

  ASSERT_EQ(a.size(), std::size(aa) / aStride);
  ASSERT_EQ(a.max_size(), std::size(aa) / aStride);
  ASSERT_EQ(a[0], b[0]);
  ASSERT_EQ(a[1], aa[1 * aStride]);

  // operator=(Array<T, N, P> const&) semantics when lhs size a < rhs size d
  using D = Kokkos::Array<int, 7>;
  static_assert(std::size(aa) / aStride < D::size());
  D d{{29, 31, 37, 41, 43, 47, 53}};

  ASSERT_LT(std::size(a), std::size(d));
  a = d;
  ASSERT_LT(std::size(a), std::size(d));

  ASSERT_EQ(a.size(), std::size(aa) / aStride);
  ASSERT_EQ(a.max_size(), std::size(aa) / aStride);
  ASSERT_EQ(a[0], d[0]);
  ASSERT_EQ(a[1], d[1]);

  // Copy assignment operator semantics when lhs size a > rhs size e
  int ee[]                 = {59, 61, 67, 71, 73, 79};
  constexpr size_t eStride = 3;
  A e(ee, std::size(ee) / eStride, eStride);

  ASSERT_GT(a.size(), e.size());
  a = e;
  ASSERT_GT(a.size(), e.size());

  ASSERT_EQ(a.size(), std::size(aa) / aStride);
  ASSERT_EQ(a.max_size(), std::size(aa) / aStride);
  ASSERT_EQ(a[0], ee[0 * eStride]);
  ASSERT_EQ(a[1], ee[1 * eStride]);

  // Copy assignment operator semantics when lhs size e < rhs size a
  e[0] = 83;  // To check that e[0] is overwritten by e = a
  ASSERT_LT(e.size(), a.size());
  e = a;
  ASSERT_LT(e.size(), a.size());
  ASSERT_EQ(e.size(), std::size(ee) / eStride);
  ASSERT_EQ(e.max_size(), std::size(ee) / eStride);
  ASSERT_EQ(e[0], ee[0]);
}

}  // namespace
