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

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <numeric>

namespace {

TEST(TEST_CATEGORY, array_capacity) {
  using A = Kokkos::Array<int, 2>;
  A a{{3, 5}};

  ASSERT_EQ(a.empty(), 0 == a.size());
  ASSERT_EQ(a.size(), 2u);
  ASSERT_EQ(a.max_size(), 2u);
}

enum Enum { EZero, EOne };
enum EnumBool : bool { EBFalse, EBTrue };
enum class ScopedEnum { SEZero, SEOne };
enum class ScopedEnumShort : short { SESZero, SESOne };

TEST(TEST_CATEGORY, array_element_access) {
  using A = Kokkos::Array<int, 2>;
  A a{{3, 5}};
  A const& ca = a;

  size_t index = 1;
  ASSERT_EQ(a[index], 5);

  auto b       = static_cast<bool>(index);
ASSERT_EQ(a[b], a[index]);
ASSERT_EQ(ca[b], a[index]);

  auto c       = static_cast<char>(index);
  auto sc      = static_cast<signed char>(index);
  auto uc      = static_cast<unsigned char>(index);
  auto c16     = static_cast<char16_t>(index);
  auto c32     = static_cast<char32_t>(index);
  auto wc      = static_cast<wchar_t>(index);
  auto s       = static_cast<short>(index);
  auto us      = static_cast<unsigned short>(index);
  auto i       = static_cast<int>(index);
  auto ui      = static_cast<unsigned int>(index);
  auto l       = static_cast<long>(index);
  auto ul      = static_cast<unsigned long>(index);
  auto ll      = static_cast<long long>(index);
  auto ull     = static_cast<unsigned long long>(index);
  auto e       = static_cast<Enum>(index);
  auto eb      = static_cast<EnumBool>(index);
  auto se      = static_cast<ScopedEnum>(index);
  auto ses     = static_cast<ScopedEnumShort>(index);


  ASSERT_EQ(a[b], a[index]);
  ASSERT_EQ(a[sc], a[index]);
  ASSERT_EQ(a[uc], a[index]);
  ASSERT_EQ(a[c16], a[index]);
  ASSERT_EQ(a[c32], a[index]);
  ASSERT_EQ(a[wc], a[index]);
  ASSERT_EQ(a[s], a[index]);
  ASSERT_EQ(a[us], a[index]);
  ASSERT_EQ(a[i], a[index]);
  ASSERT_EQ(a[ui], a[index]);
  ASSERT_EQ(a[l], a[index]);
  ASSERT_EQ(a[ul], a[index]);
  ASSERT_EQ(a[ll], a[index]);
  ASSERT_EQ(a[ull], a[index]);
  ASSERT_EQ(a[e], a[index]);
  ASSERT_EQ(a[eb], a[index]);
  ASSERT_EQ(a[se], a[index]);
  ASSERT_EQ(a[ses], a[index]);

  ASSERT_EQ(ca[b], a[index]);
  ASSERT_EQ(ca[sc], a[index]);
  ASSERT_EQ(ca[uc], a[index]);
  ASSERT_EQ(ca[c16], a[index]);
  ASSERT_EQ(ca[c32], a[index]);
  ASSERT_EQ(ca[wc], a[index]);
  ASSERT_EQ(ca[s], a[index]);
  ASSERT_EQ(ca[us], a[index]);
  ASSERT_EQ(ca[i], a[index]);
  ASSERT_EQ(ca[ui], a[index]);
  ASSERT_EQ(ca[l], a[index]);
  ASSERT_EQ(ca[ul], a[index]);
  ASSERT_EQ(ca[ll], a[index]);
  ASSERT_EQ(ca[ull], a[index]);
  ASSERT_EQ(ca[e], a[index]);
  ASSERT_EQ(ca[eb], a[index]);
  ASSERT_EQ(ca[se], a[index]);
  ASSERT_EQ(ca[ses], a[index]);

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wchar-subscripts"
#endif

  ASSERT_EQ(a[c], a[index]);
  ASSERT_EQ(ca[c], a[index]);

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#if defined(__cpp_char8_t)
  auto c8 = static_cast<char8_t>(index);
  ASSERT_EQ(a[ch], a[index]);
  ASSERT_EQ(a[ch], a[index]);
#endif

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

TEST(TEST_CATEGORY, array_element_access_by_bool) {
  using A = Kokkos::Array<int, 2>;
  A a{{3, 5}};
  ASSERT_EQ(a[false], 3);
  ASSERT_EQ(a[true], 5);

  const A& c = a;
  ASSERT_EQ(c[false], 3);
  ASSERT_EQ(c[true], 5);
}

TEST(TEST_CATEGORY, array_element_access_by_int) {
  using A = Kokkos::Array<int, 2>;
  A a{{3, 5}};
  ASSERT_EQ(a[0], 3);
  ASSERT_EQ(a[1], 5);

  const A& c = a;
  ASSERT_EQ(c[0], 3);
  ASSERT_EQ(c[1], 5);
}

TEST(TEST_CATEGORY, array_element_access_by_long_long) {
  using A = Kokkos::Array<int, 2>;
  A a{{3, 5}};
  ASSERT_EQ(a[0ll], 3);
  ASSERT_EQ(a[1ll], 5);

  const A& c = a;
  ASSERT_EQ(c[0ll], 3);
  ASSERT_EQ(c[1ll], 5);
}

TEST(TEST_CATEGORY, array_element_access_by_enum) {
  using A = Kokkos::Array<int, 2>;
  A a{{3, 5}};
  ASSERT_EQ(a[EZero], 3);
  ASSERT_EQ(a[EOne], 5);

  const A& c = a;
  ASSERT_EQ(c[EZero], 3);
  ASSERT_EQ(c[EOne], 5);
}

TEST(TEST_CATEGORY, array_element_access_by_scoped_enum) {
  using A = Kokkos::Array<int, 2>;
  A a{{3, 5}};
  ASSERT_EQ(a[ScopedEnum::SEZero], 3);
  ASSERT_EQ(a[ScopedEnum::SEOne], 5);

  const A& c = a;
  ASSERT_EQ(c[ScopedEnum::SEZero], 3);
  ASSERT_EQ(c[ScopedEnum::SEOne], 5);
}
TEST(TEST_CATEGORY, array_element_access_data) {
  using A = Kokkos::Array<int, 2>;
  A a{{3, 5}};
  ASSERT_EQ(a.data()[0], 3);
  ASSERT_EQ(a.data()[1], 5);

  const A& c = a;
  ASSERT_EQ(c.data()[0], 3);
  ASSERT_EQ(c.data()[1], 5);
}

TEST(TEST_CATEGORY, array_T_0) {
  using A = Kokkos::Array<int, 0>;
  A a;

  ASSERT_EQ(a.empty(), 0 == a.size());
  ASSERT_EQ(a.size(), 0u);
  ASSERT_EQ(a.max_size(), 0u);

  ASSERT_EQ(a.data(), nullptr);

  const A& c = a;
  ASSERT_EQ(c.data(), nullptr);
}

TEST(TEST_CATEGORY, array_contiguous) {
  int aa[] = {3, 5};

  using A =
      Kokkos::Array<int, KOKKOS_INVALID_INDEX, Kokkos::Array<>::contiguous>;
  A a(aa, std::size(aa));

  // capacity
  ASSERT_EQ(a.empty(), 0 == a.size());
  ASSERT_EQ(a.size(), std::size(aa));
  ASSERT_EQ(a.max_size(), std::size(aa));

  // index 0
  ASSERT_EQ(a[0], 3);
  ASSERT_EQ(a[false], 3);
  ASSERT_EQ(a[EZero], 3);
  ASSERT_EQ(a[ScopedEnum::SEZero], 3);
  ASSERT_EQ(a[ScopedEnumShort::SESZero], 3);

  // index 1
  ASSERT_EQ(a[1], 5);
  ASSERT_EQ(a[true], 5);
  ASSERT_EQ(a[EOne], 5);
  ASSERT_EQ(a[ScopedEnum::SEOne], 5);
  ASSERT_EQ(a[ScopedEnumShort::SESOne], 5);

  // data()
  ASSERT_EQ(a.data(), aa);
  ASSERT_EQ(a[0], aa[0]);
  ASSERT_EQ(a[1], aa[1]);

  // const index, data()
  const A& c = a;

  ASSERT_EQ(c[0], 3);
  ASSERT_EQ(c[false], 3);
  ASSERT_EQ(c[EZero], 3);
  ASSERT_EQ(c[ScopedEnum::SEZero], 3);
  ASSERT_EQ(c[ScopedEnumShort::SESZero], 3);

  ASSERT_EQ(c[1], 5);
  ASSERT_EQ(c[true], 5);
  ASSERT_EQ(c[EOne], 5);
  ASSERT_EQ(c[ScopedEnum::SEOne], 5);
  ASSERT_EQ(c[ScopedEnumShort::SESOne], 5);

  ASSERT_EQ(c.data(), aa);
  ASSERT_EQ(c[0], aa[0]);
  ASSERT_EQ(c[1], aa[1]);

  // operator=(Array<T, N, P> const&) semantics when b.size() < a.size()
  using B = Kokkos::Array<int, 1>;
  static_assert(B::size() < std::size(aa));
  B b{{7}};

  ASSERT_LT(std::size(b), std::size(a));
  a = b;
  ASSERT_LT(std::size(b), std::size(a));

  ASSERT_EQ(a.size(), std::size(aa));
  ASSERT_EQ(a.max_size(), std::size(aa));
  ASSERT_EQ(a[0], 7);
  ASSERT_EQ(a[1], 5);

  // operator=(Array<T, N, P> const&) semantics when a.size() < d.size()
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

  int ee[] = {23};
  A e(ee, std::size(ee));
  ASSERT_LT(e.size(), a.size());

  ASSERT_LT(e.size(), a.size());
  a = e;
  ASSERT_LT(e.size(), a.size());

  ASSERT_EQ(a.size(), std::size(aa));
  ASSERT_EQ(a.max_size(), std::size(aa));
  ASSERT_EQ(a[0], 23);
  ASSERT_EQ(a[1], 13);

  e[0] = 29;

  ASSERT_LT(e.size(), a.size());
  e = a;
  ASSERT_LT(e.size(), a.size());

  ASSERT_EQ(e.size(), std::size(ee));
  ASSERT_EQ(e.max_size(), std::size(ee));
  ASSERT_EQ(e[0], 23);
}
TEST(TEST_CATEGORY, array_strided) {
  int aa[] = {5, 7, 11, 13, 17, 19};

  using A = Kokkos::Array<int, KOKKOS_INVALID_INDEX, Kokkos::Array<>::strided>;

  constexpr size_t aStride = 2;

  A a(aa, std::size(aa) / aStride, aStride);

  // capacity
  ASSERT_EQ(a.empty(), 0 == a.size());
  ASSERT_EQ(a.size(), std::size(aa) / aStride);
  ASSERT_EQ(a.max_size(), std::size(aa) / aStride);

  // index 0
  ASSERT_EQ(a[0], aa[0 * aStride]);
  ASSERT_EQ(a[false], aa[0 * aStride]);
  ASSERT_EQ(a[EZero], aa[0 * aStride]);
  ASSERT_EQ(a[ScopedEnum::SEZero], aa[0 * aStride]);
  ASSERT_EQ(a[ScopedEnumShort::SESZero], aa[0 * aStride]);

  // index 1
  ASSERT_EQ(a[1], aa[1 * aStride]);
  ASSERT_EQ(a[true], aa[1 * aStride]);
  ASSERT_EQ(a[EOne], aa[1 * aStride]);
  ASSERT_EQ(a[ScopedEnum::SEOne], aa[1 * aStride]);
  ASSERT_EQ(a[ScopedEnumShort::SESOne], aa[1 * aStride]);

  // data()
  ASSERT_EQ(a.data(), aa);

  // const index, data()
  const A& c = a;

  ASSERT_EQ(c[0], aa[0 * aStride]);
  ASSERT_EQ(c[false], aa[0 * aStride]);
  ASSERT_EQ(c[EZero], aa[0 * aStride]);
  ASSERT_EQ(c[ScopedEnum::SEZero], aa[0 * aStride]);
  ASSERT_EQ(c[ScopedEnumShort::SESZero], aa[0 * aStride]);

  ASSERT_EQ(c[1], aa[1 * aStride]);
  ASSERT_EQ(c[true], aa[1 * aStride]);
  ASSERT_EQ(c[EOne], aa[1 * aStride]);
  ASSERT_EQ(c[ScopedEnum::SEOne], aa[1 * aStride]);
  ASSERT_EQ(c[ScopedEnumShort::SESOne], aa[1 * aStride]);

  ASSERT_EQ(c.data(), aa);

  // operator=(Array<T, N, P> const&) semantics when b.size() < a.size()
  using B = Kokkos::Array<int, 1>;
  B b{{23}};

  ASSERT_LT(std::size(b), std::size(a));
  a = b;

  ASSERT_EQ(a.size(), std::size(aa) / aStride);
  ASSERT_EQ(a.max_size(), std::size(aa) / aStride);
  ASSERT_EQ(a[0], b[0]);
  ASSERT_EQ(a[1], aa[1 * aStride]);

  // operator=(Array<T, N, P> const&) semantics when a.size() < d.size()
  using D = Kokkos::Array<int, 7>;
  D d{{29, 31, 37, 41, 43, 47, 53}};

  ASSERT_LT(std::size(a), std::size(d));
  a = d;
  ASSERT_LT(std::size(a), std::size(d));

  ASSERT_EQ(a.size(), std::size(aa) / aStride);
  ASSERT_EQ(a.max_size(), std::size(aa) / aStride);
  ASSERT_EQ(a[0], d[0]);
  ASSERT_EQ(a[1], d[1]);

  int ee[]                 = {59, 61, 67, 71, 73, 79};
  constexpr size_t eStride = 3;
  A e(ee, std::size(ee) / eStride, eStride);

  ASSERT_LT(e.size(), a.size());
  a = e;
  ASSERT_LT(e.size(), a.size());

  ASSERT_EQ(a.size(), std::size(aa) / aStride);
  ASSERT_EQ(a.max_size(), std::size(aa) / aStride);
  ASSERT_EQ(a[0], ee[0 * eStride]);
  ASSERT_EQ(a[1], ee[1 * eStride]);

  e[0] = 83;
  e    = a;
  ASSERT_LT(e.size(), a.size());
  ASSERT_EQ(e.size(), std::size(ee) / eStride);
  ASSERT_EQ(e.max_size(), std::size(ee) / eStride);
  ASSERT_EQ(e[0], ee[0]);
}

struct SetOnMove {
  KOKKOS_INLINE_FUNCTION SetOnMove(int i_) : i(i_) {}
  KOKKOS_INLINE_FUNCTION operator int() const { return i; }

  KOKKOS_DEFAULTED_FUNCTION SetOnMove() = default;

  KOKKOS_DEFAULTED_FUNCTION SetOnMove(SetOnMove const&) = default;
  KOKKOS_DEFAULTED_FUNCTION SetOnMove& operator=(SetOnMove const&) = default;
  KOKKOS_DEFAULTED_FUNCTION ~SetOnMove()                           = default;

  KOKKOS_INLINE_FUNCTION SetOnMove(SetOnMove&& that) : i(that.i) {
    that.i = -1;
  }
  KOKKOS_INLINE_FUNCTION SetOnMove& operator=(SetOnMove&& that) {
    i      = that.i;
    that.i = -1;
    return *this;
  }

  int i = std::numeric_limits<int>::min();
};

TEST(TEST_CATEGORY, to_Array_lvalue) {
  int array[] = {
      2,
      3,
      5,
      7,
  };
  int a_sum   = 0;  // sum of array elements
  int som_sum = 0;  // sum of som_array elements
  int ka_sum  = 0;  // sum of Kokkos::Array elements

  Kokkos::parallel_reduce(
      1,
      KOKKOS_LAMBDA(int, int& asum, int& somsum, int& kasum) {
        SetOnMove som_array[std::size(array)];
        int i = 0;
        for (auto& v : array) som_array[i++] = v;

        auto ka = Kokkos::to_Array(som_array);
        static_assert(std::is_same_v<
                      Kokkos::Array<std::remove_extent_t<decltype(som_array)>,
                                    std::size(array)>,
                      decltype(ka)>);

        for (size_t j = 0; j != ka.size(); ++j) {
          asum += array[j];
          somsum += som_array[j];
          kasum += ka[j];
        }
      },
      a_sum, som_sum, ka_sum);

  ASSERT_EQ(som_sum, a_sum);
  ASSERT_EQ(ka_sum, a_sum);
}

TEST(TEST_CATEGORY, to_Array_rvalue) {
  int array[] = {
      2,
      3,
      5,
      7,
  };
  int a_sum   = 0;  // sum of array elements
  int som_sum = 0;  // sum of som_array elements
  int ka_sum  = 0;  // sum of Kokkos::Array elements

  Kokkos::parallel_reduce(
      1,
      KOKKOS_LAMBDA(int, int& asum, int& somsum, int& kasum) {
        SetOnMove som_array[std::size(array)];
        int i = 0;
        for (auto& v : array) som_array[i++] = v;

        auto ka = Kokkos::to_Array(std::move(som_array));
        static_assert(std::is_same_v<
                      Kokkos::Array<std::remove_extent_t<decltype(som_array)>,
                                    std::size(array)>,
                      decltype(ka)>);

        for (size_t j = 0; j != ka.size(); ++j) {
          asum += array[j];
          somsum += som_array[j];
          kasum += ka[j];
        }
      },
      a_sum, som_sum, ka_sum);

  ASSERT_EQ(som_sum, -1 * static_cast<int>(std::size(array)));
  ASSERT_EQ(ka_sum, a_sum);
}

}  // namespace
