// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef TESTHALFCONVERSION_HPP_
#define TESTHALFCONVERSION_HPP_

namespace Test {

template <class T>
void test_half_conversion_type() {
  // When truncating mantissa to 10bits (like f16), 3.3 becomes 3.298828125
  // 3.3 - 3.298828125 < 1.1719e-3, so conversion error should be smaller
  double epsilon = KOKKOS_HALF_T_IS_FLOAT ? 3e-7 : 1.1719e-3;
  Kokkos::Array<T, 5> test_values({T(0), T(.1), T(3.3)});
  for (T test_value : test_values) {
    Kokkos::Experimental::half_t a =
        Kokkos::Experimental::cast_to_half(test_value);
    T b = Kokkos::Experimental::cast_from_half<T>(a);
    ASSERT_NEAR(b, test_value, epsilon);
  }

  auto test_values_device = Kokkos::create_mirror_view_and_copy(
      Kokkos::DefaultExecutionSpace::memory_space{},
      Kokkos::View<T*, Kokkos::HostSpace>(test_values.data(),
                                          test_values.size()));
  Kokkos::View<T*> b_v("b_v", test_values.size());
  Kokkos::parallel_for(
      "TestHalfConversion", test_values.size(), KOKKOS_LAMBDA(int i) {
        Kokkos::Experimental::half_t d_a =
            Kokkos::Experimental::cast_to_half(test_values_device(i));
        b_v(i) = Kokkos::Experimental::cast_from_half<T>(d_a);
      });

  auto results_host = Kokkos::create_mirror_view_and_copy(b_v);
  for (unsigned int i = 0; i < test_values.size(); ++i)
    ASSERT_NEAR(results_host(i), test_values[i], epsilon);
}

template <class T>
void test_bhalf_conversion_type() {
  // When truncating mantissa to 7bits (like b16), 3.3 becomes 3.296875
  // 3.3 - 3.296875 < 3.125e-3, so conversion error should be smaller
  double epsilon = KOKKOS_BHALF_T_IS_FLOAT ? 3e-7 : 3.125e-3;
  Kokkos::Array<T, 5> test_values({T(0), T(.1), T(3.3)});
  for (T test_value : test_values) {
    Kokkos::Experimental::bhalf_t a =
        Kokkos::Experimental::cast_to_bhalf(test_value);
    T b = Kokkos::Experimental::cast_from_bhalf<T>(a);
    ASSERT_NEAR(b, test_value, epsilon);
  }

  auto test_values_device = Kokkos::create_mirror_view_and_copy(
      Kokkos::DefaultExecutionSpace::memory_space{},
      Kokkos::View<T*, Kokkos::HostSpace>(test_values.data(),
                                          test_values.size()));
  Kokkos::View<T*> b_v("b_v", test_values.size());
  Kokkos::parallel_for(
      "TestHalfConversion", test_values.size(), KOKKOS_LAMBDA(int i) {
        Kokkos::Experimental::bhalf_t d_a =
            Kokkos::Experimental::cast_to_bhalf(test_values_device(i));
        b_v(i) = Kokkos::Experimental::cast_from_bhalf<T>(d_a);
      });

  auto results_host = Kokkos::create_mirror_view_and_copy(b_v);
  for (unsigned int i = 0; i < test_values.size(); ++i)
    ASSERT_NEAR(results_host(i), test_values[i], epsilon);
}

void test_half_conversion() {
  test_half_conversion_type<float>();
  test_half_conversion_type<double>();
  test_half_conversion_type<short>();
  test_half_conversion_type<bool>();
  test_half_conversion_type<int>();
  test_half_conversion_type<long>();
  test_half_conversion_type<long long>();
  test_half_conversion_type<unsigned short>();
  test_half_conversion_type<unsigned int>();
  test_half_conversion_type<unsigned long>();
  test_half_conversion_type<unsigned long long>();
}

void test_bhalf_conversion() {
  test_bhalf_conversion_type<float>();
  test_bhalf_conversion_type<double>();
  test_bhalf_conversion_type<short>();
  test_bhalf_conversion_type<bool>();
  test_bhalf_conversion_type<int>();
  test_bhalf_conversion_type<long>();
  test_bhalf_conversion_type<long long>();
  test_bhalf_conversion_type<unsigned short>();
  test_bhalf_conversion_type<unsigned int>();
  test_bhalf_conversion_type<unsigned long>();
  test_bhalf_conversion_type<unsigned long long>();
}

TEST(TEST_CATEGORY, half_conversion) { test_half_conversion(); }

TEST(TEST_CATEGORY, bhalf_conversion) { test_bhalf_conversion(); }

}  // namespace Test
#endif
