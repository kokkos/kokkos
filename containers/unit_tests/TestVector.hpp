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

#ifndef KOKKOS_TEST_VECTOR_HPP
#define KOKKOS_TEST_VECTOR_HPP

#include <gtest/gtest.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <Kokkos_Macros.hpp>
KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_PUSH()
#include <Kokkos_Vector.hpp>

namespace Test {

namespace Impl {

template <typename Scalar, class Device>
struct test_vector_insert {
  using scalar_type     = Scalar;
  using execution_space = Device;

  template <typename Vector>
  void run_test(Vector& a) {
    auto n = a.size();

    auto it = a.begin();
    if (n > 0) {
      ASSERT_EQ(a.data(), &a[0]);
    }
    it += 15;
    ASSERT_EQ(*it, scalar_type(1));

    auto it_return = a.insert(it, scalar_type(3));
    ASSERT_EQ(a.size(), n + 1);
    ASSERT_EQ(std::distance(it_return, a.begin() + 15), 0);

    it = a.begin();
    it += 17;
    it_return = a.insert(it, n + 5, scalar_type(5));

    ASSERT_EQ(a.size(), n + 1 + n + 5);
    ASSERT_EQ(std::distance(it_return, a.begin() + 17), 0);

    Vector b;

    b.insert(b.begin(), 7, 9);
    ASSERT_EQ(b.size(), 7u);
    ASSERT_EQ(b[0], scalar_type(9));

    it = a.begin();
    it += 27 + n;
    it_return = a.insert(it, b.begin(), b.end());

    ASSERT_EQ(a.size(), n + 1 + n + 5 + 7);
    ASSERT_EQ(std::distance(it_return, a.begin() + 27 + n), 0);

    // Testing insert at end via all three function interfaces
    a.insert(a.end(), 11);
    a.insert(a.end(), 2, 12);
    a.insert(a.end(), b.begin(), b.end());
  }

  template <typename Vector>
  void check_test(Vector& a, int n) {
    for (int i = 0; i < (int)a.size(); i++) {
      if (i == 15)
        ASSERT_EQ(a[i], scalar_type(3));
      else if (i > 16 && i < 16 + 6 + n)
        ASSERT_EQ(a[i], scalar_type(5));
      else if (i > 26 + n && i < 34 + n)
        ASSERT_EQ(a[i], scalar_type(9));
      else if (i == (int)a.size() - 10)
        ASSERT_EQ(a[i], scalar_type(11));
      else if ((i == (int)a.size() - 9) || (i == (int)a.size() - 8))
        ASSERT_EQ(a[i], scalar_type(12));
      else if (i > (int)a.size() - 8)
        ASSERT_EQ(a[i], scalar_type(9));
      else
        ASSERT_EQ(a[i], scalar_type(1));
    }
  }

  test_vector_insert(unsigned int size) {
    {
      std::vector<Scalar> a(size, scalar_type(1));
      run_test(a);
      check_test(a, size);
    }
    {
      Kokkos::vector<Scalar, Device> a(size, scalar_type(1));
      a.sync_device();
      run_test(a);
      a.sync_host();
      check_test(a, size);
    }
    {
      Kokkos::vector<Scalar, Device> a(size, scalar_type(1));
      a.sync_host();
      run_test(a);
      check_test(a, size);
    }
    { test_vector_insert_into_empty(size); }
  }

  void test_vector_insert_into_empty(const size_t size) {
    using Vector = Kokkos::vector<Scalar, Device>;
    {
      Vector a;
      Vector b(size);
      a.insert(a.begin(), b.begin(), b.end());
      ASSERT_EQ(a.size(), size);
    }

    {
      Vector c;
      c.insert(c.begin(), size, Scalar{});
      ASSERT_EQ(c.size(), size);
    }
  }
};

template <typename Scalar, class Device>
struct test_vector_allocate {
  using self_type = test_vector_allocate<Scalar, Device>;

  using scalar_type     = Scalar;
  using execution_space = Device;

  bool result = false;

  template <typename Vector>
  Scalar run_me(unsigned int n) {
    {
      Vector v1;
      if (v1.is_allocated() == true) return false;

      v1 = Vector(n, 1);
      Vector v2(v1);
      Vector v3(n, 1);

      if (v1.is_allocated() == false) return false;
      if (v2.is_allocated() == false) return false;
      if (v3.is_allocated() == false) return false;
    }
    return true;
  }

  test_vector_allocate(unsigned int size) {
    result = run_me<Kokkos::vector<Scalar, Device> >(size);
  }
};

template <typename Scalar, class Device>
struct test_vector_combinations {
  using self_type = test_vector_combinations<Scalar, Device>;

  using scalar_type     = Scalar;
  using execution_space = Device;

  Scalar reference;
  Scalar result;

  template <typename Vector>
  Scalar run_me(unsigned int n) {
    Vector a(n, 1);

    a.push_back(2);
    a.resize(n + 4);
    a[n + 1] = 3;
    a[n + 2] = 4;
    a[n + 3] = 5;

    Scalar temp1 = a[2];
    Scalar temp2 = a[n];
    Scalar temp3 = a[n + 1];

    a.assign(n + 2, -1);

    a[2]     = temp1;
    a[n]     = temp2;
    a[n + 1] = temp3;

    Scalar test1 = 0;
    for (unsigned int i = 0; i < a.size(); i++) test1 += a[i];

    a.assign(n + 1, -2);
    Scalar test2 = 0;
    for (unsigned int i = 0; i < a.size(); i++) test2 += a[i];

    a.reserve(n + 10);

    Scalar test3 = 0;
    for (unsigned int i = 0; i < a.size(); i++) test3 += a[i];

    return (test1 * test2 + test3) * test2 + test1 * test3;
  }

  test_vector_combinations(unsigned int size) {
    reference = run_me<std::vector<Scalar> >(size);
    result    = run_me<Kokkos::vector<Scalar, Device> >(size);
  }
};

}  // namespace Impl

template <typename Scalar, typename Device>
void test_vector_combinations(unsigned int size) {
  Impl::test_vector_combinations<Scalar, Device> test(size);
  ASSERT_EQ(test.reference, test.result);
}

template <typename Scalar, typename Device>
void test_vector_allocate(unsigned int size) {
  Impl::test_vector_allocate<Scalar, Device> test(size);
  ASSERT_TRUE(test.result);
}

TEST(TEST_CATEGORY, vector_combination) {
  test_vector_allocate<int, TEST_EXECSPACE>(10);
  test_vector_combinations<int, TEST_EXECSPACE>(10);
  test_vector_combinations<long long int, TEST_EXECSPACE>(3057);
}

TEST(TEST_CATEGORY, vector_insert) {
  Impl::test_vector_insert<int, TEST_EXECSPACE>(3057);
}

// The particular scenario below triggered a bug where empty modified_flags
// would cause resize in push_back to be executed on the device overwriting the
// values that were stored on the host previously.
TEST(TEST_CATEGORY, vector_push_back_default_exec) {
  Kokkos::vector<int, TEST_EXECSPACE> V;
  V.clear();
  V.push_back(4);
  ASSERT_EQ(V[0], 4);
  V.push_back(3);
  ASSERT_EQ(V[1], 3);
  ASSERT_EQ(V[0], 4);
}

}  // namespace Test

#endif  // KOKKOS_TEST_UNORDERED_MAP_HPP
