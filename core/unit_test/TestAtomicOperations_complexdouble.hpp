// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <TestAtomicOperations.hpp>

using namespace TestAtomicOperations;

namespace Test {
TEST(TEST_CATEGORY, atomic_operations_complexdouble) {
#if defined(KOKKOS_ENABLE_SYCL) && \
    !defined(KOKKOS_IMPL_SYCL_DEVICE_GLOBAL_SUPPORTED)
  if (std::is_same_v<TEST_EXECSPACE, Kokkos::SYCL>)
    GTEST_SKIP() << "skipping since device_global variables are not available";
#endif
  const int start = -5;
  const int end   = 11;
  for (int i = start; i < end; ++i) {
    using T   = Kokkos::complex<double>;
    T old_val = static_cast<T>(i);
    T update  = static_cast<T>(end - i - start);
    ASSERT_TRUE(
        (atomic_op_test<AddAtomicTest, T, TEST_EXECSPACE>(old_val, update)));
    ASSERT_TRUE(
        (atomic_op_test<SubAtomicTest, T, TEST_EXECSPACE>(old_val, update)));
    ASSERT_TRUE(
        (atomic_op_test<MulAtomicTest, T, TEST_EXECSPACE>(old_val, update)));

    if (sizeof(void*) == 4) {
      // 32-bit x86 may do reference division in 80-bit x87, so allow up to
      // one ULP.
      ASSERT_TRUE((update != 0
                       ? atomic_op_test<DivAtomicTest, T, TEST_EXECSPACE, true>(
                             old_val, update)
                       : true));
    } else {
      ASSERT_TRUE((update != 0
                       ? atomic_op_test<DivAtomicTest, T, TEST_EXECSPACE>(
                             old_val, update)
                       : true));
    }
    ASSERT_TRUE((atomic_op_test<LoadStoreAtomicTest, T, TEST_EXECSPACE>(
        old_val, update)));
  }
}
}  // namespace Test
