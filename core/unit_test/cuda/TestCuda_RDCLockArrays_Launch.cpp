// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <TestCuda_Category.hpp>
#include <TestCudaRDCLockArrays.hpp>

namespace Test {
namespace RDCLockArrays {

// The kernel is launched from this translation unit, but the lock-based
// atomic runs inside a device function defined in another one.
void launch_calling_device_function_elsewhere(view_type data, int n) {
  Kokkos::parallel_for(
      Kokkos::RangePolicy<TEST_EXECSPACE>(0, n), KOKKOS_LAMBDA(int) {
        atomic_add_elsewhere(data.data(), value_type{1., 1.});
      });
}

TEST(TEST_CATEGORY, rdc_cross_tu_device_function_lock_based_atomic) {
  view_type data("data");
  launch_calling_device_function_elsewhere(data, 100);

  value_type data_h;
  Kokkos::deep_copy(data_h, data);
  ASSERT_FLOAT_EQ(data_h.real(), 100.);
  ASSERT_FLOAT_EQ(data_h.imag(), 100.);
}

}  // namespace RDCLockArrays
}  // namespace Test
