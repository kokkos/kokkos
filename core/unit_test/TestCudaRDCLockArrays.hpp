// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_TEST_CUDA_RDC_LOCK_ARRAYS_HPP
#define KOKKOS_TEST_CUDA_RDC_LOCK_ARRAYS_HPP

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

namespace Test {
namespace RDCLockArrays {

// Kokkos::complex<double> is 16 bytes, so atomic_add on it is lock-based.
using value_type = Kokkos::complex<double>;
using view_type  = Kokkos::View<value_type, Kokkos::Cuda>;

// Defined in cuda/TestCuda_RDCLockArrays_DeviceFunction.cpp, which never
// launches a kernel.
KOKKOS_FUNCTION void atomic_add_elsewhere(value_type* dest, value_type val);

}  // namespace RDCLockArrays
}  // namespace Test

#endif
