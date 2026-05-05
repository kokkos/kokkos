// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#endif

#include <HIP/Kokkos_HIP_ZeroMemset.hpp>
#include <HIP/Kokkos_HIP_ParallelFor_Range.hpp>

#include <cstdint>
#include <limits>

namespace Kokkos {
namespace Impl {

template <typename data_type, typename index_type>
struct ZeroMemsetKernel {
  using policy_type =
      Kokkos::RangePolicy<Kokkos::HIP, Kokkos::IndexType<index_type>,
                          Kokkos::Experimental::StaticBatchSize<4>>;

  ZeroMemsetKernel(const HIP& exec_space, void* dst, size_t cnt) {
    Kokkos::parallel_for(
        "Kokkos::ZeroMemset via parallel_for", policy_type(exec_space, 0, cnt),
        KOKKOS_LAMBDA(const index_type i) {
          static_cast<data_type*>(dst)[i] = data_type{0};
        });
  }
};

// alternative to hipMemsetAsync, which sets the first `cnt` bytes of `dst` to 0
void zero_with_hip_kernel(const HIP& exec_space, void* dst, size_t cnt) {
  if (cnt < static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    if (cnt % 8 == 0 && reinterpret_cast<uintptr_t>(dst) % 8 == 0) {
      ZeroMemsetKernel<uint64_t, uint32_t>(exec_space, dst, cnt / 8);
    } else if (cnt % 4 == 0 && reinterpret_cast<uintptr_t>(dst) % 4 == 0) {
      ZeroMemsetKernel<uint32_t, uint32_t>(exec_space, dst, cnt / 4);
    } else if (cnt % 2 == 0 && reinterpret_cast<uintptr_t>(dst) % 2 == 0) {
      ZeroMemsetKernel<uint16_t, uint32_t>(exec_space, dst, cnt / 2);
    } else {
      ZeroMemsetKernel<uint8_t, uint32_t>(exec_space, dst, cnt);
    }
  } else {
    if (cnt % 8 == 0 && reinterpret_cast<uintptr_t>(dst) % 8 == 0) {
      ZeroMemsetKernel<uint64_t, uint64_t>(exec_space, dst, cnt / 8);
    } else if (cnt % 4 == 0 && reinterpret_cast<uintptr_t>(dst) % 4 == 0) {
      ZeroMemsetKernel<uint32_t, uint64_t>(exec_space, dst, cnt / 4);
    } else if (cnt % 2 == 0 && reinterpret_cast<uintptr_t>(dst) % 2 == 0) {
      ZeroMemsetKernel<uint16_t, uint64_t>(exec_space, dst, cnt / 2);
    } else {
      ZeroMemsetKernel<uint8_t, uint64_t>(exec_space, dst, cnt);
    }
  }
}

}  // namespace Impl
}  // namespace Kokkos
