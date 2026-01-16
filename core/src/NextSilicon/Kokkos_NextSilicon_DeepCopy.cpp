// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#define KOKKOS_IMPL_PUBLIC_INCLUDE

#include <nsapi/intrinsics.h>
#include <nsapi/memory.h>

#include <NextSilicon/Kokkos_NextSilicon.hpp>
#include <NextSilicon/Kokkos_NextSiliconSpace.hpp>
#include <NextSilicon/Kokkos_NextSilicon_DeepCopy.hpp>
#include <NextSilicon/Kokkos_NextSilicon_Instance.hpp>

#ifdef KOKKOS_ENABLE_NEXTSILICON

namespace Kokkos {
namespace Impl {
void DeepCopySharedNextSilicon(void* dst, const void* src, size_t n) {
  nsapi_memory_copy(dst, src, n);
}

void DeepCopyDeviceNextSilicon(void* dst, const void* src, size_t n) {
  nsapi_memory_copy(dst, src, n);
}

}  // namespace Impl
}  // namespace Kokkos

#endif
