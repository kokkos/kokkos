// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#define KOKKOS_IMPL_PUBLIC_INCLUDE

#include <nsapi/intrinsics.h>
#include <nsapi/memory.h>

#include <NextSilicon/Kokkos_NextSilicon.hpp>
#include <NextSilicon/Kokkos_NextSiliconSpace.hpp>
#include <NextSilicon/Kokkos_NextSilicon_ZeroMemset.hpp>
#include <NextSilicon/Kokkos_NextSilicon_Instance.hpp>

#ifdef KOKKOS_ENABLE_NEXTSILICON

namespace Kokkos {
namespace Impl {
void ZeroMemsetNextSilicon(void* buffer, size_t buffer_size) {
  uint8_t pattern = 0;
  nsapi_memory_fill(buffer, &pattern, sizeof(pattern), buffer_size);
}

}  // namespace Impl
}  // namespace Kokkos

#endif
