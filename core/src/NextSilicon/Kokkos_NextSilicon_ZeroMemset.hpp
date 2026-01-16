// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_NEXTSILICONSPACE_ZEROMEMSET_HPP
#define KOKKOS_NEXTSILICONSPACE_ZEROMEMSET_HPP

#include <nsapi/memory.h>

#include <Kokkos_Macros.hpp>
#include "Kokkos_NextSiliconSpace.hpp"
#include <impl/Kokkos_ZeroMemset_fwd.hpp>

namespace Kokkos {
namespace Impl {

void ZeroMemsetNextSilicon(void* buffer, size_t buffer_size);

template <>
struct ZeroMemset<Kokkos::Experimental::NextSilicon> {
  ZeroMemset(const Kokkos::Experimental::NextSilicon& exec_space, void* dst,
             size_t cnt) {
    ZeroMemsetNextSilicon(dst, cnt);
  }
};

}  // end namespace Impl
}  // end namespace Kokkos

#endif  // KOKKOS_NEXTSILICONSPACE_ZEROMEMSET_HPP
