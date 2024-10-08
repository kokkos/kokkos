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

#ifndef KOKKOS_HOSTSPACE_ZEROMEMSET_HPP
#define KOKKOS_HOSTSPACE_ZEROMEMSET_HPP

#include <Kokkos_Macros.hpp>
#include <Kokkos_HostSpace.hpp>
#include <impl/Kokkos_ZeroMemset_fwd.hpp>

namespace Kokkos {
namespace Impl {

template <>
struct ZeroMemset<HostSpace::execution_space> {
  ZeroMemset(const HostSpace::execution_space& exec, void* dst, size_t cnt) {
    // Host spaces, except for HPX, are synchronous and we need to fence for HPX
    // since we can't properly enqueue a std::memset otherwise.
    // We can't use exec.fence() directly since we don't have a full definition
    // of HostSpace here.
    hostspace_fence(exec);
    std::memset(dst, 0, cnt);
  }
};

}  // end namespace Impl
}  // end namespace Kokkos

#endif  // KOKKOS_HOSTSPACE_ZEROMEMSET_HPP
