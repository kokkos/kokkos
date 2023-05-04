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

#ifndef KOKKOS_ZEROMEMSET_HPP
#define KOKKOS_ZEROMEMSET_HPP

#include <Kokkos_Macros.hpp>
#include <impl/Kokkos_ZeroMemset_fwd.hpp>
#include <Kokkos_View.hpp>

namespace Kokkos {
namespace Impl {

// Default implementation for execution spaces that don't provide a definition
template <typename ExecutionSpace, class DT, class... DP>
struct ZeroMemset {
  ZeroMemset(const ExecutionSpace& exec_space, const View<DT, DP...>& dst,
             typename ViewTraits<DT, DP...>::const_value_type& value) {
    contiguous_fill(exec_space, dst, value);
  }

  ZeroMemset(const View<DT, DP...>& dst,
             typename ViewTraits<DT, DP...>::const_value_type& value) {
    contiguous_fill(ExecutionSpace(), dst, value);
  }
};

// Default HostSpace implementation
template <class DT, class... DP>
struct ZeroMemset<typename HostSpace::execution_space, DT, DP...> {
  ZeroMemset(const typename HostSpace::execution_space& exec,
             const View<DT, DP...>& dst,
             typename View<DT, DP...>::const_value_type&) {
    // Host spaces, except for HPX, are synchronous and we need to fence for HPX
    // since we can't properly enqueue a std::memset otherwise.
    // We can't use exec.fence() directly since we don't have a full definition
    // of HostSpace here.
    hostspace_fence(exec);
    using ValueType = typename View<DT, DP...>::value_type;
    std::memset(dst.data(), 0, sizeof(ValueType) * dst.size());
  }

  ZeroMemset(const View<DT, DP...>& dst,
             typename View<DT, DP...>::const_value_type&) {
    using ValueType = typename View<DT, DP...>::value_type;
    std::memset(dst.data(), 0, sizeof(ValueType) * dst.size());
  }
};

}  // end namespace Impl
}  // end namespace Kokkos

#endif  // KOKKOS_ZEROMEMSET_HPP
