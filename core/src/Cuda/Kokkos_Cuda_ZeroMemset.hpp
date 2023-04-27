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
#ifndef KOKKOS_CUDA_ZEROMEMSET_HPP
#define KOKKOS_CUDA_ZEROMEMSET_HPP

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_CUDA

#include <impl/Kokkos_ZeroMemset_fwd.hpp>

namespace Kokkos {
namespace Impl {

template <class DT, class... DP>
struct ZeroMemset<Kokkos::Cuda, DT, DP...> {
  ZeroMemset(const Kokkos::Cuda& exec_space_instance,
             const View<DT, DP...>& dst,
             typename View<DT, DP...>::const_value_type&) {
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMemsetAsync(
        dst.data(), 0,
        dst.size() * sizeof(typename View<DT, DP...>::value_type),
        exec_space_instance.cuda_stream()));
  }

  ZeroMemset(const View<DT, DP...>& dst,
             typename View<DT, DP...>::const_value_type&) {
    KOKKOS_IMPL_CUDA_SAFE_CALL(
        cudaMemset(dst.data(), 0,
                   dst.size() * sizeof(typename View<DT, DP...>::value_type)));
  }
};

}  // namespace Impl
}  // namespace Kokkos

#endif  // defined(KOKKOS_ENABLE_CUDA)
#endif  // !defined(KOKKOS_CUDA_ZEROMEMSET_HPP)
