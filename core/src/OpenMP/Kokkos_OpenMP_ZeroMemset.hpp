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

#ifndef KOKKOS_OPENMP_ZEROMEMSET_HPP
#define KOKKOS_OPENMP_ZEROMEMSET_HPP

#include <Kokkos_Macros.hpp>
#include <OpenMP/Kokkos_OpenMP.hpp>
#include <OpenMP/Kokkos_OpenMP_Instance.hpp>
#include <impl/Kokkos_ZeroMemset_fwd.hpp>

namespace Kokkos {
namespace Impl {

template <>
struct ZeroMemset<OpenMP> {
  ZeroMemset(const OpenMP& exec_space, void* dst, size_t cnt) {
    if (cnt < 0x20000ul) {  // 2^17
      std::memset(dst, 0, cnt);
    } else {
      Kokkos::parallel_for(
          "Kokkos::ZeroMemset via parallel_for",
          Kokkos::RangePolicy<Kokkos::OpenMP, Kokkos::IndexType<size_t>>(
              exec_space, 0, cnt),
          KOKKOS_LAMBDA(size_t i) { static_cast<char*>(dst)[i] = 0; });
    }
  }
};

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_OPENMP_ZEROMEMSET_HPP
