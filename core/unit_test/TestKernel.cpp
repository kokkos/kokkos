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

#include <Kokkos_Core.hpp>

extern "C" double time_step(size_t N, double dx, const Kokkos::View<double*>& u,
                            const Kokkos::View<double*>& c) {
  double dt_global = INFINITY;

  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<>(0, N),
      KOKKOS_LAMBDA(const Kokkos::RangePolicy<>::index_type i, double& dt) {
        double max_cx =
            Kokkos::max(Kokkos::abs(u[i] + c[i]), Kokkos::abs(u[i] - c[i]));
        dt = Kokkos::min(dt, dx / max_cx);
      },
      Kokkos::Min<double>(dt_global));

  return dt_global;
}
