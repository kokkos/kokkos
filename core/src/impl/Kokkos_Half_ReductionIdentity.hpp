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

#ifndef KOKKOS_HALF_REDUCTION_IDENTITY_HPP_
#define KOKKOS_HALF_REDUCTION_IDENTITY_HPP_

#include <Kokkos_ReductionIdentity.hpp>

#if defined(KOKKOS_HALF_T_IS_FLOAT) && !KOKKOS_HALF_T_IS_FLOAT
template <>
struct Kokkos::reduction_identity<Kokkos::Experimental::half_t> {
  KOKKOS_FUNCTION static Experimental::half_t sum() noexcept { return 0; }
  KOKKOS_FUNCTION static Experimental::half_t prod() noexcept { return 1; }
  KOKKOS_FUNCTION static Experimental::half_t max() noexcept {
    using namespace Kokkos::Experimental;
#if __FINITE_MATH_ONLY__
    return finite_min_v<half_t>;
#else
    return -min();
#endif
  }
  KOKKOS_FUNCTION static Experimental::half_t min() noexcept {
    using namespace Kokkos::Experimental;
#if __FINITE_MATH_ONLY__
    return finite_max_v<half_t>;
#else
    return infinity_v<half_t>;
#endif
  }
};
#endif

#if defined(KOKKOS_BHALF_T_IS_FLOAT) && !KOKKOS_BHALF_T_IS_FLOAT
template <>
struct Kokkos::reduction_identity<Kokkos::Experimental::bhalf_t> {
  KOKKOS_FUNCTION static Experimental::bhalf_t sum() noexcept { return 0; }
  KOKKOS_FUNCTION static Experimental::bhalf_t prod() noexcept { return 1; }
  KOKKOS_FUNCTION static Experimental::bhalf_t max() noexcept {
    using namespace Kokkos::Experimental;
#if __FINITE_MATH_ONLY__
    return finite_min_v<bhalf_t>;
#else
    return -min();
#endif
  }
  KOKKOS_FUNCTION static Experimental::bhalf_t min() noexcept {
    using namespace Kokkos::Experimental;
#if __FINITE_MATH_ONLY__
    return finite_max_v<bhalf_t>;
#else
    return infinity_v<bhalf_t>;
#endif
  }
};
#endif

#endif
