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

#ifndef KOKKOS_SIMD_PERFTEST_OPERATORS_HPP
#define KOKKOS_SIMD_PERFTEST_OPERATORS_HPP

#include <Kokkos_SIMD.hpp>
#include <SIMDTesting_Ops.hpp>

#if defined(KOKKOS_ENABLE_DEPRECATED_CODE_4)
#if defined(KOKKOS_ENABLE_DEPRECATION_WARNINGS)
KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_PUSH()
#endif

class hmin {
 public:
  template <typename T, typename U, typename MaskType>
  auto on_host(T const& a, U, MaskType const& mask) const {
    return Kokkos::Experimental::hmin(where(mask, a));
  }

  template <typename T, typename U, typename MaskType>
  KOKKOS_INLINE_FUNCTION auto on_device(T const& a, U,
                                        MaskType const& mask) const {
    return Kokkos::Experimental::hmin(where(mask, a));
  }
};

class hmax {
 public:
  template <typename T, typename U, typename MaskType>
  auto on_host(T const& a, U, MaskType const& mask) const {
    return Kokkos::Experimental::hmax(where(mask, a));
  }

  template <typename T, typename U, typename MaskType>
  KOKKOS_INLINE_FUNCTION auto on_device(T const& a, U,
                                        MaskType const& mask) const {
    return Kokkos::Experimental::hmax(where(mask, a));
  }
};

#if defined(KOKKOS_ENABLE_DEPRECATION_WARNINGS)
KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_POP()
#endif
#endif

// need to redefine reduce and masked_reduce ops because the ones in the testing
// code are templated
class reduce_ {
 public:
  template <typename T, typename U, typename MaskType>
  auto on_host(T const& a, U, MaskType) const {
    return Kokkos::Experimental::reduce(a, std::plus<>());
  }
  template <typename T, typename U, typename MaskType>
  KOKKOS_INLINE_FUNCTION auto on_device(T const& a, U, MaskType) const {
    return Kokkos::Experimental::reduce(a, std::plus<>());
  }
};

class masked_reduce_ {
 public:
  template <typename T, typename MaskType>
  auto on_host(T const& a, typename T::value_type const identity,
               MaskType const& mask) const {
    return Kokkos::Experimental::reduce(a, mask, identity, std::plus<>());
  }

  template <typename T, typename MaskType>
  KOKKOS_INLINE_FUNCTION auto on_device(T const& a,
                                        typename T::value_type const identity,
                                        MaskType const& mask) const {
    return Kokkos::Experimental::reduce(a, mask, identity, std::plus<>());
  }
};

#endif
