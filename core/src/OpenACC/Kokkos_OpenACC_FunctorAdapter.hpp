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

#ifndef KOKKOS_OPENACC_FUNCTOR_ADAPTER_HPP
#define KOKKOS_OPENACC_FUNCTOR_ADAPTER_HPP

#include <type_traits>

namespace Kokkos::Experimental::Impl {

template <class Functor, class Policy>
class FunctorAdapter {
  Functor m_functor;
  using WorkTag = typename Policy::work_tag;

 public:
  FunctorAdapter(Functor const &functor) : m_functor(functor) {}

  template <class... Args>
  KOKKOS_FUNCTION void operator()(Args &&... args) const {
    if constexpr (std::is_void_v<WorkTag>) {
      m_functor(static_cast<Args &&>(args)...);
    } else {
      m_functor(WorkTag(), static_cast<Args &&>(args)...);
    }
  }
};

}  // namespace Kokkos::Experimental::Impl

#endif
