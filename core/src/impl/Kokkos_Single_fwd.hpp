// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#include <Kokkos_Macros.hpp>
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif

#ifndef KOKKOS_SINGLE_FWD_HPP
#define KOKKOS_SINGLE_FWD_HPP

namespace Kokkos::Impl {
  template<class FunctorType, class ExecutionPolicy, class ExecutionSpace = typename ExecutionPolicy::execution_space>
  struct Single;
  template<class FunctorType, class ExecutionPolicy, class ReturnViewType, class ExecutionSpace = typename ExecutionPolicy::execution_space>
  struct SingleWithReturn;
}

#endif // KOKKOS_SINGLE_FWD_HPP
