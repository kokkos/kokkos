// Copyright Contributors to the Kokkos project
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef KOKKOS_OPENACC_MACROS_HPP
#define KOKKOS_OPENACC_MACROS_HPP

#define KOKKOS_IMPL_ACC_PRAGMA_HELPER(x) _Pragma(#x)
#define KOKKOS_IMPL_ACC_PRAGMA(x) KOKKOS_IMPL_ACC_PRAGMA_HELPER(acc x)

#endif
