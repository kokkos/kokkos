// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_SETUP_NEXTSILICON_HPP_
#define KOKKOS_SETUP_NEXTSILICON_HPP_

#if !defined(KOKKOS_ENABLE_NEXTSILICON)
#error \
    "KOKKOS_ENABLE_NEXTSILICON was not defined, but Kokkos_Setup_NextSilicon.hpp was included anyway."
#endif

// FIXME_NEXTSILICON: Marking functions as weak prevents host code from being
// inlined. If host code is inlined or speculated, it may prevent offloading to
// the device.
#define KOKKOS_IMPL_HOST_FUNCTION __attribute__((weak))

#endif
