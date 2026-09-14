// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_STD_HALF_IMPL_TYPE_HPP_
#define KOKKOS_STD_HALF_IMPL_TYPE_HPP_

#include <Kokkos_Macros.hpp>

#if defined(KOKKOS_IMPL_HALF_TYPE_STANDARD_SUPPORT) || \
    defined(KOKKOS_IMPL_BHALF_TYPE_STANDARD_SUPPORT)
#include <stdfloat>
#endif

namespace Kokkos::Impl {
#if defined(KOKKOS_IMPL_HALF_TYPE_STANDARD_SUPPORT)
struct half_impl_t {
  using type = std::float16_t;
};
#endif

#if defined(KOKKOS_IMPL_BHALF_TYPE_STANDARD_SUPPORT)
struct bhalf_impl_t {
  using type = std::bfloat16_t;
};
#endif
}  // namespace Kokkos::Impl

#endif  // KOKKOS_STD_HALF_IMPL_TYPE_HPP_
