// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>
#include <type_traits>

// Checking requirement of explict type conversion to View

namespace {

using T     = int;
static_assert(!std::is_convertible_v<T*, Kokkos::View<T>>);
static_assert(!std::is_convertible_v<T*, Kokkos::View<T*>>);
static_assert(!std::is_convertible_v<T*, Kokkos::View<T[4]>>);

}  // namespace
