// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>
#include <type_traits>

// Checking requirement of explict type conversion to View

namespace {

using T     = int;
using ptr_t = T*;

using view0_t = Kokkos::View<T>;
using view1_t = Kokkos::View<T*>;
using view2_t = Kokkos::View<T[4]>;

static_assert(!std::is_convertible_v<ptr_t, view0_t>);
static_assert(!std::is_convertible_v<ptr_t, view1_t>);
static_assert(!std::is_convertible_v<ptr_t, view2_t>);

}  // namespace
