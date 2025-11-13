// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
import kokkos.core_impl;
#else
#include <Kokkos_Core.hpp>
#endif

namespace {

using test_atomic_view = Kokkos::View<double *, Kokkos::Serial,
                                      Kokkos::MemoryTraits<Kokkos::Atomic>>;
static_assert(std::is_same_v<
              decltype(std::declval<test_atomic_view>()(std::declval<int>())),
              desul::AtomicRef<double, desul::MemoryOrderRelaxed,
                               desul::MemoryScopeDevice>>);

}  // namespace