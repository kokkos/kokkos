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

// Make sure Kokkos::restrict view has a restrict accessor

#include <Kokkos_Core.hpp>
#include <type_traits>
#include <view/AccessorAliases.hpp>

#ifndef KOKKOS_ENABLE_IMPL_VIEW_LEGACY

template <class T>
struct test_restrict_accessor {
  using view_type = Kokkos::View<T *, Kokkos::MemoryTraits<Kokkos::Restrict>>;
  using memory_space  = typename view_type::memory_space;
  using mdspan_type   = typename view_type::mdspan_type;
  using accessor_type = typename mdspan_type::accessor_type;
  using expected_accessor_type =
      CheckedReferenceCountedRestrictAccessor<T, memory_space>;

  constexpr static bool value =
      std::is_same_v<accessor_type, expected_accessor_type>;
};

static_assert(test_restrict_accessor<int>::value, "");
static_assert(test_restrict_accessor<double>::value, "");
static_assert(test_restrict_accessor<Kokkos::complex<double>>::value, "");

#endif
