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

#ifndef ACCESSOR_ALIASES_HPP
#define ACCESSOR_ALIASES_HPP

#include <Kokkos_Core.hpp>

// shorthand for some accessors
template <class ElementType, class MemorySpace>
using CheckedReferenceCountedAccessor = Kokkos::Impl::SpaceAwareAccessor<
    MemorySpace,
    Kokkos::Impl::ReferenceCountedAccessor<
        ElementType, MemorySpace, Kokkos::default_accessor<ElementType>>>;

template <class ElementType, class MemorySpace,
          class MemoryScope = desul::MemoryScopeDevice>
using CheckedRelaxedAtomicAccessor = Kokkos::Impl::SpaceAwareAccessor<
    MemorySpace, Kokkos::Impl::AtomicAccessorRelaxed<ElementType>>;

template <class ElementType, class MemorySpace,
          class MemoryScope = desul::MemoryScopeDevice>
using CheckedReferenceCountedRelaxedAtomicAccessor =
    Kokkos::Impl::SpaceAwareAccessor<
        MemorySpace, Kokkos::Impl::ReferenceCountedAccessor<
                         ElementType, MemorySpace,
                         Kokkos::Impl::AtomicAccessorRelaxed<ElementType>>>;

template <class ElementType, class MemorySpace>
using CheckedReferenceCountedRestrictAccessor =
    Kokkos::Impl::SpaceAwareAccessor<
        MemorySpace, Kokkos::Impl::ReferenceCountedAccessor<
                         ElementType, MemorySpace,
                         Kokkos::Impl::RestrictAccessor<ElementType>>>;

#endif
