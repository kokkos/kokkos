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

#ifndef KOKKOS_NUMANODE_HPP
#define KOKKOS_NUMANODE_HPP

#include <Kokkos_Macros.hpp>
#include <Kokkos_Array.hpp>

namespace Kokkos {

// Nodes ID mapping on either logical nodes or physical ones
// - If Physical: they will be used (mostly) as-is for memory binding
// - If Logical : they will be converted to a physical bitmap for memory binding
enum class ID_type : int { Physical, Logical };

template <unsigned N, Kokkos::ID_type id_type = Kokkos::ID_type::Physical>
struct NUMANodes: public Kokkos::Array<unsigned, N> {
  public:
    KOKKOS_INLINE_FUNCTION static constexpr bool is_physical() { return id_type == Kokkos::ID_type::Physical; }
};

template <typename T>
struct is_kokkos_numanodes : public std::false_type{};

template <unsigned N, Kokkos::ID_type id_type>
struct is_kokkos_numanodes<Kokkos::NUMANodes<N, id_type>> : public std::true_type{};

template <unsigned N>
struct is_kokkos_numanodes<Kokkos::NUMANodes<N>> : public std::true_type{};

template <typename T>
KOKKOS_INLINE_FUNCTION constexpr bool is_kokkos_numanodes_v = is_kokkos_numanodes<std::decay_t<T>>::value;

}  // namespace Kokkos

#endif /* #ifndef KOKKOS_NUMANODE_HPP */
