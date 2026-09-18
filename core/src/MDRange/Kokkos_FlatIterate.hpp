// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_MDRANGE_FLATITERATE_HPP
#define KOKKOS_MDRANGE_FLATITERATE_HPP

#include <type_traits>
#include <utility>
#include "../Kokkos_Array.hpp"
#include "../Kokkos_Layout.hpp"

namespace Kokkos::Impl {

template <class MDRP, class Functor, class Tag>
class FlatIterate;


template <typename IndexType, IndexType End, typename ItegerSequence>
struct make_reverse_integer_sequence_impl;

template <typename IndexType, IndexType End, IndexType... Indices>
struct make_reverse_integer_sequence_impl<
    IndexType, End, std::integer_sequence<IndexType, Indices...>>
    : std::type_identity<std::integer_sequence<IndexType, End - 1 - Indices...>> {};

template <typename IndexType, IndexType N>
using make_reverse_integer_sequence =
    typename make_reverse_integer_sequence_impl<
        IndexType, N, std::make_integer_sequence<IndexType, N>>::type;
}  // namespace Kokkos::Impl

#endif  // KOKKOS_MDRANGE_FLATITERATE_HPP
