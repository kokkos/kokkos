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

#ifndef KOKKOS_SORT_FREE_FUNCTION_HPP_
#define KOKKOS_SORT_FREE_FUNCTION_HPP_

#include "impl/Kokkos_SortFreeFunction.hpp"
#include <Kokkos_Core.hpp>
#include <std_algorithms/Kokkos_BeginEnd.hpp>
#include <std_algorithms/Kokkos_Copy.hpp>
#include <algorithm>

namespace Kokkos {

// clang-format off
template <class ExecutionSpace, class DataType, class... Properties>
void sort([[maybe_unused]] const ExecutionSpace& exec,
          const Kokkos::View<DataType, Properties...>& view)
{
  if (view.extent(0) == 0) {
    return;
  }

  using ViewType = Kokkos::View<DataType, Properties...>;
  using MemSpace = typename ViewType::memory_space;
  static_assert(
      SpaceAccessibility<ExecutionSpace, MemSpace>::accessible,
      "Kokkos::sort: execution space instance is not able to access the memory space of the "
      "View argument!");

  if constexpr (SpaceAccessibility<HostSpace, MemSpace>::accessible) {
    auto first = ::Kokkos::Experimental::begin(view);
    auto last  = ::Kokkos::Experimental::end(view);
    std::sort(first, last);
  } else {
    Impl::sort_without_comparator(exec, view);
  }
}

// clang-format off
template <class DataType, class... Properties>
void sort(const Kokkos::View<DataType, Properties...>& view)
{
  if (view.extent(0) == 0) {
    return;
  }

  using ViewType = Kokkos::View<DataType, Properties...>;
  typename ViewType::execution_space exec;
  sort(exec, view);
}


// clang-format off
template <class ExecutionSpace, class CompType, class DataType,
          class... Properties>
void sort([[maybe_unused]] const ExecutionSpace& exec,
          const Kokkos::View<DataType, Properties...>& view,
          CompType const& comp)
{
  if (view.extent(0) == 0) {
    return;
  }

  using ViewType = Kokkos::View<DataType, Properties...>;
  using MemSpace = typename ViewType::memory_space;
  static_assert(
      SpaceAccessibility<ExecutionSpace, MemSpace>::accessible,
      "Kokkos::sort: execution space instance is not able to access the memory space of the "
      "View argument!");

  static_assert(
      ViewType::rank == 1 &&
          (std::is_same_v<typename ViewType::array_layout, LayoutRight> ||
           std::is_same_v<typename ViewType::array_layout, LayoutLeft> ||
	   std::is_same_v<typename ViewType::array_layout, LayoutStride>),
      "sort: only supports 1D Views with LayoutRight, LayoutLeft or LayoutStride.");

  if constexpr (SpaceAccessibility<HostSpace, MemSpace>::accessible) {
    auto first = ::Kokkos::Experimental::begin(view);
    auto last  = ::Kokkos::Experimental::end(view);
    std::sort(first, last, comp);
  } else {
    Impl::sort_device_view_with_comparator(exec, view, comp);
  }
}

template <class CompType, class DataType, class... Properties>
void sort(const Kokkos::View<DataType, Properties...>& view,
          CompType const& comp)
{
  if (view.extent(0) == 0) {
    return;
  }

  using ViewType = Kokkos::View<DataType, Properties...>;
  typename ViewType::execution_space exec;
  sort(exec, view, comp);
}

//
// subrange via integers begin, end
//
// clang-format off
template <class ExecutionSpace, class ViewType>
std::enable_if_t<
  Kokkos::is_execution_space<ExecutionSpace>::value
  && (is_view_v<ViewType> || is_dynamic_view_v<ViewType>)
  >
sort(const ExecutionSpace& exec,
     ViewType view,
     size_t const begin,
     size_t const end)
{
  if (view.extent(0) == 0) { return; }
  using MemSpace = typename ViewType::memory_space;
  static_assert(
      SpaceAccessibility<ExecutionSpace, MemSpace>::accessible,
      "Kokkos::sort: execution space instance is not able to access the memory space of the "
      "View argument!");

  using range_policy = Kokkos::RangePolicy<typename ViewType::execution_space>;
  using CompType     = BinOp1D<ViewType>;
  Kokkos::MinMaxScalar<typename ViewType::non_const_value_type> result;
  Kokkos::MinMax<typename ViewType::non_const_value_type> reducer(result);
  parallel_reduce("Kokkos::Sort::FindExtent", range_policy(exec, begin, end),
                  Impl::min_max_functor<ViewType>(view), reducer);
  if (result.min_val == result.max_val) return;

  BinSort<ViewType, CompType> bin_sort(
      exec, view, begin, end,
      CompType((end - begin) / 2, result.min_val, result.max_val), true);

  bin_sort.create_permute_vector(exec);
  bin_sort.sort(exec, view, begin, end);
}

// clang-format off
template <class ViewType>
std::enable_if_t<
  is_view_v<ViewType> || is_dynamic_view_v<ViewType>
  >
sort(ViewType view,
     size_t const begin,
     size_t const end)
{
  if (view.extent(0) == 0) { return; }

  using ViewType = Kokkos::View<DataType, Properties...>;
  Kokkos::fence("Kokkos::sort: before");
  typename ViewType::execution_space exec;
  sort(exec, view, begin, end);
  exec.fence("Kokkos::Sort: fence after sorting");
}

}  // namespace Kokkos

#endif
