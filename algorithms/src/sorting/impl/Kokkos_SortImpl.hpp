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

#ifndef KOKKOS_SORT_FREE_FUNCS_IMPL_HPP_
#define KOKKOS_SORT_FREE_FUNCS_IMPL_HPP_

#include "../Kokkos_BinOpsPublicAPI.hpp"
#include "../Kokkos_BinSortPublicAPI.hpp"
#include <std_algorithms/Kokkos_BeginEnd.hpp>
#include <std_algorithms/Kokkos_Copy.hpp>
#include <Kokkos_Core.hpp>

#if defined(KOKKOS_ENABLE_CUDA)

// Workaround for `Instruction 'shfl' without '.sync' is not supported on
// .target sm_70 and higher from PTX ISA version 6.4`.
// Also see https://github.com/NVIDIA/cub/pull/170.
#if !defined(CUB_USE_COOPERATIVE_GROUPS)
#define CUB_USE_COOPERATIVE_GROUPS
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wsuggest-override"

#if defined(KOKKOS_COMPILER_CLANG)
// Some versions of Clang fail to compile Thrust, failing with errors like
// this:
//    <snip>/thrust/system/cuda/detail/core/agent_launcher.h:557:11:
//    error: use of undeclared identifier 'va_printf'
// The exact combination of versions for Clang and Thrust (or CUDA) for this
// failure was not investigated, however even very recent version combination
// (Clang 10.0.0 and Cuda 10.0) demonstrated failure.
//
// Defining _CubLog here locally allows us to avoid that code path, however
// disabling some debugging diagnostics
#pragma push_macro("_CubLog")
#ifdef _CubLog
#undef _CubLog
#endif
// NOLINTNEXTLINE(bugprone-reserved-identifier)
#define _CubLog
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#pragma pop_macro("_CubLog")
#else
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#endif

#pragma GCC diagnostic pop

#endif

#if defined(KOKKOS_ENABLE_ROCTHRUST)
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#endif

#if defined(KOKKOS_ENABLE_ONEDPL)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-local-typedef"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#pragma GCC diagnostic pop

#define KOKKOS_IMPL_ONEDPL_VERSION                            \
  ONEDPL_VERSION_MAJOR * 10000 + ONEDPL_VERSION_MINOR * 100 + \
      ONEDPL_VERSION_PATCH
#define KOKKOS_IMPL_ONEDPL_VERSION_GREATER_EQUAL(MAJOR, MINOR, PATCH) \
  (KOKKOS_IMPL_ONEDPL_VERSION >= ((MAJOR)*10000 + (MINOR)*100 + (PATCH)))

namespace Kokkos::Impl {
template <typename Comparator, typename ValueType>
struct ComparatorWrapper {
  Comparator comparator;
  KOKKOS_FUNCTION bool operator()(const ValueType& i,
                                  const ValueType& j) const {
    return comparator(i, j);
  }
};
}  // namespace Kokkos::Impl

template <typename Comparator, typename ValueType>
struct sycl::is_device_copyable<
    Kokkos::Impl::ComparatorWrapper<Comparator, ValueType>> : std::true_type {};
#endif

namespace Kokkos {
namespace Impl {

template <class ExecutionSpace>
struct better_off_calling_std_sort : std::false_type {};

#if defined KOKKOS_ENABLE_SERIAL
template <>
struct better_off_calling_std_sort<Kokkos::Serial> : std::true_type {};
#endif

#if defined KOKKOS_ENABLE_OPENMP
template <>
struct better_off_calling_std_sort<Kokkos::OpenMP> : std::true_type {};
#endif

#if defined KOKKOS_ENABLE_THREADS
template <>
struct better_off_calling_std_sort<Kokkos::Threads> : std::true_type {};
#endif

#if defined KOKKOS_ENABLE_HPX
template <>
struct better_off_calling_std_sort<Kokkos::Experimental::HPX> : std::true_type {
};
#endif

template <class T>
inline constexpr bool better_off_calling_std_sort_v =
    better_off_calling_std_sort<T>::value;

template <class ViewType>
struct min_max_functor {
  using minmax_scalar =
      Kokkos::MinMaxScalar<typename ViewType::non_const_value_type>;

  ViewType view;
  min_max_functor(const ViewType& view_) : view(view_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t& i, minmax_scalar& minmax) const {
    if (view(i) < minmax.min_val) minmax.min_val = view(i);
    if (view(i) > minmax.max_val) minmax.max_val = view(i);
  }
};

template <class ExecutionSpace, class DataType, class... Properties>
void sort_via_binsort(const ExecutionSpace& exec,
                      const Kokkos::View<DataType, Properties...>& view) {
  // Although we are using BinSort below, which could work on rank-2 views,
  // for now view must be rank-1 because the min_max_functor
  // used below only works for rank-1 views
  using ViewType = Kokkos::View<DataType, Properties...>;
  static_assert(ViewType::rank == 1,
                "Kokkos::sort: currently only supports rank-1 Views.");

  if (view.extent(0) <= 1) {
    return;
  }

  Kokkos::MinMaxScalar<typename ViewType::non_const_value_type> result;
  Kokkos::MinMax<typename ViewType::non_const_value_type> reducer(result);
  parallel_reduce("Kokkos::Sort::FindExtent",
                  Kokkos::RangePolicy<typename ViewType::execution_space>(
                      exec, 0, view.extent(0)),
                  min_max_functor<ViewType>(view), reducer);
  if (result.min_val == result.max_val) return;
  // For integral types the number of bins may be larger than the range
  // in which case we can exactly have one unique value per bin
  // and then don't need to sort bins.
  bool sort_in_bins = true;
  // TODO: figure out better max_bins then this ...
  int64_t max_bins = view.extent(0) / 2;
  if (std::is_integral_v<typename ViewType::non_const_value_type>) {
    // Cast to double to avoid possible overflow when using integer
    auto const max_val = static_cast<double>(result.max_val);
    auto const min_val = static_cast<double>(result.min_val);
    // using 10M as the cutoff for special behavior (roughly 40MB for the count
    // array)
    if ((max_val - min_val) < 10000000) {
      max_bins     = max_val - min_val + 1;
      sort_in_bins = false;
    }
  }
  if (std::is_floating_point_v<typename ViewType::non_const_value_type>) {
    KOKKOS_ASSERT(std::isfinite(static_cast<double>(result.max_val) -
                                static_cast<double>(result.min_val)));
  }

  using CompType = BinOp1D<ViewType>;
  BinSort<ViewType, CompType> bin_sort(
      view, CompType(max_bins, result.min_val, result.max_val), sort_in_bins);
  bin_sort.create_permute_vector(exec);
  bin_sort.sort(exec, view);
}

#if defined(KOKKOS_ENABLE_CUDA)
template <class DataType, class... Properties, class... MaybeComparator>
void sort_cudathrust(const Cuda& space,
                     const Kokkos::View<DataType, Properties...>& view,
                     MaybeComparator&&... maybeComparator) {
  using ViewType = Kokkos::View<DataType, Properties...>;
  static_assert(ViewType::rank == 1,
                "Kokkos::sort: currently only supports rank-1 Views.");

  if (view.extent(0) <= 1) {
    return;
  }
  const auto exec = thrust::cuda::par.on(space.cuda_stream());
  auto first      = ::Kokkos::Experimental::begin(view);
  auto last       = ::Kokkos::Experimental::end(view);
  thrust::sort(exec, first, last,
               std::forward<MaybeComparator>(maybeComparator)...);
}
#endif

#if defined(KOKKOS_ENABLE_ROCTHRUST)
template <class DataType, class... Properties, class... MaybeComparator>
void sort_rocthrust(const HIP& space,
                    const Kokkos::View<DataType, Properties...>& view,
                    MaybeComparator&&... maybeComparator) {
  using ViewType = Kokkos::View<DataType, Properties...>;
  static_assert(ViewType::rank == 1,
                "Kokkos::sort: currently only supports rank-1 Views.");

  if (view.extent(0) <= 1) {
    return;
  }
  const auto exec = thrust::hip::par.on(space.hip_stream());
  auto first      = ::Kokkos::Experimental::begin(view);
  auto last       = ::Kokkos::Experimental::end(view);
  thrust::sort(exec, first, last,
               std::forward<MaybeComparator>(maybeComparator)...);
}
#endif

#if defined(KOKKOS_ENABLE_ONEDPL)
template <class DataType, class... Properties, class... MaybeComparator>
void sort_onedpl(const Kokkos::SYCL& space,
                 const Kokkos::View<DataType, Properties...>& view,
                 MaybeComparator&&... maybeComparator) {
  using ViewType = Kokkos::View<DataType, Properties...>;
  static_assert(SpaceAccessibility<Kokkos::SYCL,
                                   typename ViewType::memory_space>::accessible,
                "SYCL execution space is not able to access the memory space "
                "of the View argument!");

#if KOKKOS_IMPL_ONEDPL_VERSION_GREATER_EQUAL(2022, 8, 0)
  static_assert(ViewType::rank == 1,
                "Kokkos::sort currently only supports rank-1 Views.");
#else
  static_assert(
      (ViewType::rank == 1) &&
          (std::is_same_v<typename ViewType::array_layout, LayoutRight> ||
           std::is_same_v<typename ViewType::array_layout, LayoutLeft> ||
           std::is_same_v<typename ViewType::array_layout, LayoutStride>),
      "SYCL sort only supports contiguous rank-1 Views with LayoutLeft, "
      "LayoutRight or LayoutStride"
      "For the latter, this means the View must have stride(0) = 1, enforced "
      "at runtime.");

  if (view.stride(0) != 1) {
    Kokkos::abort("SYCL sort only supports rank-1 Views with stride(0) = 1.");
  }
#endif

  if (view.extent(0) <= 1) {
    return;
  }

  auto queue  = space.sycl_queue();
  auto policy = oneapi::dpl::execution::make_device_policy(queue);

#if KOKKOS_IMPL_ONEDPL_VERSION_GREATER_EQUAL(2022, 8, 0)
  auto view_begin = ::Kokkos::Experimental::begin(view);
  auto view_end   = ::Kokkos::Experimental::end(view);
#else
  // Can't use Experimental::begin/end here since the oneDPL then assumes that
  // the data is on the host.
  const int n     = view.extent(0);
  auto view_begin = view.data();
  auto view_end   = view.data() + n;
#endif

  if constexpr (sizeof...(MaybeComparator) == 0)
    oneapi::dpl::sort(policy, view_begin, view_end);
  else {
    using value_type =
        typename Kokkos::View<DataType, Properties...>::value_type;
    auto comparator =
        std::get<0>(std::tuple<MaybeComparator...>(maybeComparator...));
    oneapi::dpl::sort(
        policy, view_begin, view_end,
        ComparatorWrapper<decltype(comparator), value_type>{comparator});
  }
}
#endif

template <class ExecutionSpace, class DataType, class... Properties,
          class... MaybeComparator>
void copy_to_host_run_stdsort_copy_back(
    const ExecutionSpace& exec,
    const Kokkos::View<DataType, Properties...>& view,
    MaybeComparator&&... maybeComparator) {
  namespace KE = ::Kokkos::Experimental;

  using ViewType = Kokkos::View<DataType, Properties...>;
  using layout   = typename ViewType::array_layout;
  if constexpr (std::is_same_v<LayoutStride, layout>) {
    // for strided views we cannot just deep_copy from device to host,
    // so we need to do a few more jumps
    using view_value_type      = typename ViewType::non_const_value_type;
    using view_exespace        = typename ViewType::execution_space;
    using view_deep_copyable_t = Kokkos::View<view_value_type*, view_exespace>;
    view_deep_copyable_t view_dc("view_dc", view.extent(0));
    KE::copy(exec, view, view_dc);

    // run sort on the mirror of view_dc
    auto mv_h  = create_mirror_view_and_copy(Kokkos::HostSpace(), view_dc);
    auto first = KE::begin(mv_h);
    auto last  = KE::end(mv_h);
    std::sort(first, last, std::forward<MaybeComparator>(maybeComparator)...);
    Kokkos::deep_copy(exec, view_dc, mv_h);

    // copy back to argument view
    KE::copy(exec, KE::cbegin(view_dc), KE::cend(view_dc), KE::begin(view));
  } else {
    auto view_h = create_mirror_view_and_copy(Kokkos::HostSpace(), view);
    auto first  = KE::begin(view_h);
    auto last   = KE::end(view_h);
    std::sort(first, last, std::forward<MaybeComparator>(maybeComparator)...);
    Kokkos::deep_copy(exec, view, view_h);
  }
}

// --------------------------------------------------
//
// specialize cases for sorting without comparator
//
// --------------------------------------------------

#if defined(KOKKOS_ENABLE_CUDA)
template <class DataType, class... Properties>
void sort_device_view_without_comparator(
    const Cuda& exec, const Kokkos::View<DataType, Properties...>& view) {
  sort_cudathrust(exec, view);
}
#endif

#if defined(KOKKOS_ENABLE_ROCTHRUST)
template <class DataType, class... Properties>
void sort_device_view_without_comparator(
    const HIP& exec, const Kokkos::View<DataType, Properties...>& view) {
  sort_rocthrust(exec, view);
}
#endif

#if defined(KOKKOS_ENABLE_ONEDPL)
template <class DataType, class... Properties>
void sort_device_view_without_comparator(
    const Kokkos::SYCL& exec,
    const Kokkos::View<DataType, Properties...>& view) {
  using ViewType = Kokkos::View<DataType, Properties...>;
  static_assert(
      (ViewType::rank == 1) &&
          (std::is_same_v<typename ViewType::array_layout, LayoutRight> ||
           std::is_same_v<typename ViewType::array_layout, LayoutLeft> ||
           std::is_same_v<typename ViewType::array_layout, LayoutStride>),
      "sort_device_view_without_comparator: supports rank-1 Views "
      "with LayoutLeft, LayoutRight or LayoutStride");

#if KOKKOS_IMPL_ONEDPL_VERSION_GREATER_EQUAL(2022, 8, 0)
  sort_onedpl(exec, view);
#else
  if (view.stride(0) == 1) {
    sort_onedpl(exec, view);
  } else {
    copy_to_host_run_stdsort_copy_back(exec, view);
  }
#endif
}
#endif

// fallback case
template <class ExecutionSpace, class DataType, class... Properties>
std::enable_if_t<Kokkos::is_execution_space<ExecutionSpace>::value>
sort_device_view_without_comparator(
    const ExecutionSpace& exec,
    const Kokkos::View<DataType, Properties...>& view) {
  sort_via_binsort(exec, view);
}

// --------------------------------------------------
//
// specialize cases for sorting with comparator
//
// --------------------------------------------------

#if defined(KOKKOS_ENABLE_CUDA)
template <class ComparatorType, class DataType, class... Properties>
void sort_device_view_with_comparator(
    const Cuda& exec, const Kokkos::View<DataType, Properties...>& view,
    const ComparatorType& comparator) {
  sort_cudathrust(exec, view, comparator);
}
#endif

#if defined(KOKKOS_ENABLE_ROCTHRUST)
template <class ComparatorType, class DataType, class... Properties>
void sort_device_view_with_comparator(
    const HIP& exec, const Kokkos::View<DataType, Properties...>& view,
    const ComparatorType& comparator) {
  sort_rocthrust(exec, view, comparator);
}
#endif

#if defined(KOKKOS_ENABLE_ONEDPL)
template <class ComparatorType, class DataType, class... Properties>
void sort_device_view_with_comparator(
    const Kokkos::SYCL& exec, const Kokkos::View<DataType, Properties...>& view,
    const ComparatorType& comparator) {
  using ViewType = Kokkos::View<DataType, Properties...>;
  static_assert(
      (ViewType::rank == 1) &&
          (std::is_same_v<typename ViewType::array_layout, LayoutRight> ||
           std::is_same_v<typename ViewType::array_layout, LayoutLeft> ||
           std::is_same_v<typename ViewType::array_layout, LayoutStride>),
      "sort_device_view_with_comparator: supports rank-1 Views "
      "with LayoutLeft, LayoutRight or LayoutStride");

#if KOKKOS_IMPL_ONEDPL_VERSION_GREATER_EQUAL(2022, 8, 0)
  sort_onedpl(exec, view, comparator);
#else
  if (view.stride(0) == 1) {
    sort_onedpl(exec, view, comparator);
  } else {
    copy_to_host_run_stdsort_copy_back(exec, view, comparator);
  }
#endif
}
#endif

template <class ExecutionSpace, class ComparatorType, class DataType,
          class... Properties>
std::enable_if_t<Kokkos::is_execution_space<ExecutionSpace>::value>
sort_device_view_with_comparator(
    const ExecutionSpace& exec,
    const Kokkos::View<DataType, Properties...>& view,
    const ComparatorType& comparator) {
  // This is a fallback case if a more specialized overload does not exist:
  // for now, this fallback copies data to host, runs std::sort
  // and then copies data back. Potentially, this can later be changed
  // with a better solution like our own quicksort on device or similar.

// Note with HIP unified memory this code path is still the right thing to do
// if we end up here when RocThrust is not enabled.
// The create_mirror_view_and_copy will do the right thing (no copy).
#ifndef KOKKOS_IMPL_HIP_UNIFIED_MEMORY
  using ViewType = Kokkos::View<DataType, Properties...>;
  using MemSpace = typename ViewType::memory_space;
  static_assert(!SpaceAccessibility<HostSpace, MemSpace>::accessible,
                "Impl::sort_device_view_with_comparator: should not be called "
                "on a view that is already accessible on the host");
#endif

  copy_to_host_run_stdsort_copy_back(exec, view, comparator);
}

}  // namespace Impl
}  // namespace Kokkos

#undef KOKKOS_IMPL_ONEDPL_VERSION
#undef KOKKOS_IMPL_ONEDPL_VERSION_GREATER_EQUAL
#endif
