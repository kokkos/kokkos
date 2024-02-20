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

#ifndef KOKKOS_SORT_BY_KEY_FREE_FUNCS_IMPL_HPP_
#define KOKKOS_SORT_BY_KEY_FREE_FUNCS_IMPL_HPP_

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

#if defined(KOKKOS_ENABLE_ONEDPL) && \
    (ONEDPL_VERSION_MAJOR > 2022 ||  \
     (ONEDPL_VERSION_MAJOR == 2022 && ONEDPL_VERSION_MINOR >= 2))
#define KOKKOS_ONEDPL_HAS_SORT_BY_KEY
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#endif

namespace Kokkos::Impl {

template <typename T>
constexpr inline bool is_admissible_to_kokkos_sort_by_key =
    ::Kokkos::is_view<T>::value&& T::rank() == 1 &&
    (std::is_same<typename T::traits::array_layout,
                  Kokkos::LayoutLeft>::value ||
     std::is_same<typename T::traits::array_layout,
                  Kokkos::LayoutRight>::value ||
     std::is_same<typename T::traits::array_layout,
                  Kokkos::LayoutStride>::value);

template <class ViewType>
KOKKOS_INLINE_FUNCTION constexpr void
static_assert_is_admissible_to_kokkos_sort_by_key(const ViewType& /* view */) {
  static_assert(is_admissible_to_kokkos_sort_by_key<ViewType>,
                "Kokkos::sort_by_key only accepts 1D values View with "
                "LayoutRight, LayoutLeft or LayoutStride.");
}

#if defined(KOKKOS_ENABLE_CUDA)
template <class KeysDataType, class... KeysProperties, class ValuesDataType,
          class... ValuesProperties, class... MaybeComparator>
void sort_by_key_cudathrust(
    const Kokkos::Cuda& exec,
    const Kokkos::View<KeysDataType, KeysProperties...>& keys,
    const Kokkos::View<ValuesDataType, ValuesProperties...>& values,
    MaybeComparator&&... maybeComparator) {
  const auto policy = thrust::cuda::par.on(exec.cuda_stream());
  auto keys_first   = ::Kokkos::Experimental::begin(keys);
  auto keys_last    = ::Kokkos::Experimental::end(keys);
  auto values_first = ::Kokkos::Experimental::begin(values);
  thrust::sort_by_key(policy, keys_first, keys_last, values_first,
                      std::forward<MaybeComparator>(maybeComparator)...);
}
#endif

#ifdef KOKKOS_ONEDPL_HAS_SORT_BY_KEY
template <class KeysDataType, class... KeysProperties, class ValuesDataType,
          class... ValuesProperties, class... MaybeComparator>
void sort_by_key_onedpl(
    const Kokkos::SYCL& exec,
    const Kokkos::View<KeysDataType, KeysProperties...>& keys,
    const Kokkos::View<ValuesDataType, ValuesProperties...>& values,
    MaybeComparator&&... maybeComparator) {
  if (keys.stride(0) != 1 && values.stride(0) != 1) {
    Kokkos::abort(
        "SYCL sort_by_key only supports rank-1 Views with stride(0) = 1.");
  }

  // Can't use Experimental::begin/end here since the oneDPL then assumes that
  // the data is on the host.
  auto queue  = exec.sycl_queue();
  auto policy = oneapi::dpl::execution::make_device_policy(queue);
  const int n = keys.extent(0);
  oneapi::dpl::sort_by_key(policy, keys.data(), keys.data() + n, values.data(),
                           std::forward<MaybeComparator>(maybeComparator)...);
}
#endif

template <typename ExecutionSpace, typename PermutationView, typename ViewType>
void applyPermutation(const ExecutionSpace& space,
                      const PermutationView& permutation,
                      const ViewType& view) {
  static_assert(std::is_integral<typename PermutationView::value_type>::value);

  auto view_copy = Kokkos::create_mirror(
      Kokkos::view_alloc(space, typename ExecutionSpace::memory_space{},
                         Kokkos::WithoutInitializing),
      view);
  Kokkos::deep_copy(space, view_copy, view);
  Kokkos::parallel_for(
      "Kokkos::sort_by_key_via_sort::permute_" + view.label(),
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, view.extent(0)),
      KOKKOS_LAMBDA(int i) { view(i) = view_copy(permutation(i)); });
}

template <class ExecutionSpace, class KeysDataType, class... KeysProperties,
          class ValuesDataType, class... ValuesProperties,
          class... MaybeComparator>
void sort_by_key_via_sort(
    const ExecutionSpace& exec,
    const Kokkos::View<KeysDataType, KeysProperties...>& keys,
    const Kokkos::View<ValuesDataType, ValuesProperties...>& values,
    MaybeComparator&&... maybeComparator) {
  auto const n = keys.size();

  Kokkos::View<unsigned int*, ExecutionSpace> permute(
      Kokkos::view_alloc(exec, Kokkos::WithoutInitializing,
                         "Kokkos::sort_by_key_via_sort::permute"),
      n);

  // iota
  Kokkos::parallel_for(
      "Kokkos::sort_by_key_via_sort::iota",
      Kokkos::RangePolicy<ExecutionSpace>(exec, 0, n),
      KOKKOS_LAMBDA(int i) { permute(i) = i; });

// FIXME OPENMPTARGET The sort happens on the host so we have to copy keys there
#ifdef KOKKOS_ENABLE_OPENMPTARGET
  auto keys_in_comparator = Kokkos::create_mirror_view(
      Kokkos::view_alloc(Kokkos::HostSpace{}, Kokkos::WithoutInitializing),
      keys);
  Kokkos::deep_copy(exec, keys_in_comparator, keys);
#else
  auto keys_in_comparator = keys;
#endif

  static_assert(sizeof...(MaybeComparator) <= 1);
  if constexpr (sizeof...(MaybeComparator) == 0) {
    Kokkos::sort(
        exec, permute, KOKKOS_LAMBDA(int i, int j) {
          return keys_in_comparator(i) < keys_in_comparator(j);
        });
  } else {
    auto keys_comparator =
        std::get<0>(std::tuple<MaybeComparator...>(maybeComparator...));
    Kokkos::sort(
        exec, permute, KOKKOS_LAMBDA(int i, int j) {
          return keys_comparator(keys_in_comparator(i), keys_in_comparator(j));
        });
  }

  applyPermutation(exec, permute, keys);
  applyPermutation(exec, permute, values);
}

// ------------------------------------------------------
//
// specialize cases for sorting by key without comparator
//
// ------------------------------------------------------

#if defined(KOKKOS_ENABLE_CUDA)
template <class KeysDataType, class... KeysProperties, class ValuesDataType,
          class... ValuesProperties>
void sort_by_key_device_view_without_comparator(
    const Kokkos::Cuda& exec,
    const Kokkos::View<KeysDataType, KeysProperties...>& keys,
    const Kokkos::View<ValuesDataType, ValuesProperties...>& values) {
  sort_by_key_cudathrust(exec, keys, values);
}
#endif

#if defined(KOKKOS_ENABLE_ONEDPL)
template <class KeysDataType, class... KeysProperties, class ValuesDataType,
          class... ValuesProperties>
void sort_by_key_device_view_without_comparator(
    const Kokkos::Experimental::SYCL& exec,
    const Kokkos::View<KeysDataType, KeysProperties...>& keys,
    const Kokkos::View<ValuesDataType, ValuesProperties...>& values) {
#ifdef KOKKOS_ONEDPL_HAS_SORT_BY_KEY
  if (keys.stride(0) == 1 && values.stride(0) == 1)
    sort_by_key_onedpl(exec, keys, values);
  else
#else
  sort_by_key_via_sort(exec, keys, values);
#endif
}
#endif

// fallback case
template <class ExecutionSpace, class KeysDataType, class... KeysProperties,
          class ValuesDataType, class... ValuesProperties>
std::enable_if_t<Kokkos::is_execution_space<ExecutionSpace>::value>
sort_by_key_device_view_without_comparator(
    const ExecutionSpace& exec,
    const Kokkos::View<KeysDataType, KeysProperties...>& keys,
    const Kokkos::View<ValuesDataType, ValuesProperties...>& values) {
  sort_by_key_via_sort(exec, keys, values);
}

// ---------------------------------------------------
//
// specialize cases for sorting by key with comparator
//
// ---------------------------------------------------

#if defined(KOKKOS_ENABLE_CUDA)
template <class ComparatorType, class KeysDataType, class... KeysProperties,
          class ValuesDataType, class... ValuesProperties>
void sort_by_key_device_view_with_comparator(
    const Kokkos::Cuda& exec,
    const Kokkos::View<KeysDataType, KeysProperties...>& keys,
    const Kokkos::View<ValuesDataType, ValuesProperties...>& values,
    const ComparatorType& comparator) {
  sort_by_key_cudathrust(exec, keys, values, comparator);
}
#endif

#if defined(KOKKOS_ENABLE_ONEDPL)
template <class ComparatorType, class KeysDataType, class... KeysProperties,
          class ValuesDataType, class... ValuesProperties>
void sort_by_key_device_view_with_comparator(
    const Kokkos::Experimental::SYCL& exec,
    const Kokkos::View<KeysDataType, KeysProperties...>& keys,
    const Kokkos::View<ValuesDataType, ValuesProperties...>& values,
    const ComparatorType& comparator) {
#ifdef KOKKOS_ONEDPL_HAS_SORT_BY_KEY
  if (keys.stride(0) == 1 && values.stride(0) == 1)
    sort_by_key_onedpl(exec, keys, values, comparator);
  else
#else
  sort_by_key_via_sort(exec, keys, values, comparator);
#endif
}
#endif

// fallback case
template <class ComparatorType, class ExecutionSpace, class KeysDataType,
          class... KeysProperties, class ValuesDataType,
          class... ValuesProperties>
std::enable_if_t<Kokkos::is_execution_space<ExecutionSpace>::value>
sort_by_key_device_view_with_comparator(
    const ExecutionSpace& exec,
    const Kokkos::View<KeysDataType, KeysProperties...>& keys,
    const Kokkos::View<ValuesDataType, ValuesProperties...>& values,
    const ComparatorType& comparator) {
  sort_by_key_via_sort(exec, keys, values, comparator);
}

#undef KOKKOS_ONEDPL_HAS_SORT_BY_KEY

}  // namespace Kokkos::Impl
#endif
