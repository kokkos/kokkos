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

#ifndef KOKKOS_STD_ALGORITHMS_FOR_EACH_IMPL_HPP
#define KOKKOS_STD_ALGORITHMS_FOR_EACH_IMPL_HPP

#include <Kokkos_Core.hpp>
#include "Kokkos_Constraints.hpp"
#include "Kokkos_HelperPredicates.hpp"
#include <std_algorithms/Kokkos_Distance.hpp>
#include <string>

#define KOKKOS_DO_PRAGMA(x) _Pragma(#x)
#if defined __NVCC__ && defined __NVCC_DIAG_PRAGMA_SUPPORT__
#define KOKKOS_NVCC_PRAGMA_PUSH KOKKOS_DO_PRAGMA(nv_diagnostic push)
#define KOKKOS_NVCC_PRAGMA_SUPPRESS(WARNING) \
  KOKKOS_DO_PRAGMA(nv_diag_suppress = WARNING)
#define KOKKOS_NVCC_PRAGMA_POP KOKKOS_DO_PRAGMA(nv_diagnostic pop)
#elif defined __NVCC__
#define KOKKOS_NVCC_PRAGMA_PUSH KOKKOS_DO_PRAGMA(diagnostic push)
#define KOKKOS_NVCC_PRAGMA_SUPPRESS(WARNING) \
  KOKKOS_DO_PRAGMA(diag_suppress WARNING)
#define KOKKOS_NVCC_PRAGMA_POP KOKKOS_DO_PRAGMA(diagnostic pop)
#else
#define KOKKOS_NVCC_PRAGMA_PUSH
#define KOKKOS_NVCC_PRAGMA_SUPPRESS(WARNING)
#define KOKKOS_NVCC_PRAGMA_POP
#endif

namespace Kokkos {
namespace Experimental {
namespace Impl {

template <class IteratorType, class UnaryFunctorType>
struct StdForEachFunctor {
  using index_type = typename IteratorType::difference_type;
  IteratorType m_first;
  UnaryFunctorType m_functor;

  KOKKOS_FUNCTION
  void operator()(index_type i) const { m_functor(m_first[i]); }

  KOKKOS_FUNCTION
  StdForEachFunctor(IteratorType _first, UnaryFunctorType _functor)
      : m_first(std::move(_first)), m_functor(std::move(_functor)) {}
};

KOKKOS_NVCC_PRAGMA_PUSH
KOKKOS_NVCC_PRAGMA_SUPPRESS(20011)
KOKKOS_NVCC_PRAGMA_SUPPRESS(20014)

template <class HandleType, class IteratorType, class UnaryFunctorType>
KOKKOS_FUNCTION UnaryFunctorType
for_each_impl([[maybe_unused]] const char* label, const HandleType& handle,
              IteratorType first, IteratorType last, UnaryFunctorType functor) {
  // checks
  Impl::static_assert_random_access_and_accessible(handle, first);
  Impl::expect_valid_range(first, last);

  // run
  const auto num_elements = Kokkos::Experimental::distance(first, last);
  if constexpr (is_execution_space_v<HandleType>) {
    KOKKOS_IF_ON_DEVICE(
        (Kokkos::abort("Can't use an execution space on the device!");))
    ::Kokkos::parallel_for(
        label, RangePolicy<HandleType>(handle, 0, num_elements),
        StdForEachFunctor<IteratorType, UnaryFunctorType>(first, functor));
    handle.fence("Kokkos::for_each: fence after operation");
  } else {
    ::Kokkos::parallel_for(
        TeamThreadRange(handle, 0, num_elements),
        StdForEachFunctor<IteratorType, UnaryFunctorType>(first, functor));
    handle.team_barrier();
  }

  return functor;
}

KOKKOS_NVCC_PRAGMA_POP

template <class ExecutionSpace, class IteratorType, class SizeType,
          class UnaryFunctorType>
IteratorType for_each_n_exespace_impl(const std::string& label,
                                      const ExecutionSpace& ex,
                                      IteratorType first, SizeType n,
                                      UnaryFunctorType functor) {
  auto last = first + n;
  Impl::static_assert_random_access_and_accessible(ex, first, last);
  Impl::expect_valid_range(first, last);

  if (n == 0) {
    return first;
  }

  for_each_impl(label.c_str(), ex, first, last, std::move(functor));
  // no neeed to fence since for_each_exespace_impl fences already

  return last;
}

//
// team impl
//
// template <class TeamHandleType, class IteratorType, class UnaryFunctorType>
// KOKKOS_FUNCTION UnaryFunctorType
// for_each_team_impl(const TeamHandleType& teamHandle, IteratorType first,
//                    IteratorType last, UnaryFunctorType functor) {
//   // checks
//   Impl::static_assert_random_access_and_accessible(teamHandle, first);
//   Impl::expect_valid_range(first, last);
//   // run
//   const auto num_elements = Kokkos::Experimental::distance(first, last);
//   ::Kokkos::parallel_for(
//       TeamThreadRange(teamHandle, 0, num_elements),
//       StdForEachFunctor<IteratorType, UnaryFunctorType>(first, functor));
//   teamHandle.team_barrier();
//   return functor;
// }

template <class TeamHandleType, class IteratorType, class SizeType,
          class UnaryFunctorType>
KOKKOS_FUNCTION IteratorType
for_each_n_team_impl(const TeamHandleType& teamHandle, IteratorType first,
                     SizeType n, UnaryFunctorType functor) {
  auto last = first + n;
  Impl::static_assert_random_access_and_accessible(teamHandle, first, last);
  Impl::expect_valid_range(first, last);

  if (n == 0) {
    return first;
  }

  for_each_impl("", teamHandle, first, last, std::move(functor));
  // no neeed to fence since for_each_team_impl fences already

  return last;
}

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#undef KOKKOS_DO_PRAGMA
#undef KOKKOS_NVCC_PRAGMA_PUSH
#undef KOKKOS_NVCC_PRAGMA_SUPPRESS
#undef KOKKOS_NVCC_PRAGMA_POP

#endif
