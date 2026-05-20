// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_PARALLEL_RANGEPOLICYHANDLES_DISPATCH_HPP
#define KOKKOS_PARALLEL_RANGEPOLICYHANDLES_DISPATCH_HPP

#include <Kokkos_Macros.hpp>

namespace Kokkos {
namespace Impl {

template <class iType, class Member, class Closure>
struct ParallelForTeamVectorRangePolicyDispatch;

template <class iType, class Member, class Closure>
struct ParallelForTeamThreadRangePolicyDispatch;

}  // namespace Impl
}  // namespace Kokkos

// Expand at namespace scope after backend parallel_for(Team*RangeBoundariesStruct,
// ...) overloads are declared (see Kokkos_Declare_* include order).
#define KOKKOS_IMPL_PARALLEL_FOR_RANGE_POLICY_TEAM_DISPATCH(MemberType)        \
  namespace Kokkos {                                                           \
  namespace Impl {                                                             \
  template <class iType, class Closure>                                        \
  struct ParallelForTeamVectorRangePolicyDispatch<iType, MemberType, Closure> { \
    static KOKKOS_INLINE_FUNCTION void apply(                                  \
        TeamVectorRangeBoundariesStruct<iType, MemberType> const& bounds,      \
        Closure const& closure) {                                              \
      Kokkos::parallel_for(bounds, closure);                                   \
    }                                                                          \
  };                                                                           \
  template <class iType, class Closure>                                        \
  struct ParallelForTeamThreadRangePolicyDispatch<iType, MemberType, Closure> { \
    static KOKKOS_INLINE_FUNCTION void apply(                                  \
        TeamThreadRangeBoundariesStruct<iType, MemberType> const& bounds,      \
        Closure const& closure) {                                              \
      Kokkos::parallel_for(bounds, closure);                                   \
    }                                                                          \
  };                                                                           \
  } /* namespace Impl */                                                       \
  } /* namespace Kokkos */

#endif /* #define KOKKOS_PARALLEL_RANGEPOLICYHANDLES_DISPATCH_HPP */
