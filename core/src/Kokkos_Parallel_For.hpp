// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

/// \file Kokkos_Parallel_For.hpp
/// \brief Declaration of parallel_for interface

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#include <Kokkos_Macros.hpp>
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif
#ifndef KOKKOS_PARALLEL_FOR_HPP
#define KOKKOS_PARALLEL_FOR_HPP

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_ExecPolicy.hpp>
#include <Kokkos_View.hpp>

#include <impl/Kokkos_CheckUsage.hpp>
#include <impl/Kokkos_FunctorAnalysis.hpp>
#include <impl/Kokkos_Tools_Generic.hpp>

#include <type_traits>
#include <string>

namespace Kokkos {

/** \brief Execute \c functor in parallel according to the execution \c policy.
 *
 * A "functor" is a class containing the function to execute in parallel,
 * data needed for that execution, and an optional \c execution_space
 * alias.  Here is an example functor for parallel_for:
 *
 * \code
 *  class FunctorType {
 *  public:
 *    using execution_space = ...;
 *    void operator() ( WorkType iwork ) const ;
 *  };
 * \endcode
 *
 * In the above example, \c WorkType is any integer type for which a
 * valid conversion from \c size_t to \c IntType exists.  Its
 * <tt>operator()</tt> method defines the operation to parallelize,
 * over the range of integer indices <tt>iwork=[0,work_count-1]</tt>.
 * This compares to a single iteration \c iwork of a \c for loop.
 * If \c execution_space is not defined DefaultExecutionSpace will be used.
 */
template <class Label, Kokkos::ExecutionPolicy ExecPolicy, class FunctorType>
  requires(std::is_constructible_v<std::string, const Label&>)
inline void parallel_for([[maybe_unused]] const Label& label,
                         const ExecPolicy& policy, const FunctorType& functor) {
  // Work around unsuppressable warning of calling host (constexpr) function
  // from host device function.
  // This occurs when trying to instantiate this parallel for from inside
  // a host-device function which is called on the host.
  // The problem is the ctor from c-string for std::string
  std::string str;
  KOKKOS_IF_ON_HOST(str = std::string(label);)

  /** Enforce correct use **/
  Impl::CheckUsage<Impl::UsageRequires::insideExecEnv>::check(
      "parallel_for", policy, str.c_str());

  uint64_t kpID = 0;
  /** Request a tuned policy from the tools subsystem */
  const auto& response =
      Kokkos::Tools::Impl::begin_parallel_for(policy, functor, str, kpID);
  const auto& inner_policy = response.policy;

  auto closure =
      Kokkos::Impl::construct_with_shared_allocation_tracking_disabled<
          Impl::ParallelFor<FunctorType, ExecPolicy>>(functor, inner_policy);

  closure.execute();
  Kokkos::Tools::Impl::end_parallel_for(inner_policy, functor, str, kpID);
}

template <Kokkos::ExecutionPolicy ExecPolicy, class FunctorType>
KOKKOS_INLINE_FUNCTION void parallel_for(const ExecPolicy& policy,
                                         const FunctorType& functor) {
  KOKKOS_IF_ON_DEVICE(
      Kokkos::abort("Kokkos::parallel_for(ExecutionPolicy, functor) cannot be "
                    "called from device.\n");)

  // Work around nvcc complaint about calling __host__ function from __host__
  // __device__ function. Assert above that we are not actually calling on
  // device.
  KOKKOS_IMPL_DISABLE_CALLING_HOST_FROM_DEVICE_WARNINGS_PUSH()
  /** Enforce correct use **/
  Impl::CheckUsage<Impl::UsageRequires::insideExecEnv>::check("parallel_for",
                                                              policy);
  Kokkos::parallel_for("", policy, functor);
  KOKKOS_IMPL_DISABLE_CALLING_HOST_FROM_DEVICE_WARNINGS_POP()
}

template <class FunctorType>
inline void parallel_for(const std::string& str, const size_t work_count,
                         const FunctorType& functor) {
  /** Enforce correct use **/
  Impl::CheckUsage<Impl::UsageRequires::insideExecEnv>::check(
      "parallel_for", work_count, str.c_str());

  using execution_space =
      typename Impl::FunctorPolicyExecutionSpace<FunctorType,
                                                 void>::execution_space;
  using policy = RangePolicy<execution_space>;

  policy execution_policy = policy(0, work_count);
  ::Kokkos::parallel_for(str, execution_policy, functor);
}

template <class FunctorType>
inline void parallel_for(const size_t work_count, const FunctorType& functor) {
  /** Enforce correct use **/
  Impl::CheckUsage<Impl::UsageRequires::insideExecEnv>::check("parallel_for",
                                                              work_count);

  ::Kokkos::parallel_for("", work_count, functor);
}
}  // namespace Kokkos
#endif
