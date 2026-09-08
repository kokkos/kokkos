// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

/// \file Kokkos_Parallel_Scan.hpp
/// \brief Declaration of parallel_scan interface

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#include <Kokkos_Macros.hpp>
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif
#ifndef KOKKOS_PARALLEL_SCAN_HPP
#define KOKKOS_PARALLEL_SCAN_HPP

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_ExecPolicy.hpp>
#include <Kokkos_View.hpp>

#include <impl/Kokkos_CheckUsage.hpp>
#include <impl/Kokkos_FunctorAnalysis.hpp>
#include <impl/Kokkos_Tools_Generic.hpp>

#include <type_traits>
#include <string>

namespace Kokkos {
/// \fn parallel_scan
/// \tparam ExecutionPolicy The execution policy type.
/// \tparam FunctorType     The scan functor type.
///
/// \param policy  [in] The execution policy.
/// \param functor [in] The scan functor.
///
/// This function implements a parallel scan pattern.  The scan can
/// be either inclusive or exclusive, depending on how you implement
/// the scan functor.
///
/// A scan functor looks almost exactly like a reduce functor, except
/// that its operator() takes a third \c bool argument, \c final_pass,
/// which indicates whether this is the last pass of the scan
/// operation.  We will show below how to use the \c final_pass
/// argument to control whether the scan is inclusive or exclusive.
///
/// Here is the minimum required interface of a scan functor for a POD
/// (plain old data) value type \c PodType.  That is, the result is a
/// View of zero or more PodType.  It is also possible for the result
/// to be an array of (same-sized) arrays of PodType, but we do not
/// show the required interface for that here.
/// \code
/// template< class ExecPolicy , class FunctorType >
/// class ScanFunctor {
/// public:
///   // The Kokkos device type
///   using execution_space = ...;
///   // Type of an entry of the array containing the result;
///   // also the type of each of the entries combined using
///   // operator() or join().
///   using value_type = PodType;
///
///   void operator () (const ExecPolicy::member_type & i,
///                     value_type& update,
///                     const bool final_pass) const;
///   void init (value_type& update) const;
///   void join (value_type& update,
//               const value_type& input) const
/// };
/// \endcode
///
/// Here is an example of a functor which computes an inclusive plus-scan
/// of an array of \c int, in place.  If given an array [1, 2, 3, 4], this
/// scan will overwrite that array with [1, 3, 6, 10].
///
/// \code
/// template<class SpaceType>
/// class InclScanFunctor {
/// public:
///   using execution_space = SpaceType;
///   using value_type = int;
///   using size_type = typename SpaceType::size_type;
///
///   InclScanFunctor( Kokkos::View<value_type*, execution_space> x
///                  , Kokkos::View<value_type*, execution_space> y ) : m_x(x),
///                  m_y(y) {}
///
///   void operator () (const size_type i, value_type& update, const bool
///   final_pass) const {
///     update += m_x(i);
///     if (final_pass) {
///       m_y(i) = update;
///     }
///   }
///   void init (value_type& update) const {
///     update = 0;
///   }
///   void join (value_type& update, const value_type& input)
///   const {
///     update += input;
///   }
///
/// private:
///   Kokkos::View<value_type*, execution_space> m_x;
///   Kokkos::View<value_type*, execution_space> m_y;
/// };
/// \endcode
///
/// Here is an example of a functor which computes an <i>exclusive</i>
/// scan of an array of \c int, in place.  In operator(), note both
/// that the final_pass test and the update have switched places, and
/// the use of a temporary.  If given an array [1, 2, 3, 4], this scan
/// will overwrite that array with [0, 1, 3, 6].
///
/// \code
/// template<class SpaceType>
/// class ExclScanFunctor {
/// public:
///   using execution_space = SpaceType;
///   using value_type = int;
///   using size_type = typename SpaceType::size_type;
///
///   ExclScanFunctor (Kokkos::View<value_type*, execution_space> x) : x_ (x) {}
///
///   void operator () (const size_type i, value_type& update, const bool
///   final_pass) const {
///     const value_type x_i = x_(i);
///     if (final_pass) {
///       x_(i) = update;
///     }
///     update += x_i;
///   }
///   void init (value_type& update) const {
///     update = 0;
///   }
///   void join (value_type& update, const value_type& input)
///   const {
///     update += input;
///   }
///
/// private:
///   Kokkos::View<value_type*, execution_space> x_;
/// };
/// \endcode
///
/// Here is an example of a functor which builds on the above
/// exclusive scan example, to compute an offsets array from a
/// population count array, in place.  We assume that the pop count
/// array has an extra entry at the end to store the final count.  If
/// given an array [1, 2, 3, 4, 0], this scan will overwrite that
/// array with [0, 1, 3, 6, 10].
///
/// \code
/// template<class SpaceType>
/// class OffsetScanFunctor {
/// public:
///   using execution_space = SpaceType;
///   using value_type = int;
///   using size_type = typename SpaceType::size_type;
///
///   // lastIndex_ is the last valid index (zero-based) of x.
///   // If x has length zero, then lastIndex_ won't be used anyway.
///   OffsetScanFunctor( Kokkos::View<value_type*, execution_space> x
///                    , Kokkos::View<value_type*, execution_space> y )
///      : m_x(x), m_y(y), last_index_ (x.dimension_0 () == 0 ? 0 :
///      x.dimension_0 () - 1)
///   {}
///
///   void operator () (const size_type i, int& update, const bool final_pass)
///   const {
///     if (final_pass) {
///       m_y(i) = update;
///     }
///     update += m_x(i);
///     // The last entry of m_y gets the final sum.
///     if (final_pass && i == last_index_) {
///       m_y(i+1) = update;
// i/     }
///   }
///   void init (value_type& update) const {
///     update = 0;
///   }
///   void join (value_type& update, const value_type& input)
///   const {
///     update += input;
///   }
///
/// private:
///   Kokkos::View<value_type*, execution_space> m_x;
///   Kokkos::View<value_type*, execution_space> m_y;
///   const size_type last_index_;
/// };
/// \endcode
///
template <Kokkos::ExecutionPolicy ExecutionPolicy, class FunctorType>
inline void parallel_scan(const std::string& str, const ExecutionPolicy& policy,
                          const FunctorType& functor) {
  /** Enforce correct use **/
  Impl::CheckUsage<Impl::UsageRequires::insideExecEnv>::check(
      "parallel_scan", policy, str.c_str());

  uint64_t kpID = 0;
  /** Request a tuned policy from the tools subsystem */
  const auto& response =
      Kokkos::Tools::Impl::begin_parallel_scan(policy, functor, str, kpID);
  const auto& inner_policy = response.policy;

  auto closure =
      Kokkos::Impl::construct_with_shared_allocation_tracking_disabled<
          Impl::ParallelScan<FunctorType, ExecutionPolicy>>(functor,
                                                            inner_policy);

  closure.execute();

  Kokkos::Tools::Impl::end_parallel_scan(inner_policy, functor, str, kpID);
}

template <Kokkos::ExecutionPolicy ExecutionPolicy, class FunctorType>
inline void parallel_scan(const ExecutionPolicy& policy,
                          const FunctorType& functor) {
  /** Enforce correct use **/
  Impl::CheckUsage<Impl::UsageRequires::insideExecEnv>::check("parallel_scan",
                                                              policy);

  ::Kokkos::parallel_scan("", policy, functor);
}

template <class FunctorType>
inline void parallel_scan(const std::string& str, const size_t work_count,
                          const FunctorType& functor) {
  /** Enforce correct use **/
  Impl::CheckUsage<Impl::UsageRequires::insideExecEnv>::check(
      "parallel_scan", work_count, str.c_str());

  using execution_space =
      typename Kokkos::Impl::FunctorPolicyExecutionSpace<FunctorType,
                                                         void>::execution_space;

  using policy = Kokkos::RangePolicy<execution_space>;

  policy execution_policy(0, work_count);
  parallel_scan(str, execution_policy, functor);
}

template <class FunctorType>
inline void parallel_scan(const size_t work_count, const FunctorType& functor) {
  /** Enforce correct use **/
  Impl::CheckUsage<Impl::UsageRequires::insideExecEnv>::check("parallel_scan",
                                                              work_count);

  ::Kokkos::parallel_scan("", work_count, functor);
}

template <Kokkos::ExecutionPolicy ExecutionPolicy, class FunctorType,
          class ReturnType>
inline void parallel_scan(const std::string& str, const ExecutionPolicy& policy,
                          const FunctorType& functor,
                          ReturnType& return_value) {
  /** Enforce correct use **/
  Impl::CheckUsage<Impl::UsageRequires::insideExecEnv>::check(
      "parallel_scan", policy, str.c_str());

  uint64_t kpID                = 0;
  ExecutionPolicy inner_policy = policy;
  Kokkos::Tools::Impl::begin_parallel_scan(inner_policy, functor, str, kpID);

  if constexpr (Kokkos::is_view<ReturnType>::value) {
    auto closure =
        Kokkos::Impl::construct_with_shared_allocation_tracking_disabled<
            Impl::ParallelScanWithTotal<FunctorType, ExecutionPolicy,
                                        typename ReturnType::value_type>>(
            functor, inner_policy, return_value);
    closure.execute();
  } else {
    Kokkos::View<ReturnType, Kokkos::HostSpace> view(&return_value);
    auto closure =
        Kokkos::Impl::construct_with_shared_allocation_tracking_disabled<
            Impl::ParallelScanWithTotal<FunctorType, ExecutionPolicy,
                                        ReturnType>>(functor, inner_policy,
                                                     view);
    closure.execute();
  }

  Kokkos::Tools::Impl::end_parallel_scan(inner_policy, functor, str, kpID);

  if (!Kokkos::is_view<ReturnType>::value)
    policy.space().fence(
        "Kokkos::parallel_scan: fence due to result being a value, not a view");
}

template <Kokkos::ExecutionPolicy ExecutionPolicy, class FunctorType,
          class ReturnType>
inline void parallel_scan(const ExecutionPolicy& policy,
                          const FunctorType& functor,
                          ReturnType& return_value) {
  /** Enforce correct use **/
  Impl::CheckUsage<Impl::UsageRequires::insideExecEnv>::check("parallel_scan",
                                                              policy);

  ::Kokkos::parallel_scan("", policy, functor, return_value);
}

template <class FunctorType, class ReturnType>
inline void parallel_scan(const std::string& str, const size_t work_count,
                          const FunctorType& functor,
                          ReturnType& return_value) {
  /** Enforce correct use **/
  Impl::CheckUsage<Impl::UsageRequires::insideExecEnv>::check(
      "parallel_scan", work_count, str.c_str());

  using execution_space =
      typename Kokkos::Impl::FunctorPolicyExecutionSpace<FunctorType,
                                                         void>::execution_space;

  using policy = Kokkos::RangePolicy<execution_space>;

  policy execution_policy(0, work_count);
  parallel_scan(str, execution_policy, functor, return_value);
}

template <class FunctorType, class ReturnType>
inline void parallel_scan(const size_t work_count, const FunctorType& functor,
                          ReturnType& return_value) {
  /** Enforce correct use **/
  Impl::CheckUsage<Impl::UsageRequires::insideExecEnv>::check("parallel_scan",
                                                              work_count);

  ::Kokkos::parallel_scan("", work_count, functor, return_value);
}
}  // namespace Kokkos
#endif
