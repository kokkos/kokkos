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

#ifndef KOKKOS_OPENMPTARGET_PARALLELREDUCE_TEAM_HPP
#define KOKKOS_OPENMPTARGET_PARALLELREDUCE_TEAM_HPP

#include <omp.h>
#include <sstream>
#include <Kokkos_Parallel.hpp>
#include <OpenMPTarget/Kokkos_OpenMPTarget_Exec.hpp>
#include <OpenMPTarget/Kokkos_OpenMPTarget_Parallel_Common.hpp>

namespace Kokkos {
namespace Impl {

template <class FunctorType, class ReducerType, class... Properties>
class ParallelReduce<FunctorType, Kokkos::TeamPolicy<Properties...>,
                     ReducerType, Kokkos::Experimental::OpenMPTarget> {
 private:
  using Policy =
      Kokkos::Impl::TeamPolicyInternal<Kokkos::Experimental::OpenMPTarget,
                                       Properties...>;

  using WorkTag = typename Policy::work_tag;
  using Member  = typename Policy::member_type;
  using ReducerTypeFwd =
      std::conditional_t<std::is_same<InvalidType, ReducerType>::value,
                         FunctorType, ReducerType>;
  using WorkTagFwd =
      std::conditional_t<std::is_same<InvalidType, ReducerType>::value, WorkTag,
                         void>;
  using Analysis = Impl::FunctorAnalysis<Impl::FunctorPatternInterface::REDUCE,
                                         Policy, ReducerTypeFwd>;

  using pointer_type   = typename Analysis::pointer_type;
  using reference_type = typename Analysis::reference_type;
  using value_type     = typename Analysis::value_type;

  bool m_result_ptr_on_device;
  const int m_result_ptr_num_elems;

  static constexpr int HasJoin =
      Impl::FunctorAnalysis<Impl::FunctorPatternInterface::REDUCE, Policy,
                            FunctorType>::has_join_member_function;
  static constexpr int UseReducer = is_reducer<ReducerType>::value;
  static constexpr int IsArray    = std::is_pointer<reference_type>::value;

  using ParReduceSpecialize =
      ParallelReduceSpecialize<FunctorType, Policy, ReducerType, pointer_type,
                               typename Analysis::value_type>;

  const FunctorType m_functor;
  const Policy m_policy;
  const ReducerType m_reducer;
  const pointer_type m_result_ptr;
  const size_t m_shmem_size;

 public:
  void execute() const {
    if constexpr (HasJoin) {
      ParReduceSpecialize::execute_init_join(m_functor, m_policy, m_result_ptr,
                                             m_result_ptr_on_device);
    } else if constexpr (UseReducer) {
      ParReduceSpecialize::execute_reducer(m_functor, m_policy, m_result_ptr,
                                           m_result_ptr_on_device);
    } else if constexpr (IsArray) {
      if (m_result_ptr_num_elems <= 2) {
        ParReduceSpecialize::template execute_array<2>(
            m_functor, m_policy, m_result_ptr, m_result_ptr_on_device);
      } else if (m_result_ptr_num_elems <= 4) {
        ParReduceSpecialize::template execute_array<4>(
            m_functor, m_policy, m_result_ptr, m_result_ptr_on_device);
      } else if (m_result_ptr_num_elems <= 8) {
        ParReduceSpecialize::template execute_array<8>(
            m_functor, m_policy, m_result_ptr, m_result_ptr_on_device);
      } else if (m_result_ptr_num_elems <= 16) {
        ParReduceSpecialize::template execute_array<16>(
            m_functor, m_policy, m_result_ptr, m_result_ptr_on_device);
      } else if (m_result_ptr_num_elems <= 32) {
        ParReduceSpecialize::template execute_array<32>(
            m_functor, m_policy, m_result_ptr, m_result_ptr_on_device);
      } else {
        Kokkos::abort("array reduction length must be <= 32");
      }
    } else {
      ParReduceSpecialize::template execute_array<1>(
          m_functor, m_policy, m_result_ptr, m_result_ptr_on_device);
    }
  }

  template <class ViewType>
  ParallelReduce(const FunctorType& arg_functor, const Policy& arg_policy,
                 const ViewType& arg_result,
                 std::enable_if_t<Kokkos::is_view<ViewType>::value &&
                                      !Kokkos::is_reducer<ReducerType>::value,
                                  void*> = nullptr)
      : m_result_ptr_on_device(
            MemorySpaceAccess<Kokkos::Experimental::OpenMPTargetSpace,
                              typename ViewType::memory_space>::accessible),
        m_result_ptr_num_elems(arg_result.size()),
        m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(InvalidType()),
        m_result_ptr(arg_result.data()),
        m_shmem_size(arg_policy.scratch_size(0) + arg_policy.scratch_size(1) +
                     FunctorTeamShmemSize<FunctorType>::value(
                         arg_functor, arg_policy.team_size())) {}

  ParallelReduce(const FunctorType& arg_functor, Policy& arg_policy,
                 const ReducerType& reducer)
      : m_result_ptr_on_device(
            MemorySpaceAccess<Kokkos::Experimental::OpenMPTargetSpace,
                              typename ReducerType::result_view_type::
                                  memory_space>::accessible),
        m_result_ptr_num_elems(reducer.view().size()),
        m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(reducer),
        m_result_ptr(reducer.view().data()),
        m_shmem_size(arg_policy.scratch_size(0) + arg_policy.scratch_size(1) +
                     FunctorTeamShmemSize<FunctorType>::value(
                         arg_functor, arg_policy.team_size())) {}
};

}  // namespace Impl
}  // namespace Kokkos

#endif
