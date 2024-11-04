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

#ifndef KOKKOS_THREADS_PARALLEL_FOR_TEAM_HPP
#define KOKKOS_THREADS_PARALLEL_FOR_TEAM_HPP

#include <Kokkos_Parallel.hpp>

namespace Kokkos {
namespace Impl {

template <class FunctorType, class... Properties>
class ParallelFor<FunctorType, Kokkos::TeamPolicy<Properties...>,
                  Kokkos::Threads> {
 private:
  using Policy =
      Kokkos::Impl::TeamPolicyInternal<Kokkos::Threads, Properties...>;
  using WorkTag = typename Policy::work_tag;
  using Member  = typename Policy::member_type;

  const FunctorType m_functor;
  const Policy m_policy;
  const size_t m_shared;

  template <class TagType, class Schedule>
  inline static std::enable_if_t<std::is_void_v<TagType> &&
                                 std::is_same_v<Schedule, Kokkos::Static>>
  exec_team(const FunctorType &functor, Member member) {
    for (; member.valid_static(); member.next_static()) {
      functor(member);
    }
  }

  template <class TagType, class Schedule>
  inline static std::enable_if_t<!std::is_void_v<TagType> &&
                                 std::is_same_v<Schedule, Kokkos::Static>>
  exec_team(const FunctorType &functor, Member member) {
    const TagType t{};
    for (; member.valid_static(); member.next_static()) {
      functor(t, member);
    }
  }

  template <class TagType, class Schedule>
  inline static std::enable_if_t<std::is_void_v<TagType> &&
                                 std::is_same_v<Schedule, Kokkos::Dynamic>>
  exec_team(const FunctorType &functor, Member member) {
    for (; member.valid_dynamic(); member.next_dynamic()) {
      functor(member);
    }
  }

  template <class TagType, class Schedule>
  inline static std::enable_if_t<!std::is_void_v<TagType> &&
                                 std::is_same_v<Schedule, Kokkos::Dynamic>>
  exec_team(const FunctorType &functor, Member member) {
    const TagType t{};
    for (; member.valid_dynamic(); member.next_dynamic()) {
      functor(t, member);
    }
  }

  static void exec(ThreadsInternal &instance, const void *arg) {
    const ParallelFor &self = *((const ParallelFor *)arg);

    ParallelFor::exec_team<WorkTag, typename Policy::schedule_type::type>(
        self.m_functor, Member(&instance, self.m_policy, self.m_shared));

    instance.barrier();
    instance.fan_in();
  }
  template <typename Policy>
  Policy fix_policy(Policy policy) {
    if (policy.impl_vector_length() < 0) {
      policy.impl_set_vector_length(1);
    }
    if (policy.team_size() < 0) {
      int team_size = policy.team_size_recommended(m_functor, ParallelForTag{});
      if (team_size <= 0)
        Kokkos::Impl::throw_runtime_exception(
            "Kokkos::Impl::ParallelFor<Threads, TeamPolicy> could not find a "
            "valid execution configuration.");
      policy.impl_set_team_size(team_size);
    }
    return policy;
  }

 public:
  inline void execute() const {
    ThreadsInternal::resize_scratch(
        0, Policy::member_type::team_reduce_size() + m_shared);

    ThreadsInternal::start(&ParallelFor::exec, this);

    ThreadsInternal::fence();
  }

  ParallelFor(const FunctorType &arg_functor, const Policy &arg_policy)
      : m_functor(arg_functor),
        m_policy(fix_policy(arg_policy)),
        m_shared(m_policy.scratch_size(0) + m_policy.scratch_size(1) +
                 FunctorTeamShmemSize<FunctorType>::value(
                     arg_functor, m_policy.team_size())) {
    if ((arg_policy.scratch_size(0, m_policy.team_size()) +
         FunctorTeamShmemSize<FunctorType>::value(m_functor,
                                                  arg_policy.team_size())) >
        static_cast<size_t>(m_policy.scratch_size_max(0))) {
      std::stringstream error;
      error << "Requested too much scratch memory on level 0. Requested: "
            << arg_policy.scratch_size(0, m_policy.team_size()) +
                   FunctorTeamShmemSize<FunctorType>::value(
                       m_functor, arg_policy.team_size())
            << ", Maximum: " << m_policy.scratch_size_max(0);
      Kokkos::abort(error.str().c_str());
    }
    if (arg_policy.scratch_size(1, m_policy.team_size()) >
        static_cast<size_t>(m_policy.scratch_size_max(1))) {
      std::stringstream error;
      error << "Requested too much scratch memory on level 1. Requested: "
            << arg_policy.scratch_size(1, m_policy.team_size())
            << ", Maximum: " << m_policy.scratch_size_max(1);
      Kokkos::abort(error.str().c_str());
    }
  }
};

}  // namespace Impl
}  // namespace Kokkos

#endif
