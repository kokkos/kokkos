/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_IMPL_OPENMP_TASK_HPP
#define KOKKOS_IMPL_OPENMP_TASK_HPP

#include <Kokkos_Macros.hpp>
#if defined( KOKKOS_ENABLE_OPENMP ) && defined( KOKKOS_ENABLE_TASKDAG )

#include <Kokkos_TaskScheduler_fwd.hpp>
#include <impl/Kokkos_HostThreadTeam.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

// TODO move this somewhere more general
template <class TeamMember, class Scheduler>
class TaskTeamMemberAdapter : public TeamMember {
private:

  Scheduler m_scheduler;

public:

  //----------------------------------------

  // Forward everything but the Scheduler to the constructor of the TeamMember
  // type that we're adapting
  template <typename... Args>
  KOKKOS_INLINE_FUNCTION
  explicit TaskTeamMemberAdapter(
    typename std::enable_if<
      std::is_constructible<TeamMember, Args...>::value,
      Scheduler
    >::type arg_scheduler,
    Args&&... args
  ) // TODO noexcept specification
    : TeamMember(std::forward<Args>(args)...),
      m_scheduler(std::move(arg_scheduler))
  { }

  TaskTeamMemberAdapter() = default;
  TaskTeamMemberAdapter(TaskTeamMemberAdapter const&) = default;
  TaskTeamMemberAdapter(TaskTeamMemberAdapter&&) = default;
  TaskTeamMemberAdapter& operator=(TaskTeamMemberAdapter const&) = default;
  TaskTeamMemberAdapter& operator=(TaskTeamMemberAdapter&&) = default;
  ~TaskTeamMemberAdapter() = default;

  //----------------------------------------

  Scheduler const& scheduler() const noexcept { return m_scheduler; }

  //----------------------------------------

};

template<>
class TaskQueueSpecialization< Kokkos::OpenMP >
{
public:

  using execution_space = Kokkos::OpenMP ;
  using queue_type      = Kokkos::Impl::TaskQueue< execution_space > ;
  using task_base_type  = Kokkos::Impl::TaskBase< void , void , void > ;
  using scheduler_type = Kokkos::BasicTaskScheduler<execution_space, queue_type>;
  using member_type = TaskTeamMemberAdapter<
    Kokkos::Impl::HostThreadTeamMember<execution_space>,
    scheduler_type
  >;

  enum : int { max_league_size = HostThreadTeamData::max_pool_members };

  // Must specify memory space
  using memory_space = Kokkos::HostSpace ;

  static
  void iff_single_thread_recursive_execute( queue_type * const );

  // Must provide task queue execution function
  static void execute(queue_type*, scheduler_type);

  template< typename TaskType >
  static
  typename TaskType::function_type
  get_function_pointer() { return TaskType::apply ; }
};

extern template class TaskQueue< Kokkos::OpenMP > ;

}} /* namespace Kokkos::Impl */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #if defined( KOKKOS_ENABLE_TASKDAG ) */
#endif /* #ifndef KOKKOS_IMPL_OPENMP_TASK_HPP */

