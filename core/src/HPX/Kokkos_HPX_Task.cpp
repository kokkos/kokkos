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

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_HPX) && defined(KOKKOS_ENABLE_TASKDAG)

#include <Kokkos_Core.hpp>

#include <HPX/Kokkos_HPX_Task.hpp>
#include <impl/Kokkos_TaskQueue_impl.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template class TaskQueue<Kokkos::HPX>;

void TaskQueueSpecialization<Kokkos::HPX>::execute(
    TaskQueue<Kokkos::HPX> *const queue) {
  hpx::run_hpx_function([queue]() {
    static task_base_type *const end = (task_base_type *)task_base_type::EndTag;
    auto num_worker_threads = HPX::impl_max_hardware_threads();

    auto single_policy = TeamPolicyInternal<Kokkos::HPX>(num_worker_threads, 1);
    member_type single_exec(
        single_policy, 0 /* team_rank */, 0 /* league_rank */,
        nullptr /* scratch_buffer.get() */, 0 /* m_shared */);

    hpx::parallel::for_loop(
        hpx::parallel::execution::par.with(
            hpx::parallel::execution::static_chunk_size(1)),
        0, num_worker_threads,
        [&single_exec, num_worker_threads, queue](const std::size_t t) {
          auto team_policy =
              TeamPolicyInternal<Kokkos::HPX>(num_worker_threads, 1);
          member_type team_exec(team_policy, 0, t, nullptr, 0);

          task_base_type *task = nullptr;

          do {
            if (0 == team_exec.team_rank()) {

              bool leader_loop = false;

              do {

                if (0 != task && end != task) {
                  queue->complete(task);
                }

                task = 0 < *((volatile int *)&queue->m_ready_count) ? end
                                                                    : nullptr;

                for (int i = 0; i < queue_type::NumQueue && end == task; ++i) {
                  for (int j = 0; j < 2 && end == task; ++j) {
                    task = queue_type::pop_ready_task(&queue->m_ready[i][j]);
                  }
                }

                leader_loop = end == task;

                if ((!leader_loop) && (0 != task) &&
                    (task_base_type::TaskSingle == task->m_task_type)) {

                  (*task->m_apply)(task, &single_exec);

                  leader_loop = true;
                }
              } while (leader_loop);
            }

            team_exec.team_broadcast(task, 0);

            if (nullptr != task) {
              (*task->m_apply)(task, &team_exec);
            }
          } while (0 != task);
        });
  });
}

void TaskQueueSpecialization<Kokkos::HPX>::iff_single_thread_recursive_execute(
    TaskQueue<Kokkos::HPX> *const queue) {}

} // namespace Impl
} // namespace Kokkos

#else
void KOKKOS_CORE_SRC_IMPL_HPX_TASK_PREVENT_LINK_ERROR() {}
#endif // #if defined( KOKKOS_ENABLE_HPX ) && defined( KOKKOS_ENABLE_TASKDAG )
