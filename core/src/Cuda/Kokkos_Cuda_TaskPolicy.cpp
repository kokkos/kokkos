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
// Questions? Contact  H. Carter Edwards (hcedwar@sandia.gov)
// 
// ************************************************************************
//@HEADER
*/

// Experimental unified task-data parallel manycore LDRD

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <Cuda/Kokkos_Cuda_TaskPolicy.hpp>

#if defined( KOKKOS_HAVE_CUDA )

#define QUEUE_LOCK_VALUE ((Task*)( ~((uintptr_t)0) ))
#define QUEUE_DEAD_VALUE ((Task*)( ~((uintptr_t)0) - 1 ))

namespace Kokkos {
namespace Experimental {
namespace Impl {

typedef TaskMember< Kokkos::Cuda , void , void > Task ;

} /* namespace Impl */
} /* namespace Experimental */
} /* namespace Kokkos */

namespace Kokkos {
namespace Experimental {
namespace Impl {

CudaTaskPolicyQueue::
CudaTaskPolicyQueue
  ( const unsigned arg_memory_pool_chunk
  , const unsigned arg_memory_pool_size
  , const unsigned arg_default_dependence_capacity
  )
  : m_space( Kokkos::CudaUVMSpace()
           , arg_memory_pool_chunk
           , arg_memory_pool_size )
  , m_ready_team(0)
  , m_ready_serial(0)
  , m_garbage(0)
  , m_team_size( 32 * 4 /* 4 warps */ )
  , m_default_dependence_capacity( arg_default_dependence_capacity )
  , m_task_count(0)
{
}

void CudaTaskPolicyQueue::Destroy::destroy_shared_allocation()
{
  Kokkos::Cuda::fence();
}

} /* namespace Impl */
} /* namespace Experimental */
} /* namespace Kokkos */

__global__
void kokkos_cuda_task_policy_queue_driver
  ( Kokkos::Experimental::Impl::CudaTaskPolicyQueue * queue )
{
  queue->driver();
}


namespace Kokkos {
namespace Experimental {

TaskPolicy< Kokkos::Cuda >::TaskPolicy
  ( const unsigned arg_memory_pool_chunk
  , const unsigned arg_memory_pool_size
  , const unsigned arg_default_dependence_capacity
  )
  : m_track()
  , m_cuda_uvm_queue(0)
{
  // Allocate the queue data sructure in UVM space

  typedef Kokkos::Experimental::Impl::SharedAllocationRecord
    < Kokkos::CudaUVMSpace , Impl::CudaTaskPolicyQueue::Destroy > record_type ;

  record_type * record =
    record_type::allocate( Kokkos::CudaUVMSpace()
                         , "CudaUVM task queue"
                         , sizeof(Impl::CudaTaskPolicyQueue)
                         );

  m_cuda_uvm_queue = reinterpret_cast< Impl::CudaTaskPolicyQueue * >( record->data() );

  new( m_cuda_uvm_queue )
    Impl::CudaTaskPolicyQueue( arg_memory_pool_chunk
                             , arg_memory_pool_size
                             , arg_default_dependence_capacity );

  record->m_destroy.m_queue = m_cuda_uvm_queue ;
  
  m_track.assign_allocated_record_to_uninitialized( record );
}

void wait( Kokkos::Experimental::TaskPolicy< Kokkos::Cuda > & policy )
{
  const dim3 grid( Kokkos::Impl::cuda_internal_multiprocessor_count() , 0 , 0 );
  const dim3 block( 32 , 4 , 0 );
  const int shared = Kokkos::Impl::CudaTraits::SharedMemoryUsage / 2 ;

  cudaFuncSetCacheConfig( ::kokkos_cuda_task_policy_queue_driver
                        , cudaFuncCachePreferL1 );

  ::kokkos_cuda_task_policy_queue_driver<<< grid , block , shared >>>
    ( policy.m_cuda_uvm_queue );

  Kokkos::Cuda::fence();
}

} /* namespace Experimental */
} /* namespace Kokkos */

namespace Kokkos {
namespace Experimental {
namespace Impl {

//----------------------------------------------------------------------------

__device__
void Task::reschedule()
{
  // Reschedule transitions from executing back to waiting.
  const int old_state =
    atomic_compare_exchange( & m_state
                           , int(TASK_STATE_EXECUTING)
                           , int(TASK_STATE_WAITING) );

  if ( old_state != int(TASK_STATE_EXECUTING) ) {
    Kokkos::abort("Task::reschedule STATE ERROR" );
  }
}

__device__
void Task::schedule()
{
  //----------------------------------------
  // State is either constructing or already waiting.
  // If constructing then transition to waiting.

  {
    const int old_state =
      atomic_compare_exchange( & m_state
                             , int(TASK_STATE_CONSTRUCTING)
                             , int(TASK_STATE_WAITING) );

    Task * const waitTask = *((Task * volatile const *) & m_wait );
    Task * const next     = *((Task * volatile const *) & m_next );

    // Required state:
    // 1) the wait linked list is open to addition (not denied)
    // 2) the next linked list is zero showing not currently
    //    in another task's wait linked list
    // 3) started in the constructing or waiting state

    if ( QUEUE_DEAD_VALUE == waitTask || 0 != next ||
         ( old_state != int(TASK_STATE_CONSTRUCTING) &&
           old_state != int(TASK_STATE_WAITING) ) ) {
      Kokkos::abort("Task::schedule STATE ERROR");
    }
  }

  // Now guaranteed that this thread has exclusive access.
  //----------------------------------------
  // Insert this task into another dependence that is not complete
  // Push on to the wait queue, fails if ( denied == m_dep[i]->m_wait )

  bool try_insert_in_queue = true ;

  for ( int i = 0 ; i < m_dep_size && try_insert_in_queue ; ) {

    // Query a task that this task is dependent upon.
    // Query that task's wait linked list.

    Task * const task_dep = m_dep[i] ;
    Task * const head_value_old = *((Task * volatile *) & task_dep->m_wait );

    if ( QUEUE_DEAD_VALUE == head_value_old ) {
      // Wait linked-list queue is closed
      // Try again with the next dependence
      ++i ;
    }
    else {

      // Wait queue is open and not locked.
      // If CAS succeeds then have acquired the lock.

      // Have exclusive access to this task.
      // Assign m_next assuming a successfull insertion into the queue.
      // Fence the memory assignment before attempting the CAS.

      *((Task * volatile *) & m_next ) = head_value_old ;

      memory_fence();

      // Attempt to insert this task into the queue

      Task * const wait_queue_head =
        atomic_compare_exchange( & task_dep->m_wait , head_value_old , this );

      if ( head_value_old == wait_queue_head ) {
        // Succeeded, stop trying to insert in a dependence
        try_insert_in_queue = false ;
      }
    }
  }

  //----------------------------------------

  if ( try_insert_in_queue ) {

    // All dependences are complete, insert into the ready queue.
    // Increment the count of ready tasks.
    // Count is decremented when task is complete.

    atomic_increment( & m_policy_queue->m_task_count );

    Task * volatile * const queue =
      m_cuda_serial ? & m_policy_queue->m_ready_serial 
                    : & m_policy_queue->m_ready_team ;

    while ( try_insert_in_queue ) {

      Task * const head_value_old = *queue ;

      if ( QUEUE_LOCK_VALUE != head_value_old ) {
        //  Read the head of ready queue, if same as previous value
        //  then CAS locks the ready queue
        //  Only access via CAS

        // Have exclusive access to this task, assign to head of queue,
        // assuming there will be a successful insert.
        // Fence assignment before attempting insert.
        *((Task * volatile *) & m_next ) = head_value_old ;

        memory_fence();

        Task * const ready_queue_head =
          atomic_compare_exchange( queue , head_value_old , this );

        if ( head_value_old == ready_queue_head ) {
          try_insert_in_queue = false ; // Successful insert
        }
      }
    }
  }
}

//----------------------------------------------------------------------------

__host__ __device__
void Task::assign( Task ** const lhs_ptr , Task * rhs )
{
  // Increment rhs reference count.
  if ( rhs ) { atomic_increment( & rhs->m_ref_count ); }

  // Assign the pointer and retrieve the previous value.

  Task * const old_lhs = atomic_exchange( lhs_ptr , rhs );

  if ( old_lhs ) {

    // Decrement former lhs reference count.
    // If reference count is zero task must be complete, then delete task.
    // Task is ready for deletion when old_lhs->wait == queue_denied_value().

    int const count = atomic_fetch_add( & (old_lhs->m_ref_count) , -1 ) - 1 ;

    // If 'count == 0' then this thread has exclusive access
    // to delete the task.
    // If 'count != 0' then 'old_lhs' may have been
    // deallocated by another thread before it can be queried.

    Task * const wait = 
      ( count == 0 )
      ? *((Task * const volatile *) & old_lhs->m_wait )
      : QUEUE_DEAD_VALUE ;

    if ( ( count < 0 ) || ( wait != QUEUE_DEAD_VALUE ) ) {
      Kokkos::abort("Task::assign reference count error");
    }

    if ( count == 0 ) {

      // When 'count == 0' this thread has exclusive access to 'old_lhs'.
      // Deallocation / release back into memory pool.

      CudaTaskPolicyQueue * const queue = old_lhs->m_policy_queue ;

      const unsigned size_alloc = old_lhs->m_size_alloc ;

      queue->m_space.deallocate( old_lhs , size_alloc );
    }
  }
}

//----------------------------------------------------------------------------

Task * Task::get_dependence( int i ) const
{
  Task * const t = m_dep[i] ;

  if ( Kokkos::Experimental::TASK_STATE_EXECUTING != m_state || i < 0 || m_dep_size <= i || 0 == t ) {

fprintf( stderr
       , "TaskMember< Cuda >::get_dependence ERROR : task[%lx]{ state(%d) dep_size(%d) dep[%d] = %lx }\n"
       , (unsigned long) this
       , m_state
       , m_dep_size
       , i
       , (unsigned long) t
       );
fflush( stderr );

    Kokkos::Impl::throw_runtime_exception("TaskMember< Cuda >::get_dependence ERROR");
  }

  return t ;
}

//----------------------------------------------------------------------------

__host__ __device__
void Task::add_dependence( Task * before )
{
  if ( before != 0 ) {

    int const state = *((volatile const int *) & m_state );

    // Can add dependence during construction or during execution

    if ( ( Kokkos::Experimental::TASK_STATE_CONSTRUCTING == state ||
           Kokkos::Experimental::TASK_STATE_EXECUTING    == state ) &&
         m_dep_size < m_dep_capacity ) {

      ++m_dep_size ;

      assign( m_dep + (m_dep_size-1) , before );

      memory_fence();
    }
    else {
      Kokkos::abort("Task::add_dependence to non-waiting task");
    }
  }
}

//----------------------------------------------------------------------------

void Task::clear_dependence()
{
  for ( int i = m_dep_size - 1 ; 0 <= i ; --i ) {
    assign( m_dep + i , 0 );
  }

  *((volatile int *) & m_dep_size ) = 0 ;

  memory_fence();
}

//----------------------------------------------------------------------------

__device__
Task * CudaTaskPolicyQueue::pop_ready_task( Task * volatile * const queue )
{
  Task * task = *queue ;

  if ( ( task != QUEUE_LOCK_VALUE ) &&
       ( task == atomic_compare_exchange( queue, task, QUEUE_LOCK_VALUE ) ) ) {

    // May have acquired the lock and task.
    // One or more other threads may have acquired this same task and lock
    // due to respawning ABA race condition.
    // Can only be sure of acquire with a successful state
    // transition from waiting to executing

    const int old_state =
      atomic_compare_exchange( & task->m_state
                             , int(TASK_STATE_WAITING)
                             , int(TASK_STATE_EXECUTING) );

    if ( old_state == int(TASK_STATE_WAITING) ) {

      // Successfully transitioned this task from waiting to executing.
      // Have exclusive access to this task and the queue.
      // Update the queue to the next entry and release the lock.

      Task * volatile * const next_ptr = & task->m_next ;
      Task * const next = *next_ptr ;

      if ( QUEUE_LOCK_VALUE !=
           atomic_compare_exchange( queue , QUEUE_LOCK_VALUE , next ) ) {
        Kokkos::abort("Task::pop_ready_task UNLOCK ERROR");
      }

      *next_ptr = 0 ;

      // Could memory_fence at this point; however,
      // the team has exclusive access and will perform
      // a synchthreads before accessing this data.
    }
    else {
      task = 0 ;
    }
  }
  else {
    task = 0 ;
  }

  return task ;
}


__device__
void CudaTaskPolicyQueue::complete_executed_task( Task * task )
{
  // State is either executing or if respawned then waiting,
  // try to transition from executing to complete.
  // Reads the current value.

  const int state_old =
    atomic_compare_exchange( & task->m_state
                           , int(Kokkos::Experimental::TASK_STATE_EXECUTING)
                           , int(Kokkos::Experimental::TASK_STATE_COMPLETE) );

  if ( Kokkos::Experimental::TASK_STATE_WAITING == state_old ) {
    task->schedule(); /* Task requested a respawn so reschedule it */
  }
  else if ( Kokkos::Experimental::TASK_STATE_EXECUTING != state_old ) {
    Kokkos::abort("Task::complete_executed_task STATE ERROR");
  }
  else {

    // Clear dependences of this task before locking wait queue

    task->clear_dependence();

    // Stop other tasks from adding themselves to this task's wait queue.
    // The wait queue is updated concurrently so guard with an atomic.
    // Setting the wait queue to dead denotes delete-ability of the task by any thread.
    // Therefore, once 'dead' the task pointer must be treated as invalid.

    Task * wait_queue = atomic_exchange( & task->m_wait , QUEUE_DEAD_VALUE );

    task = 0 ;

    // Have exclusive access to this task's data.
    // Pop waiting tasks and schedule them.
    while ( wait_queue ) {
      Task * const x = wait_queue ; wait_queue = x->m_next ; x->m_next = 0 ;
      x->schedule();
    }
  }

  atomic_decrement( & m_task_count );
}

//----------------------------------------------------------------------------
// Called by each block & thread

__device__
void Kokkos::Experimental::Impl::CudaTaskPolicyQueue::driver()
{
  typedef Kokkos::Experimental::Impl::CudaTaskPolicyQueue Queue ;
  typedef Kokkos::Experimental::Impl::TaskMember< Kokkos::Cuda , void , void > Task ;

  const bool is_team_lead = threadIdx.x == 0 && threadIdx.y == 0 ;

  // Each thread block must iterate this loop synchronously
  // to insure team-execution of team-task

  while ( 0 < m_task_count ) {

    __shared__ Task * team_task_ptr ;

    if ( is_team_lead ) {
      team_task_ptr = Queue::pop_ready_task( & m_ready_team );
    }

    __syncthreads();

    if ( team_task_ptr ) {

      Kokkos::Impl::CudaTeamMember
        member( kokkos_impl_cuda_shared_memory<void>()
              , 16
              , team_task_ptr->m_shmem_size
              , 0
              , 1
              );

      (*team_task_ptr->m_cuda_team)( team_task_ptr , member );
    }
    else {
      // One thread of one warp performs this serial task
      if ( threadIdx.x == 0 ) {
        Task * serial_task_ptr =
          Queue::pop_ready_task( & m_ready_serial );

        if ( serial_task_ptr ) {
          (*serial_task_ptr->m_cuda_serial)( serial_task_ptr );
        }
      }
    }
  }
}

} /* namespace Impl */
} /* namespace Experimental */
} /* namespace Kokkos */


#endif /* #if defined( KOKKOS_HAVE_CUDA ) */

