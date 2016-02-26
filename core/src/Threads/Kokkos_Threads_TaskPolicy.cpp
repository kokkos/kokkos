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
#include <Threads/Kokkos_Threads_TaskPolicy.hpp>

#if defined( KOKKOS_HAVE_PTHREAD )

namespace Kokkos {
namespace Experimental {
namespace Impl {

typedef TaskMember< Kokkos::Threads , void , void > Task ;

#define QLOCK   (reinterpret_cast<Task*>( ~((uintptr_t)0) ))
#define QDENIED (reinterpret_cast<Task*>( ~((uintptr_t)0) - 1 ))

} /* namespace Impl */
} /* namespace Experimental */
} /* namespace Kokkos */

namespace Kokkos {
namespace Experimental {
namespace Impl {

void ThreadsTaskPolicyQueue::Destroy::destroy_shared_allocation()
{
  // Verify the queue is empty

  if ( m_policy->m_count_ready ||
       m_policy->m_team[0] ||
       m_policy->m_team[1] ||
       m_policy->m_team[2] ||
       m_policy->m_serial[0] ||
       m_policy->m_serial[1] ||
       m_policy->m_serial[2] ) {
    Kokkos::abort("ThreadsTaskPolicyQueue ERROR : Attempt to destroy non-empty queue" );
  }

  m_policy->~ThreadsTaskPolicyQueue();
}

//----------------------------------------------------------------------------

ThreadsTaskPolicyQueue::~ThreadsTaskPolicyQueue()
{
}

ThreadsTaskPolicyQueue::ThreadsTaskPolicyQueue
  ( const unsigned arg_task_max_count
  , const unsigned arg_task_max_size
  , const unsigned arg_task_default_dependence_capacity
  , const unsigned arg_task_team_size
  )
  : m_space( Kokkos::Threads::memory_space()
           , arg_task_max_size
           , arg_task_max_size * arg_task_max_count
           )
  , m_team { 0 , 0 , 0 }
  , m_serial { 0 , 0 , 0 }
  , m_count_ready(0)
  , m_team_size( arg_task_team_size )
  , m_default_dependence_capacity( arg_task_default_dependence_capacity )
{
  const int threads_total    = Threads::thread_pool_size(0);
  const int threads_per_numa = Threads::thread_pool_size(1);
  const int threads_per_core = Threads::thread_pool_size(2);

  if ( 0 == m_team_size ) {
    // If a team task then claim for execution until count is zero
    // Issue: team collectives cannot assume which pool members are in the team.
    // Issue: team must only span a single NUMA region.

    // If more than one thread per core then map cores to work team,
    // else  map numa to work team.

    if      ( 1 < threads_per_core ) m_team_size = threads_per_core ;
    else if ( 1 < threads_per_numa ) m_team_size = threads_per_numa ;
    else                             m_team_size = 1 ;
  }

  // Verify a valid team size
  const bool valid_team_size =
    ( 0 < m_team_size && m_team_size <= threads_total ) &&
    (
      ( 1                == m_team_size ) ||
      ( threads_per_core == m_team_size ) ||
      ( threads_per_numa == m_team_size )
    );

  if ( ! valid_team_size ) {
    std::ostringstream msg ;

    msg << "Kokkos::Experimental::TaskPolicy< Kokkos::Threads > ERROR"
        << " invalid team_size(" << m_team_size << ")"
        << " threads_per_core(" << threads_per_core << ")"
        << " threads_per_numa(" << threads_per_numa << ")"
        << " threads_total(" << threads_total << ")"
        ;

    Kokkos::Impl::throw_runtime_exception( msg.str() );
  }

  Kokkos::memory_fence();
}

//----------------------------------------------------------------------------

void ThreadsTaskPolicyQueue::driver( Kokkos::Impl::ThreadsExec & exec
                                   , const void * arg )
{
  // Whole thread pool is calling this function

  typedef Kokkos::Impl::ThreadsExecTeamMember member_type ;

  ThreadsTaskPolicyQueue & self =
    * reinterpret_cast< ThreadsTaskPolicyQueue * >( const_cast<void*>(arg) );

  // Create the thread team member with shared memory for the given task.

  const TeamPolicy< Kokkos::Threads > team_policy( 1 , self.m_team_size );

  member_type team_member( & exec , team_policy , 0 );

  Kokkos::Impl::ThreadsExec & exec_team_base =
    team_member.threads_exec_team_base();

  task_root_type * volatile * const task_team_ptr =
    reinterpret_cast<Task**>( exec_team_base.reduce_memory() );

  volatile int * const work_team_ptr =
    reinterpret_cast<volatile int*>( task_team_ptr + 1 );

  // Each team must iterate this loop synchronously
  // to insure team-execution of team-task.

  const bool team_lead = team_member.team_fan_in();

  bool work_team = true ;

  while ( work_team ) {

    task_root_type * task = 0 ;

    // Start here with members in a fan_in state

    if ( team_lead ) {
      // Team lead queries the ready count for a team-consistent view.
      *work_team_ptr = 0 != self.m_count_ready ;

      // Only the team lead attempts to pop a team task from the queues
      for ( int i = 0 ; i < int(NPRIORITY) && 0 == task ; ++i ) {
        if ( ( i < 2 /* regular queue */ )
             || ( ! self.m_space.is_empty() /* waiting for memory queue */ ) ) {
          task = pop_ready_task( & self.m_team[i] );
        }
      }

      *task_team_ptr = task ;
    }

    Kokkos::memory_fence();

    team_member.team_fan_out();

    work_team = *work_team_ptr ;

    // Query if team acquired a team task

    if ( 0 != ( task = *task_team_ptr ) ) {
      // Set shared memory
      team_member.set_league_shmem( 0 , 1 , task->m_shmem_size );

      (*task->m_team)( task , team_member );

      // The team task called the functor,
      // called the team_fan_in(), and
      // if completed the team lead destroyed the task functor.

      if ( team_lead ) {
        self.complete_executed_task( task );
      }
    }
    else {
      // No team task acquired, each thread try a serial task
      // Try the priority queue, then the regular queue.
      for ( int i = 0 ; i < int(NPRIORITY) && 0 == task ; ++i ) {
        if ( ( i < 2 /* regular queue */ )
             || ( ! self.m_space.is_empty() /* waiting for memory queue */ ) ) {
          task = pop_ready_task( & self.m_serial[i] );
        }
      }

      if ( 0 != task ) {

        (*task->m_serial)( task );

        self.complete_executed_task( task );
      }

      team_member.team_fan_in();
    }
  }

  team_member.team_fan_out();

  exec.fan_in();
}

//----------------------------------------------------------------------------

ThreadsTaskPolicyQueue::task_root_type *
ThreadsTaskPolicyQueue::pop_ready_task(
  ThreadsTaskPolicyQueue::task_root_type * volatile * const queue )
{
  task_root_type * const q_lock = QLOCK ;
  task_root_type * task = 0 ;
  task_root_type * const task_claim = *queue ;

  if ( ( q_lock != task_claim ) && ( 0 != task_claim ) ) {

    // Queue is not locked and not null, try to claim head of queue.
    // Is a race among threads to claim the queue.

    if ( task_claim == atomic_compare_exchange(queue,task_claim,q_lock) ) {

      // Aquired the task which must be in the waiting state.

      const int claim_state =
        atomic_compare_exchange( & task_claim->m_state
                               , int(TASK_STATE_WAITING)
                               , int(TASK_STATE_EXECUTING) );

      task_root_type * lock_verify = 0 ;

      if ( claim_state == int(TASK_STATE_WAITING) ) {

        // Transitioned this task from waiting to executing
        // Update the queue to the next entry and release the lock

        task_root_type * const next =
          *((task_root_type * volatile *) & task_claim->m_next );

        *((task_root_type * volatile *) & task_claim->m_next ) = 0 ;

        lock_verify = atomic_compare_exchange( queue , q_lock , next );
      }

      if ( ( claim_state != int(TASK_STATE_WAITING) ) |
           ( q_lock != lock_verify ) ) {

        fprintf(stderr,"ThreadsTaskPolicyQueue::pop_ready_task(0x%lx) task(0x%lx) state(%d) ERROR %s\n"
                      , (unsigned long) queue
                      , (unsigned long) task
                      , claim_state
                      , ( claim_state != int(TASK_STATE_WAITING)
                        ? "NOT WAITING"
                        : "UNLOCK" ) );
        fflush(stderr);
        Kokkos::abort("ThreadsTaskPolicyQueue::pop_ready_task");
      }

      task = task_claim ;
    }
  }

  return task ;
}

//----------------------------------------------------------------------------

void ThreadsTaskPolicyQueue::complete_executed_task(
  ThreadsTaskPolicyQueue::task_root_type * const task )
{
  task_root_type * const q_denied = QDENIED ;

  // State is either executing or if respawned then waiting,
  // try to transition from executing to complete.
  // Reads the current value.

  const int state_old =
    atomic_compare_exchange( & task->m_state
                           , int(Kokkos::Experimental::TASK_STATE_EXECUTING)
                           , int(Kokkos::Experimental::TASK_STATE_COMPLETE) );

  if ( int(Kokkos::Experimental::TASK_STATE_WAITING) == state_old ) {
    /* Task requested a respawn so reschedule it */
    schedule_task( task );
  }
  else if ( int(Kokkos::Experimental::TASK_STATE_EXECUTING) == state_old ) {
    /* Task is complete */

    // Clear dependences of this task before locking wait queue

    task->clear_dependence();

    // Stop other tasks from adding themselves to this task's wait queue.
    // The wait queue is updated concurrently so guard with an atomic.
    // Setting the wait queue to denied denotes delete-ability of the task by any thread.
    // Therefore, once 'denied' the task pointer must be treated as invalid.

    Task * wait_queue     = *((Task * volatile *) & task->m_wait );
    Task * wait_queue_old = 0 ;

    do {
      wait_queue_old = wait_queue ;
      wait_queue     = atomic_compare_exchange( & task->m_wait , wait_queue_old , q_denied );
    } while ( wait_queue_old != wait_queue );

    // 'task' pointer is now invalid

    // Pop waiting tasks and schedule them
    while ( wait_queue ) {
      Task * const x = wait_queue ; wait_queue = x->m_next ; x->m_next = 0 ;
      schedule_task( x );
    }
  }
  else {
    fprintf( stderr
           , "ThreadsTaskPolicyQueue::complete_executed_task(0x%lx) ERROR state_old(%d) dep_size(%d)\n"
           , (unsigned long)( task )
           , int(state_old)
           , task->m_dep_size
           );
    fflush( stderr );
    Kokkos::abort("ThreadsTaskPolicyQueue::complete_executed_task" );
  }

  // If the task was respawned it may have already been
  // put in a ready queue and the count incremented.
  // By decrementing the count last it will never go to zero
  // with a ready or executing task.

  atomic_fetch_add( & m_count_ready , -1 );
}

//----------------------------------------------------------------------------

void ThreadsTaskPolicyQueue::reschedule_task(
  ThreadsTaskPolicyQueue::task_root_type * const task )
{
  // Reschedule transitions from executing back to waiting.
  const int old_state =
    atomic_compare_exchange( & task->m_state
                           , int(TASK_STATE_EXECUTING)
                           , int(TASK_STATE_WAITING) );

  if ( old_state != int(TASK_STATE_EXECUTING) ) {

    fprintf( stderr
           , "ThreadsTaskPolicyQueue::reschedule_task(0x%lx) ERROR state(%d)\n"
           , (unsigned long) task
           , old_state
           );
    fflush(stderr);
    Kokkos::abort("ThreadsTaskPolicyQueue::reschedule" );
  }
}

void ThreadsTaskPolicyQueue::schedule_task(
  ThreadsTaskPolicyQueue::task_root_type * const task )
{
  task_root_type * const q_lock = QLOCK ;
  task_root_type * const q_denied = QDENIED ;

  //----------------------------------------
  // State is either constructing or already waiting.
  // If constructing then transition to waiting.

  {
    const int old_state = atomic_compare_exchange( & task->m_state
                                                 , int(TASK_STATE_CONSTRUCTING)
                                                 , int(TASK_STATE_WAITING) );

    // Head of linked list of tasks waiting on this task
    task_root_type * const waitTask =
      *((task_root_type * volatile const *) & task->m_wait );

    // Member of linked list of tasks waiting on some other task
    task_root_type * const next =
      *((task_root_type * volatile const *) & task->m_next );

    // An incomplete and non-executing task has:
    //   task->m_state == TASK_STATE_CONSTRUCTING or TASK_STATE_WAITING
    //   task->m_wait  != q_denied
    //   task->m_next  == 0
    //
    if ( ( q_denied == waitTask ) ||
         ( 0 != next ) ||
         ( old_state != int(TASK_STATE_CONSTRUCTING) &&
           old_state != int(TASK_STATE_WAITING) ) ) {
      fprintf(stderr,"ThreadsTaskPolicyQueue::schedule_task(0x%lx) STATE ERROR: state(%d) wait(0x%lx) next(0x%lx)\n"
                    , (unsigned long) task
                    , old_state
                    , (unsigned long) waitTask
                    , (unsigned long) next );
      fflush(stderr);
      Kokkos::abort("ThreadsTaskPolicyQueue::schedule" );
    }
  }

  //----------------------------------------
  // Insert this task into a dependence task that is not complete.
  // Push on to that task's wait queue.

  bool attempt_insert_in_queue = true ;

  task_root_type * volatile * queue =
    task->m_dep_size ? & task->m_dep[0]->m_wait : (task_root_type **) 0 ;

  for ( int i = 0 ; attempt_insert_in_queue && ( 0 != queue ) ; ) {

    task_root_type * const head_value_old = *queue ;

    if ( q_denied == head_value_old ) {
      // Wait queue is closed because task is complete,
      // try again with the next dependence wait queue.
      ++i ;
      queue = i < task->m_dep_size ? & task->m_dep[i]->m_wait
                                   : (task_root_type **) 0 ;
    }
    else {

      // Wait queue is open and not denied.
      // Have exclusive access to this task.
      // Assign m_next assuming a successfull insertion into the queue.
      // Fence the memory assignment before attempting the CAS.

      *((task_root_type * volatile *) & task->m_next ) = head_value_old ;

      memory_fence();

      // Attempt to insert this task into the queue.
      // If fails then continue the attempt.

      attempt_insert_in_queue =
        head_value_old != atomic_compare_exchange(queue,head_value_old,task);
    }
  }

  //----------------------------------------
  // All dependences are complete, insert into the ready list

  if ( attempt_insert_in_queue ) {

    // Increment the count of ready tasks.
    // Count will be decremented when task is complete.

    atomic_fetch_add( & m_count_ready , 1 );

    queue = task->m_queue ;

    while ( attempt_insert_in_queue ) {

      // A locked queue is being popped.

      task_root_type * const head_value_old = *queue ;

      if ( q_lock != head_value_old ) {
        // Read the head of ready queue,
        // if same as previous value then CAS locks the ready queue

        // Have exclusive access to this task,
        // assign to head of queue, assuming successful insert
        // Fence assignment before attempting insert.
        *((task_root_type * volatile *) & task->m_next ) = head_value_old ;

        memory_fence();

        attempt_insert_in_queue =
          head_value_old != atomic_compare_exchange(queue,head_value_old,task);
      }
    }
  }
}

/*
namespace {

int alloc_count = 0 ;

}
*/

void * ThreadsTaskPolicyQueue::allocate_task( unsigned size_alloc )
{
  void * const ptr = m_space.allocate( size_alloc );

/*
  const int n = atomic_fetch_add( & alloc_count , 1 ) + 1 ;

  fprintf( stderr
         , "ThreadsTaskPolicyQueue::allocate_task(%d) ptr(0x%lx) count(%d)\n"
         , size_alloc
         , (unsigned long) ptr
         , n
         );
  fflush( stderr );
*/

  return ptr ;
}

void ThreadsTaskPolicyQueue::deallocate_task( void * ptr , unsigned size_alloc )
{
/*
  const int n = atomic_fetch_add( & alloc_count , -1 ) - 1 ;

  fprintf( stderr
         , "ThreadsTaskPolicyQueue::deallocate_task(0x%lx,%d) count(%d)\n"
         , (unsigned long) ptr
         , size_alloc
         , n
         );
  fflush( stderr );
*/

  m_space.deallocate( ptr , size_alloc );
}

//----------------------------------------------------------------------------

void ThreadsTaskPolicyQueue::
  add_dependence( Task * const after ,  Task * const before )
{
  if ( ( after != 0 ) && ( before != 0 ) ) {

    int const state = *((volatile const int *) & after->m_state );

    // Only add dependence during construction or during execution.
    // Both tasks must have the same policy.
    // Dependence on non-full memory cannot be mixed with any other dependence.

    const bool ok_state =
      Kokkos::Experimental::TASK_STATE_CONSTRUCTING == state ||
      Kokkos::Experimental::TASK_STATE_EXECUTING    == state ;

    const bool ok_capacity =
      after->m_dep_size < after->m_dep_capacity ;

    const bool ok_policy =
      after->m_policy == this && before->m_policy == this ;

    if ( ok_state && ok_capacity && ok_policy ) {

      ++after->m_dep_size ;

      task_root_type::assign( after->m_dep + (after->m_dep_size-1) , before );

      memory_fence();
    }
    else {

fprintf( stderr
       , "ThreadsTaskPolicyQueue::add_dependence( 0x%lx , 0x%lx ) ERROR %s\n"
       , (unsigned long) after
       , (unsigned long) before
       , ( ! ok_state    ? "Task not constructing or executing" :
         ( ! ok_capacity ? "Task Exceeded dependence capacity" 
                         : "Tasks from different policies" 
         )) );

fflush( stderr );

      Kokkos::abort("ThreadsTaskPolicyQueue::add_dependence ERROR");
    }
  }
}
} /* namespace Impl */
} /* namespace Experimental */
} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Experimental {

TaskPolicy< Kokkos::Threads >::TaskPolicy
  ( const unsigned arg_task_max_count
  , const unsigned arg_task_max_size
  , const unsigned arg_task_default_dependence_capacity
  , const unsigned arg_task_team_size
  )
  : m_track()
  , m_policy(0)
{
  typedef Kokkos::Experimental::Impl::SharedAllocationRecord
    < Kokkos::HostSpace , Impl::ThreadsTaskPolicyQueue::Destroy > record_type ;

  record_type * record =
    record_type::allocate( Kokkos::HostSpace()
                         , "Threads task queue"
                         , sizeof(Impl::ThreadsTaskPolicyQueue)
                         );

  m_policy =
    reinterpret_cast< Impl::ThreadsTaskPolicyQueue * >( record->data() );

  new( m_policy )
    Impl::ThreadsTaskPolicyQueue( arg_task_max_count
                                , arg_task_max_size
                                , arg_task_default_dependence_capacity
                                , arg_task_team_size );

  record->m_destroy.m_policy = m_policy ;

  m_track.assign_allocated_record_to_uninitialized( record );
}


TaskPolicy< Kokkos::Threads >::member_type &
TaskPolicy< Kokkos::Threads >::member_single()
{
  static member_type s ;
  return s ;
}

void wait( Kokkos::Experimental::TaskPolicy< Kokkos::Threads > & policy )
{
  typedef Kokkos::Impl::ThreadsExecTeamMember member_type ;

  enum { BASE_SHMEM = 1024 };

  Kokkos::Impl::ThreadsExec::resize_scratch( 0 , member_type::team_reduce_size() + BASE_SHMEM );

  Kokkos::Impl::ThreadsExec::start( & Impl::ThreadsTaskPolicyQueue::driver
                                  , policy.m_policy );

  Kokkos::Impl::ThreadsExec::fence();
}

} /* namespace Experimental */
} /* namespace Kokkos */

namespace Kokkos {
namespace Experimental {
namespace Impl {

//----------------------------------------------------------------------------

Task::~TaskMember()
{
}

//----------------------------------------------------------------------------

#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )

void Task::assign( Task ** const lhs_ptr , Task * rhs )
{
  Task * const q_denied = QDENIED ;

  // Increment rhs reference count.
  if ( rhs ) { atomic_increment( & rhs->m_ref_count ); }

  // Assign the pointer and retrieve the previous value.

  Task * const old_lhs = atomic_exchange( lhs_ptr , rhs );

  if ( old_lhs && rhs && old_lhs->m_policy != rhs->m_policy ) {
    Kokkos::abort( "Kokkos::Impl::TaskMember<Kokkos::Threads>::assign ERROR different queues");
  }

  if ( old_lhs ) {

    // Decrement former lhs reference count.
    // If reference count is zero task must be complete, then delete task.
    // Task is ready for deletion when  wait == q_denied

    int const count = atomic_fetch_add( & (old_lhs->m_ref_count) , -1 ) - 1 ;
    int const state = old_lhs->m_state ;
    Task * const wait = *((Task * const volatile *) & old_lhs->m_wait );

    const bool ok_count = 0 <= count ;

    // If count == 0 then will be deleting
    // and must either be constructing or complete.
    const bool ok_state = 0 < count ? true :
      ( ( state == int(TASK_STATE_CONSTRUCTING) && wait == 0 ) ||
        ( state == int(TASK_STATE_COMPLETE)     && wait == q_denied ) )
      &&
     old_lhs->m_next == 0 &&
     old_lhs->m_dep_size == 0 ;

    if ( ! ok_count || ! ok_state ) {

      fprintf( stderr , "Kokkos::Impl::TaskManager<Kokkos::Threads>::assign ERROR deleting task(0x%lx) m_ref_count(%d) , m_wait(0x%ld)\n"
                      , (unsigned long) old_lhs
                      , count
                      , (unsigned long) wait );
      fflush(stderr);
      Kokkos::abort( "Kokkos::Impl::TaskMember<Kokkos::Threads>::assign ERROR deleting");
    }

    if ( count == 0 ) {
      // When 'count == 0' this thread has exclusive access to 'old_lhs'

      ThreadsTaskPolicyQueue & queue = *( old_lhs->m_policy );

      queue.deallocate_task( old_lhs , old_lhs->m_size_alloc );
    }
  }
}

#endif

//----------------------------------------------------------------------------

Task * Task::get_dependence( int i ) const
{
  Task * const t = m_dep[i] ;

  if ( Kokkos::Experimental::TASK_STATE_EXECUTING != m_state || i < 0 || m_dep_size <= i || 0 == t ) {

fprintf( stderr
       , "TaskMember< Threads >::get_dependence ERROR : task[%lx]{ state(%d) dep_size(%d) dep[%d] = %lx }\n"
       , (unsigned long) this
       , m_state
       , m_dep_size
       , i
       , (unsigned long) t
       );
fflush( stderr );

    Kokkos::Impl::throw_runtime_exception("TaskMember< Threads >::get_dependence ERROR");
  }

  return t ;
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

} /* namespace Impl */
} /* namespace Experimental */
} /* namespace Kokkos */

#endif /* #if defined( KOKKOS_HAVE_PTHREAD ) */

