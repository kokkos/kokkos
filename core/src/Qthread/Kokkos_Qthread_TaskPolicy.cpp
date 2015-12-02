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

#include <Kokkos_Core_fwd.hpp>

#if defined( KOKKOS_HAVE_QTHREAD )

#include <stdio.h>

#include <stdlib.h>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <string>

#include <Kokkos_Atomic.hpp>
#include <Qthread/Kokkos_Qthread_TaskPolicy.hpp>

//----------------------------------------------------------------------------

namespace Kokkos {
namespace Experimental {
namespace Impl {

typedef TaskMember< Kokkos::Qthread , void , void > Task ;

namespace {

inline
unsigned padded_sizeof_derived( unsigned sizeof_derived )
{
  return sizeof_derived +
    ( sizeof_derived % sizeof(Task*) ? sizeof(Task*) - sizeof_derived % sizeof(Task*) : 0 );
}

// int lock_alloc_dealloc = 0 ;

} // namespace

void Task::deallocate( void * ptr )
{
  // Counting on 'free' thread safety so lock/unlock not required.
  // However, isolate calls here to mitigate future need to introduce lock/unlock.

  // lock

  // while ( ! Kokkos::atomic_compare_exchange_strong( & lock_alloc_dealloc , 0 , 1 ) );

  free( ptr );

  // unlock

  // Kokkos::atomic_compare_exchange_strong( & lock_alloc_dealloc , 1 , 0 );
}

void * Task::allocate( const unsigned arg_sizeof_derived
                     , const unsigned arg_dependence_capacity )
{
  // Counting on 'malloc' thread safety so lock/unlock not required.
  // However, isolate calls here to mitigate future need to introduce lock/unlock.

  // lock

  // while ( ! Kokkos::atomic_compare_exchange_strong( & lock_alloc_dealloc , 0 , 1 ) );

  void * const ptr = malloc( padded_sizeof_derived( arg_sizeof_derived ) + arg_dependence_capacity * sizeof(Task*) );

  // unlock

  // Kokkos::atomic_compare_exchange_strong( & lock_alloc_dealloc , 1 , 0 );

  return ptr ;
}

Task::~TaskMember()
{

}


Task::TaskMember( const function_verify_type        arg_verify
                , const function_dealloc_type       arg_dealloc
                , const function_apply_single_type  arg_apply_single
                , const function_apply_team_type    arg_apply_team
                , volatile int &                    arg_active_count
                , const unsigned                    arg_sizeof_derived
                , const unsigned                    arg_dependence_capacity
                )
  : m_dealloc( arg_dealloc )
  , m_verify(  arg_verify )
  , m_apply_single( arg_apply_single )
  , m_apply_team( arg_apply_team )
  , m_active_count( & arg_active_count )
  , m_qfeb(0)
  , m_dep( (Task **)( ((unsigned char *) this) + padded_sizeof_derived( arg_sizeof_derived ) ) )
  , m_dep_capacity( arg_dependence_capacity )
  , m_dep_size( 0 )
  , m_ref_count( 0 )
  , m_state( Kokkos::Experimental::TASK_STATE_CONSTRUCTING )
{
  qthread_empty( & m_qfeb ); // Set to full when complete
  for ( unsigned i = 0 ; i < arg_dependence_capacity ; ++i ) m_dep[i] = 0 ;
}

Task::TaskMember( const function_dealloc_type       arg_dealloc
                , const function_apply_single_type  arg_apply_single
                , const function_apply_team_type    arg_apply_team
                , volatile int &                    arg_active_count
                , const unsigned                    arg_sizeof_derived
                , const unsigned                    arg_dependence_capacity
                )
  : m_dealloc( arg_dealloc )
  , m_verify(  & Task::verify_type<void> )
  , m_apply_single( arg_apply_single )
  , m_apply_team( arg_apply_team )
  , m_active_count( & arg_active_count )
  , m_qfeb(0)
  , m_dep( (Task **)( ((unsigned char *) this) + padded_sizeof_derived( arg_sizeof_derived ) ) )
  , m_dep_capacity( arg_dependence_capacity )
  , m_dep_size( 0 )
  , m_ref_count( 0 )
  , m_state( Kokkos::Experimental::TASK_STATE_CONSTRUCTING )
{
  qthread_empty( & m_qfeb ); // Set to full when complete
  for ( unsigned i = 0 ; i < arg_dependence_capacity ; ++i ) m_dep[i] = 0 ;
}

//----------------------------------------------------------------------------

void Task::throw_error_add_dependence() const
{
  std::cerr << "TaskMember< Qthread >::add_dependence ERROR"
            << " state(" << m_state << ")"
            << " dep_size(" << m_dep_size << ")"
            << std::endl ;
  throw std::runtime_error("TaskMember< Qthread >::add_dependence ERROR");
}

void Task::throw_error_verify_type()
{
  throw std::runtime_error("TaskMember< Qthread >::verify_type ERROR");
}

//----------------------------------------------------------------------------

#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
void Task::assign( Task ** const lhs , Task * rhs , const bool no_throw )
{
  static const char msg_error_header[]      = "Kokkos::Impl::TaskManager<Kokkos::Qthread>::assign ERROR" ;
  static const char msg_error_count[]       = ": negative reference count" ;
  static const char msg_error_complete[]    = ": destroy task that is not complete" ;
  static const char msg_error_dependences[] = ": destroy task that has dependences" ;
  static const char msg_error_exception[]   = ": caught internal exception" ;

  if ( rhs ) { Kokkos::atomic_fetch_add( & (*rhs).m_ref_count , 1 ); }

  Task * const lhs_val = Kokkos::atomic_exchange( lhs , rhs );

  if ( lhs_val ) {

    const int count = Kokkos::atomic_fetch_add( & (*lhs_val).m_ref_count , -1 );

    const char * msg_error = 0 ;

    try {

      if ( 1 == count ) {

        // Reference count at zero, delete it

        // Should only be deallocating a completed task
        if ( (*lhs_val).m_state == Kokkos::Experimental::TASK_STATE_COMPLETE ) {

          // A completed task should not have dependences...
          for ( int i = 0 ; i < (*lhs_val).m_dep_size && 0 == msg_error ; ++i ) {
            if ( (*lhs_val).m_dep[i] ) msg_error = msg_error_dependences ;
          }
        }
        else {
          msg_error = msg_error_complete ;
        }

        if ( 0 == msg_error ) {
          // Get deletion function and apply it
          const Task::function_dealloc_type d = (*lhs_val).m_dealloc ;

          (*d)( lhs_val );
        }
      }
      else if ( count <= 0 ) {
        msg_error = msg_error_count ;
      }
    }
    catch( ... ) {
      if ( 0 == msg_error ) msg_error = msg_error_exception ;
    }

    if ( 0 != msg_error ) {
      if ( no_throw ) {
        std::cerr << msg_error_header << msg_error << std::endl ;
        std::cerr.flush();
      }
      else {
        std::string msg(msg_error_header);
        msg.append(msg_error);
        throw std::runtime_error( msg );
      }
    }
  }
}
#endif


//----------------------------------------------------------------------------

void Task::closeout()
{
  enum { RESPAWN = int( Kokkos::Experimental::TASK_STATE_WAITING ) |
                   int( Kokkos::Experimental::TASK_STATE_EXECUTING ) };

#if 0
fprintf( stdout
       , "worker(%d.%d) task 0x%.12lx %s\n"
       , qthread_shep()
       , qthread_worker_local(NULL)
       , reinterpret_cast<unsigned long>(this)
       , ( m_state == RESPAWN ? "respawn" : "complete" )
       );
fflush(stdout);
#endif

  // When dependent tasks run there would be a race
  // condition between destroying this task and
  // querying the active count pointer from this task.
  int volatile * const active_count = m_active_count ;

  if ( m_state == RESPAWN ) {
    // Task requests respawn, set state to waiting and reschedule the task
    m_state = Kokkos::Experimental::TASK_STATE_WAITING ;
    schedule();
  }
  else {

    // Task did not respawn, is complete
    m_state = Kokkos::Experimental::TASK_STATE_COMPLETE ;

    // Release dependences before allowing dependent tasks to run.
    // Otherwise there is a thread race condition for removing dependences.
    for ( int i = 0 ; i < m_dep_size ; ++i ) {
      assign( & m_dep[i] , 0 );
    }

    // Set qthread FEB to full so that dependent tasks are allowed to execute.
    // This 'task' may be deleted immediately following this function call.
    qthread_fill( & m_qfeb );

    // The dependent task could now complete and destroy 'this' task
    // before the call to 'qthread_fill' returns.  Therefore, for
    // thread safety assume that 'this' task has now been destroyed.
  }

  // Decrement active task count before returning.
  Kokkos::atomic_decrement( active_count );
}

aligned_t Task::qthread_func( void * arg )
{
  Task * const task = reinterpret_cast< Task * >(arg);

  // First member of the team change state to executing.
  // Use compare-exchange to avoid race condition with a respawn.
  Kokkos::atomic_compare_exchange_strong( & task->m_state
                                        , int(Kokkos::Experimental::TASK_STATE_WAITING)
                                        , int(Kokkos::Experimental::TASK_STATE_EXECUTING)
                                        );

  // It is a single thread's responsibility to close out
  // this task's execution.
  bool close_out = false ;

  if ( task->m_apply_team && ! task->m_apply_single ) {
    const Kokkos::Impl::QthreadTeamPolicyMember::TaskTeam task_team_tag ;

    // Initialize team size and rank with shephered info
    Kokkos::Impl::QthreadTeamPolicyMember member( task_team_tag );

    (*task->m_apply_team)( task , member );

#if 0
fprintf( stdout
       , "worker(%d.%d) task 0x%.12lx executed by member(%d:%d)\n"
       , qthread_shep()
       , qthread_worker_local(NULL)
       , reinterpret_cast<unsigned long>(task)
       , member.team_rank()
       , member.team_size()
       );
fflush(stdout);
#endif

    member.team_barrier();
    if ( member.team_rank() == 0 ) task->closeout();
    member.team_barrier();
  }
  else if ( task->m_apply_team && task->m_apply_single == reinterpret_cast<function_apply_single_type>(1) ) {
    // Team hard-wired to one, no cloning
    Kokkos::Impl::QthreadTeamPolicyMember member ;
    (*task->m_apply_team)( task , member );
    task->closeout();
  }
  else {
    (*task->m_apply_single)( task );
    task->closeout();
  }

#if 0
fprintf( stdout
       , "worker(%d.%d) task 0x%.12lx return\n"
       , qthread_shep()
       , qthread_worker_local(NULL)
       , reinterpret_cast<unsigned long>(task)
       );
fflush(stdout);
#endif

  return 0 ;
}

void Task::respawn()
{
  // Change state from pure executing to ( waiting | executing )
  // to avoid confusion with simply waiting.
  Kokkos::atomic_compare_exchange_strong( & m_state
                                        , int(Kokkos::Experimental::TASK_STATE_EXECUTING)
                                        , int(Kokkos::Experimental::TASK_STATE_WAITING |
                                              Kokkos::Experimental::TASK_STATE_EXECUTING)
                                        );
}

void Task::schedule()
{
  // Is waiting for execution

  // Increment active task count before spawning.
  Kokkos::atomic_increment( m_active_count );

  // spawn in qthread.  must malloc the precondition array and give to qthread.
  // qthread will eventually free this allocation so memory will not be leaked.

  // concern with thread safety of malloc, does this need to be guarded?
  aligned_t ** qprecon = (aligned_t **) malloc( ( m_dep_size + 1 ) * sizeof(aligned_t *) );

  qprecon[0] = reinterpret_cast<aligned_t *>( uintptr_t(m_dep_size) );

  for ( int i = 0 ; i < m_dep_size ; ++i ) {
    qprecon[i+1] = & m_dep[i]->m_qfeb ; // Qthread precondition flag
  }

  if ( m_apply_team && ! m_apply_single ) {
    // If more than one shepherd spawn on a shepherd other than this shepherd
    const int num_shepherd            = qthread_num_shepherds();
    const int num_worker_per_shepherd = qthread_num_workers_local(NO_SHEPHERD);
    const int this_shepherd           = qthread_shep();

    int spawn_shepherd = ( this_shepherd + 1 ) % num_shepherd ;

#if 0
fprintf( stdout
       , "worker(%d.%d) task 0x%.12lx spawning on shepherd(%d) clone(%d)\n"
       , qthread_shep()
       , qthread_worker_local(NULL)
       , reinterpret_cast<unsigned long>(this)
       , spawn_shepherd
       , num_worker_per_shepherd - 1
       );
fflush(stdout);
#endif

    qthread_spawn_cloneable
      ( & Task::qthread_func
      , this
      , 0
      , NULL
      , m_dep_size , qprecon /* dependences */
      , spawn_shepherd
      , unsigned( QTHREAD_SPAWN_SIMPLE | QTHREAD_SPAWN_LOCAL_PRIORITY )
      , num_worker_per_shepherd - 1
      );
  }
  else {
    qthread_spawn( & Task::qthread_func /* function */
                 , this                 /* function argument */
                 , 0
                 , NULL
                 , m_dep_size , qprecon /* dependences */
                 , NO_SHEPHERD
                 , QTHREAD_SPAWN_SIMPLE /* allows optimization for non-blocking task */
                 );
  }
}

} // namespace Impl
} // namespace Experimental
} // namespace Kokkos

namespace Kokkos {
namespace Experimental {

TaskPolicy< Kokkos::Qthread >::
  TaskPolicy( const unsigned arg_default_dependence_capacity
            , const unsigned arg_team_size )
  : m_default_dependence_capacity( arg_default_dependence_capacity )
  , m_team_size( arg_team_size != 0 ? arg_team_size : unsigned(qthread_num_workers_local(NO_SHEPHERD)) )
  , m_active_count_root(0)
  , m_active_count( m_active_count_root )
{
  const unsigned num_worker_per_shepherd = unsigned( qthread_num_workers_local(NO_SHEPHERD) );

  if ( m_team_size != 1 && m_team_size != num_worker_per_shepherd ) {
    std::ostringstream msg ;
    msg << "Kokkos::Experimental::TaskPolicy< Kokkos::Qthread >( "
        << "default_depedence = " << arg_default_dependence_capacity
        << " , team_size = " << arg_team_size
        << " ) ERROR, valid team_size arguments are { (omitted) , 1 , " << num_worker_per_shepherd << " }" ;
    Kokkos::Impl::throw_runtime_exception(msg.str());
  }
}

TaskPolicy< Kokkos::Qthread >::member_type &
TaskPolicy< Kokkos::Qthread >::member_single()
{
  static member_type s ;
  return s ;
}

void wait( Kokkos::Experimental::TaskPolicy< Kokkos::Qthread > & policy )
{
  volatile int * const active_task_count = & policy.m_active_count ;
  while ( *active_task_count ) qthread_yield();
}

} // namespace Experimental
} // namespace Kokkos

#endif /* #if defined( KOKKOS_HAVE_QTHREAD ) */

