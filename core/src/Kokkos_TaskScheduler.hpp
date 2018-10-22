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

#ifndef KOKKOS_TASKSCHEDULER_HPP
#define KOKKOS_TASKSCHEDULER_HPP

//----------------------------------------------------------------------------

#include <Kokkos_Macros.hpp>
#if defined( KOKKOS_ENABLE_TASKDAG )

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_TaskScheduler_fwd.hpp>
//----------------------------------------------------------------------------

#include <Kokkos_MemoryPool.hpp>
#include <impl/Kokkos_Tags.hpp>

#include <Kokkos_Future.hpp>
#include <impl/Kokkos_TaskQueue.hpp>
#include <impl/Kokkos_TaskQueueMultiple.hpp>
#include <impl/Kokkos_TaskPolicyData.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

namespace Impl {

struct TaskSchedulerBase { };

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
      m_scheduler(std::move(arg_scheduler).get_team_scheduler(this->league_rank()))
  { }

  KOKKOS_INLINE_FUNCTION
  TaskTeamMemberAdapter() = default;

  KOKKOS_INLINE_FUNCTION
  TaskTeamMemberAdapter(TaskTeamMemberAdapter const&) = default;

  KOKKOS_INLINE_FUNCTION
  TaskTeamMemberAdapter(TaskTeamMemberAdapter&&) = default;

  KOKKOS_INLINE_FUNCTION
  TaskTeamMemberAdapter& operator=(TaskTeamMemberAdapter const&) = default;

  KOKKOS_INLINE_FUNCTION
  TaskTeamMemberAdapter& operator=(TaskTeamMemberAdapter&&) = default;

  KOKKOS_INLINE_FUNCTION ~TaskTeamMemberAdapter() = default;

  //----------------------------------------

  KOKKOS_INLINE_FUNCTION
  Scheduler const& scheduler() const noexcept { return m_scheduler; }

  KOKKOS_INLINE_FUNCTION
  Scheduler& scheduler() noexcept { return m_scheduler; }

  //----------------------------------------

};

template <class, class>
class TaskExec;

} // end namespace Impl

template<class ExecSpace, class QueueType>
class BasicTaskScheduler : public Impl::TaskSchedulerBase
{
public:

  using scheduler_type = BasicTaskScheduler;
  using execution_space = ExecSpace;
  using queue_type = QueueType;
  using memory_space = typename queue_type::memory_space;
  using memory_pool = typename queue_type::memory_pool;
  using specialization = Impl::TaskQueueSpecialization<BasicTaskScheduler>;
  using member_type = typename specialization::member_type;
  using team_scheduler_type = BasicTaskScheduler;

private:

  using track_type = Kokkos::Impl::SharedAllocationTracker ;
  using task_base  = Impl::TaskBase;

  track_type m_track;
  queue_type * m_queue;

  //----------------------------------------

  template <typename, typename>
  friend class Impl::TaskQueue;
  template <typename>
  friend class Impl::TaskQueueSpecialization;
  template <typename, typename>
  friend class Impl::TaskQueueSpecializationConstrained;
  template <typename, typename>
  friend class Impl::TaskTeamMemberAdapter;
  template <typename, typename>
  friend class Impl::TaskExec;

  //----------------------------------------

  KOKKOS_INLINE_FUNCTION
  BasicTaskScheduler(
    track_type arg_track,
    queue_type* arg_queue
  )
    : m_track(std::move(arg_track)),
      m_queue(std::move(arg_queue))
  { }

  KOKKOS_INLINE_FUNCTION
  team_scheduler_type get_team_scheduler(int team_rank) const {
    return { m_track, &m_queue->get_team_queue(team_rank) };
  }

  KOKKOS_INLINE_FUNCTION
  constexpr queue_type& queue() const noexcept {
    return *m_queue;
  }

public:


  KOKKOS_INLINE_FUNCTION
  BasicTaskScheduler() : m_track(), m_queue(0) {}

  KOKKOS_INLINE_FUNCTION
  BasicTaskScheduler( BasicTaskScheduler && rhs ) noexcept
    : m_track(rhs.m_track), // TODO should this be moved?
      m_queue(std::move(rhs.m_queue))
  { }

  KOKKOS_INLINE_FUNCTION
  BasicTaskScheduler( BasicTaskScheduler const & rhs )
    : m_track(rhs.m_track),
      m_queue(rhs.m_queue)
  { }

  KOKKOS_INLINE_FUNCTION
  BasicTaskScheduler& operator=(BasicTaskScheduler&& rhs) noexcept
  {
    m_track = rhs.m_track; // TODO should this be moved?
    m_queue = std::move(rhs.m_queue);
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  BasicTaskScheduler& operator=(BasicTaskScheduler const& rhs)
  {
    m_track = rhs.m_track;
    m_queue = rhs.m_queue;
    return *this;
  }

  explicit BasicTaskScheduler(memory_pool const & arg_memory_pool) noexcept
    : m_track(), m_queue(0)
    {
      typedef Kokkos::Impl::SharedAllocationRecord
        < memory_space , typename queue_type::Destroy >
          record_type ;

      record_type * record =
        record_type::allocate( memory_space()
                             , "TaskQueue"
                             , sizeof(queue_type)
                             );

      m_queue = new( record->data() ) queue_type( arg_memory_pool );

      record->m_destroy.m_queue = m_queue ;

      m_track.assign_allocated_record_to_uninitialized( record );
    }

  BasicTaskScheduler( memory_space const & arg_memory_space
               , size_t const mempool_capacity
               , unsigned const mempool_min_block_size  // = 1u << 6
               , unsigned const mempool_max_block_size  // = 1u << 10
               , unsigned const mempool_superblock_size // = 1u << 12
               )
    : BasicTaskScheduler( memory_pool( arg_memory_space
                                , mempool_capacity
                                , mempool_min_block_size
                                , mempool_max_block_size
                                , mempool_superblock_size ) )
    {}

  //----------------------------------------

  KOKKOS_INLINE_FUNCTION
  memory_pool * memory() const noexcept
    { return m_queue ? &( m_queue->m_memory ) : (memory_pool*) 0 ; }

  //----------------------------------------
  /**\brief  Allocation size for a spawned task */
  template< typename FunctorType >
  KOKKOS_FUNCTION
  size_t spawn_allocation_size() const
    { return m_queue->template spawn_allocation_size< FunctorType >(); }

  /**\brief  Allocation size for a when_all aggregate */
  KOKKOS_FUNCTION
  size_t when_all_allocation_size( int narg ) const
    { return m_queue->when_all_allocation_size( narg ); }


  //----------------------------------------

  template< int TaskEnum , typename DepFutureType , typename FunctorType >
  KOKKOS_FUNCTION static
  Kokkos::BasicFuture<typename FunctorType::value_type, scheduler_type>
  spawn( Impl::TaskPolicyData<TaskEnum,DepFutureType> const & arg_policy
       , typename task_base::function_type                    arg_function
       , typename task_base::destroy_type                    arg_destory
       , FunctorType                                       && arg_functor
       )
    {
      using value_type  = typename FunctorType::value_type ;
      using future_type = BasicFuture< value_type , scheduler_type > ;
      using task_type = Impl::Task<BasicTaskScheduler, value_type, FunctorType>;

      BasicTaskScheduler const* scheduler_ptr = nullptr;
      if(arg_policy.m_scheduler != nullptr) {
        scheduler_ptr = static_cast<BasicTaskScheduler const*>(
          arg_policy.m_scheduler
        );
      }
      else if(arg_policy.m_dependence.m_task != nullptr) {
        scheduler_ptr = static_cast<BasicTaskScheduler const*>(
          arg_policy.m_dependence.m_task->m_scheduler
        );
      }

      if(scheduler_ptr == nullptr) {
        Kokkos::abort("Kokkos spawn requires scheduler or non-null Future");
        return {}; // should be unreachable
      }

      auto& scheduler = *scheduler_ptr;
      auto& queue = *scheduler.m_queue;

      if (
        arg_policy.m_dependence.m_task != 0
        && static_cast<BasicTaskScheduler const*>(
          arg_policy.m_dependence.m_task->m_scheduler
        )->m_queue != &queue
      ) {
        Kokkos::abort("Kokkos spawn given incompatible scheduler and Future");
      }

      //----------------------------------------
      // Give single-thread back-ends an opportunity to clear
      // queue of ready tasks before allocating a new task

      specialization::iff_single_thread_recursive_execute(scheduler);

      //----------------------------------------

      future_type f ;

      // Allocate task from memory pool

      const size_t alloc_size =
        queue.template spawn_allocation_size< FunctorType >();

      f.m_task =
        reinterpret_cast< task_type * >(queue.allocate(alloc_size) );

      if ( f.m_task ) {

        // Placement new construction
        // Reference count starts at two:
        //   +1 for the matching decrement when task is complete
        //   +1 for the future
        new ( f.m_task ) task_type( std::forward<FunctorType>(arg_functor) );

        f.m_task->m_apply      = arg_function;
        f.m_task->m_destroy    = arg_destory;
        f.m_task->m_scheduler  = scheduler_ptr; // may need to change before running
        f.m_task->m_next       = arg_policy.m_dependence.m_task;
        f.m_task->m_ref_count  = 2;
        f.m_task->m_alloc_size = alloc_size;
        f.m_task->m_task_type  = arg_policy.m_task_type;
        f.m_task->m_priority   = arg_policy.m_priority;

        Kokkos::memory_fence();

        // The dependence (if any) is processed immediately
        // within the schedule function, as such the dependence's
        // reference count does not need to be incremented for
        // the assignment.

        queue.schedule_runnable( f.m_task );
        // This task may be updated or executed at any moment,
        // even during the call to 'schedule'.
      }

      return f ;
    }

  template<typename FunctorType, typename ValueType, typename Scheduler>
  KOKKOS_FUNCTION static
  void
  respawn( FunctorType         * arg_self
         , BasicFuture<ValueType,Scheduler> const & arg_dependence
         , TaskPriority  const & arg_priority
         )
    {
      // Precondition: task is in Executing state

      using value_type  = typename FunctorType::value_type ;
      using task_type = Impl::Task<BasicTaskScheduler, value_type, FunctorType>;

      task_type * const task = static_cast< task_type * >( arg_self );

      task->m_priority = static_cast<int>(arg_priority);

      task->add_dependence( arg_dependence.m_task );

      // Postcondition: task is in Executing-Respawn state
    }

  template< typename FunctorType >
  KOKKOS_FUNCTION static
  void
  respawn( FunctorType         * arg_self
         , BasicTaskScheduler const &
         , TaskPriority  const & arg_priority
         )
    {
      // Precondition: task is in Executing state

      using value_type  = typename FunctorType::value_type ;
      using task_type = Impl::Task<BasicTaskScheduler, value_type, FunctorType>;

      task_type * const task = static_cast< task_type * >( arg_self );

      task->m_priority = static_cast<int>(arg_priority);

      task->add_dependence( (task_base*) 0 );

      // Postcondition: task is in Executing-Respawn state
    }

  //----------------------------------------
  /**\brief  Return a future that is complete
   *         when all input futures are complete.
   */
  template<typename ValueType>
  KOKKOS_FUNCTION static
  BasicFuture< void, scheduler_type >
  when_all(BasicFuture<ValueType, BasicTaskScheduler> const arg[], int narg)
    {
      using future_type = BasicFuture< void, scheduler_type > ;

      future_type f ;

      if ( narg ) {

        queue_type * queue = 0 ;

        BasicTaskScheduler const* scheduler_ptr = nullptr;

        for ( int i = 0 ; i < narg ; ++i ) {
          task_base * const t = arg[i].m_task ;
          if ( nullptr != t ) {
            // Increment reference count to track subsequent assignment.
            Kokkos::atomic_increment( &(t->m_ref_count) );
            if ( queue == 0 ) {
              scheduler_ptr = static_cast< BasicTaskScheduler const* >( t->m_scheduler );
              queue = scheduler_ptr->m_queue;
            }
            else if ( queue != static_cast< BasicTaskScheduler const* >(t->m_scheduler)->m_queue ) {
              Kokkos::abort("Kokkos when_all Futures must be in the same scheduler" );
            }
          }
        }

        if ( queue != 0 ) {

          size_t const alloc_size = queue->when_all_allocation_size( narg );

          f.m_task =
            reinterpret_cast< task_base * >( queue->allocate( alloc_size ) );

          if ( f.m_task ) {

            // Reference count starts at two:
            // +1 to match decrement when task completes
            // +1 for the future

            new( f.m_task ) task_base();

            f.m_task->m_scheduler = scheduler_ptr;
            f.m_task->m_ref_count = 2 ;
            f.m_task->m_alloc_size = static_cast<int32_t>(alloc_size);
            f.m_task->m_dep_count = narg ;
            f.m_task->m_task_type = task_base::Aggregate ;

            // Assign dependences, reference counts were already incremented

            task_base * volatile * const dep =
              f.m_task->aggregate_dependences();

            for ( int i = 0 ; i < narg ; ++i ) { dep[i] = arg[i].m_task ; }

            Kokkos::memory_fence();

            queue->schedule_aggregate( f.m_task );
            // this when_all may be processed at any moment
          }
        }
      }

      return f ;
    }

  template < class F >
  KOKKOS_FUNCTION
  BasicFuture< void, scheduler_type >
  when_all( int narg , F const func )
    {
      using input_type  = decltype( func(0) );
      using future_type = BasicFuture< void, scheduler_type > ;

      static_assert( is_future< input_type >::value
                   , "Functor must return a Kokkos::Future" );

      future_type f ;

      if ( 0 == narg ) return f ;

      size_t const alloc_size = m_queue->when_all_allocation_size( narg );

      f.m_task =
        reinterpret_cast< task_base * >( m_queue->allocate( alloc_size ) );

      if ( f.m_task ) {

        // Reference count starts at two:
        // +1 to match decrement when task completes
        // +1 for the future

        new( f.m_task ) task_base();

        f.m_task->m_scheduler = this;
        f.m_task->m_ref_count = 2 ;
        f.m_task->m_alloc_size = static_cast<int32_t>(alloc_size);
        f.m_task->m_dep_count = narg ;
        f.m_task->m_task_type = task_base::Aggregate ;

        // Assign dependences, reference counts were already incremented

        task_base * volatile * const dep =
          f.m_task->aggregate_dependences();

        for ( int i = 0 ; i < narg ; ++i ) {
          const input_type arg_f = func(i);
          if ( 0 != arg_f.m_task ) {

            // Not scheduled, so task scheduler is not yet set
            //if ( m_queue != static_cast< BasicTaskScheduler const * >( arg_f.m_task->m_scheduler )->m_queue ) {
            //  Kokkos::abort("Kokkos when_all Futures must be in the same scheduler" );
            //}
            // Increment reference count to track subsequent assignment.
            Kokkos::atomic_increment( &(arg_f.m_task->m_ref_count) );
            dep[i] = arg_f.m_task ;
          }
        }

        Kokkos::memory_fence();

        m_queue->schedule_aggregate( f.m_task );
        // this when_all may be processed at any moment
      }
      return f ;
    }

  //----------------------------------------

  KOKKOS_INLINE_FUNCTION
  int allocation_capacity() const noexcept
    { return m_queue->m_memory.capacity(); }

  KOKKOS_INLINE_FUNCTION
  int allocated_task_count() const noexcept
    { return m_queue->m_count_alloc ; }

  KOKKOS_INLINE_FUNCTION
  int allocated_task_count_max() const noexcept
    { return m_queue->m_max_alloc ; }

  KOKKOS_INLINE_FUNCTION
  long allocated_task_count_accum() const noexcept
    { return m_queue->m_accum_alloc ; }

  //----------------------------------------

  template<class S, class Q>
  friend
  void wait(Kokkos::BasicTaskScheduler<S, Q> const&);

};

} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

//----------------------------------------------------------------------------
// Construct a TaskTeam execution policy

template< typename T >
Kokkos::Impl::TaskPolicyData
  < Kokkos::Impl::TaskBase::TaskTeam
  , typename std::conditional< Kokkos::is_future< T >::value , T ,
    typename Kokkos::BasicFuture< void, typename T::scheduler_type > >::type
  >
KOKKOS_INLINE_FUNCTION
TaskTeam( T            const & arg
        , TaskPriority const & arg_priority = TaskPriority::Regular
        )
{
  static_assert( Kokkos::is_future<T>::value ||
                 Kokkos::is_scheduler<T>::value
               , "Kokkos TaskTeam argument must be Future or BasicTaskScheduler" );

  return
    Kokkos::Impl::TaskPolicyData
      < Kokkos::Impl::TaskBase::TaskTeam
      , typename std::conditional< Kokkos::is_future< T >::value , T ,
        typename Kokkos::BasicFuture< void, typename T::scheduler_type > >::type
      >( arg , arg_priority );
}

template<typename E, typename Q, typename F>
Kokkos::Impl::
  TaskPolicyData< Kokkos::Impl::TaskBase::TaskTeam , F >
KOKKOS_INLINE_FUNCTION
TaskTeam( BasicTaskScheduler<E, Q> const & arg_scheduler
        , F                const & arg_future
        , typename std::enable_if< Kokkos::is_future<F>::value ,
            TaskPriority >::type const & arg_priority = TaskPriority::Regular
        )
{
  return
    Kokkos::Impl::TaskPolicyData
      < Kokkos::Impl::TaskBase::TaskTeam , F >
        ( arg_scheduler , arg_future , arg_priority );
}

// Construct a TaskSingle execution policy

template< typename T >
Kokkos::Impl::TaskPolicyData
  < Kokkos::Impl::TaskBase::TaskSingle
  , typename std::conditional< Kokkos::is_future< T >::value , T ,
    typename Kokkos::BasicFuture< void, typename T::scheduler_type > >::type
  >
KOKKOS_INLINE_FUNCTION
TaskSingle( T            const & arg
          , TaskPriority const & arg_priority = TaskPriority::Regular
          )
{
  static_assert( Kokkos::is_future<T>::value ||
                 Kokkos::is_scheduler<T>::value
               , "Kokkos TaskSingle argument must be Future or Scheduler" );

  return
    Kokkos::Impl::TaskPolicyData
      < Kokkos::Impl::TaskBase::TaskSingle
      , typename std::conditional< Kokkos::is_future< T >::value , T ,
        typename Kokkos::BasicFuture< void, typename T::scheduler_type > >::type
      >( arg , arg_priority );
}

template <typename E, typename Q, typename F>
Kokkos::Impl::
  TaskPolicyData< Kokkos::Impl::TaskBase::TaskSingle , F >
KOKKOS_INLINE_FUNCTION
TaskSingle( BasicTaskScheduler<E, Q> const & arg_scheduler
          , F                const & arg_future
          , typename std::enable_if< Kokkos::is_future<F>::value ,
              TaskPriority >::type const & arg_priority = TaskPriority::Regular
          )
{
  return
    Kokkos::Impl::TaskPolicyData
      < Kokkos::Impl::TaskBase::TaskSingle , F >
        ( arg_scheduler , arg_future , arg_priority );
}

//----------------------------------------------------------------------------

/**\brief  A host control thread spawns a task with options
 *
 *  1) Team or Serial
 *  2) With scheduler or dependence
 *  3) High, Normal, or Low priority
 */
template< int TaskEnum
        , typename DepFutureType
        , typename FunctorType >
Kokkos::BasicFuture< typename FunctorType::value_type, typename DepFutureType::scheduler_type >
host_spawn( Impl::TaskPolicyData<TaskEnum,DepFutureType> const & arg_policy
          , FunctorType                                       && arg_functor
          )
{
  using exec_space = typename DepFutureType::execution_space ;
  using queue_type = typename DepFutureType::queue_type;
  using scheduler_type = BasicTaskScheduler<exec_space, queue_type>;

  using task_type =
    Impl::Task<scheduler_type, typename FunctorType::value_type, FunctorType>;

  static_assert( TaskEnum == task_type::TaskTeam ||
                 TaskEnum == task_type::TaskSingle
               , "Kokkos host_spawn requires TaskTeam or TaskSingle" );

  // May be spawning a Cuda task, must use the specialization
  // to query on-device function pointer.
  typename task_type::function_type ptr;
  typename task_type::destroy_type dtor;
  Kokkos::Impl::TaskQueueSpecialization< scheduler_type >::
    template get_function_pointer< task_type >(ptr, dtor);

  return scheduler_type::spawn( arg_policy , ptr , dtor, std::move(arg_functor) );
}

/**\brief  A task spawns a task with options
 *
 *  1) Team or Serial
 *  2) With scheduler or dependence
 *  3) High, Normal, or Low priority
 */
template< int TaskEnum
        , typename DepFutureType
        , typename FunctorType >
Kokkos::BasicFuture< typename FunctorType::value_type
      , typename DepFutureType::scheduler_type >
KOKKOS_INLINE_FUNCTION
task_spawn( Impl::TaskPolicyData<TaskEnum,DepFutureType> const & arg_policy
          , FunctorType                                       && arg_functor
          )
{
  using exec_space = typename DepFutureType::execution_space ;
  using queue_type = typename DepFutureType::queue_type;
  using scheduler_type = BasicTaskScheduler<exec_space, queue_type>;

  using task_type =
    Impl::Task<scheduler_type, typename FunctorType::value_type, FunctorType>;

  // TODO figure out why this is here...
// #if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST ) && \
//     defined( KOKKOS_ENABLE_CUDA )

//   static_assert( ! std::is_same< Kokkos::Cuda , exec_space >::value
//                , "Error calling Kokkos::task_spawn for Cuda space within Host code" );

// #endif

  static_assert( TaskEnum == task_type::TaskTeam ||
                 TaskEnum == task_type::TaskSingle
               , "Kokkos host_spawn requires TaskTeam or TaskSingle" );

  typename task_type::function_type const ptr = task_type::apply ;
  typename task_type::destroy_type const dtor = task_type::destroy ;

  return scheduler_type::spawn( arg_policy , ptr , dtor, std::move(arg_functor) );
}

/**\brief  A task respawns itself with options
 *
 *  1) With scheduler or dependence
 *  2) High, Normal, or Low priority
 */
template< typename FunctorType , typename T >
void
KOKKOS_INLINE_FUNCTION
respawn( FunctorType         * arg_self
       , T             const & arg
       , TaskPriority  const & arg_priority = TaskPriority::Regular
       )
{
  static_assert( Kokkos::is_future<T>::value ||
                 Kokkos::is_scheduler<T>::value
               , "Kokkos respawn argument must be Future or TaskScheduler" );

  BasicTaskScheduler<typename T::execution_space, typename T::queue_type>::respawn(
    arg_self , arg , arg_priority
  );
}

//----------------------------------------------------------------------------

template<typename ValueType, typename Scheduler>
KOKKOS_INLINE_FUNCTION
BasicFuture<void, Scheduler>
when_all(BasicFuture<ValueType, Scheduler> const arg[], int narg)
{
  return BasicFuture<void, Scheduler>::scheduler_type::when_all(arg, narg);
}

//----------------------------------------------------------------------------
// Wait for all runnable tasks to complete

template<class ExecSpace, class QueueType>
inline
void wait(BasicTaskScheduler<ExecSpace, QueueType> const& scheduler)
{
  using scheduler_type = BasicTaskScheduler<ExecSpace, QueueType>;
  scheduler_type::specialization::execute(scheduler);
  //scheduler.m_queue->execute();
}

} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------


#endif /* #if defined( KOKKOS_ENABLE_TASKDAG ) */
#endif /* #ifndef KOKKOS_TASKSCHEDULER_HPP */

