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

#ifndef KOKKOS_IMPL_SERIAL_TASK_HPP
#define KOKKOS_IMPL_SERIAL_TASK_HPP

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

//----------------------------------------------------------------------------

template<>
class TaskQueueSpecialization< Kokkos::Serial >
{
public:

  using execution_space = Kokkos::Serial ;
  using memory_space    = Kokkos::HostSpace ;
  using queue_type      = Kokkos::Impl::TaskQueue< execution_space > ;
  using task_base_type  = Kokkos::Impl::TaskBase< execution_space , void , void > ;

  static
  void execute( queue_type * const );

  template< typename FunctorType >
  static
  void proc_set_apply( task_base_type::function_type * ptr )
    {
      using TaskType = TaskBase< Kokkos::Serial
                               , typename FunctorType::value_type
                               , FunctorType
                               > ;
       *ptr = TaskType::apply ;
    }
};

extern template class TaskQueue< Kokkos::Serial > ;

//----------------------------------------------------------------------------

template<>
class TaskExec< Kokkos::Serial >
{
public:

  KOKKOS_INLINE_FUNCTION void team_barrier() const {}
  KOKKOS_INLINE_FUNCTION int team_rank() const { return 0 ; }
  KOKKOS_INLINE_FUNCTION int team_size() const { return 1 ; }
};

template<typename iType>
struct TeamThreadRangeBoundariesStruct<iType, TaskExec< Kokkos::Serial > >
{
  typedef iType index_type;
  const iType begin ;
  const iType end ;
  enum {increment = 1};
  const  TaskExec< Kokkos::Serial > & thread;

  KOKKOS_INLINE_FUNCTION
  TeamThreadRangeBoundariesStruct
    ( const TaskExec< Kokkos::Serial > & arg_thread, const iType& arg_count)
    : begin(0)
    , end(arg_count)
    , thread(arg_thread)
    {}

  KOKKOS_INLINE_FUNCTION
  TeamThreadRangeBoundariesStruct
    ( const TaskExec< Kokkos::Serial > & arg_thread
    , const iType& arg_begin
    , const iType & arg_end
    )
    : begin( arg_begin )
    , end(   arg_end)
    , thread( arg_thread )
    {}
};

}} /* namespace Kokkos::Impl */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

template<typename iType>
KOKKOS_INLINE_FUNCTION
Impl::TeamThreadRangeBoundariesStruct<iType,Impl::TaskExec< Kokkos::Serial > >
TeamThreadRange( const Impl::TaskExec< Kokkos::Serial > & thread
               , const iType & count )
{
  return Impl::TeamThreadRangeBoundariesStruct<iType,Impl::TaskExec< Kokkos::Serial > >(thread,count);
}

template<typename iType>
KOKKOS_INLINE_FUNCTION
Impl::TeamThreadRangeBoundariesStruct<iType,Impl:: TaskExec< Kokkos::Serial > >
TeamThreadRange( const Impl:: TaskExec< Kokkos::Serial > & thread, const iType & begin , const iType & end )
{
  return Impl::TeamThreadRangeBoundariesStruct<iType,Impl:: TaskExec< Kokkos::Serial > >(thread,begin,end);
}

  /** \brief  Inter-thread parallel_for. Executes lambda(iType i) for each i=0..N-1.
   *
   * The range i=0..N-1 is mapped to all threads of the the calling thread team.
   * This functionality requires C++11 support.*/
template<typename iType, class Lambda>
KOKKOS_INLINE_FUNCTION
void parallel_for(const Impl::TeamThreadRangeBoundariesStruct<iType,Impl:: TaskExec< Kokkos::Serial > >& loop_boundaries, const Lambda& lambda) {
  for( iType i = loop_boundaries.begin; i < loop_boundaries.end; i+=loop_boundaries.increment)
    lambda(i);
}

} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #ifndef KOKKOS_IMPL_SERIAL_TASK_HPP */

