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


namespace Kokkos {

template<class Scalar>
struct Max {
  typedef Scalar value_type;
  value_type min_value;

  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) {
    dest = dest<src?dest:src;
  }

  KOKKOS_INLINE_FUNCTION
  void join(volatile value_type& dest, const volatile value_type& src) {
    dest = dest<src?dest:src;
  }

  KOKKOS_INLINE_FUNCTION
  void init( value_type& val) {
    val = min_value;
  }
};

}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

/** \brief  Parallel reduction
 *
 * Example of a parallel_reduce functor for a POD (plain old data) value type:
 * \code
 *  class FunctorType { // For POD value type
 *  public:
 *    typedef    ...     execution_space ;
 *    typedef <podType>  value_type ;
 *    void operator()( <intType> iwork , <podType> & update ) const ;
 *    void init( <podType> & update ) const ;
 *    void join( volatile       <podType> & update ,
 *               volatile const <podType> & input ) const ;
 *
 *    typedef true_type has_final ;
 *    void final( <podType> & update ) const ;
 *  };
 * \endcode
 *
 * Example of a parallel_reduce functor for an array of POD (plain old data) values:
 * \code
 *  class FunctorType { // For array of POD value
 *  public:
 *    typedef    ...     execution_space ;
 *    typedef <podType>  value_type[] ;
 *    void operator()( <intType> , <podType> update[] ) const ;
 *    void init( <podType> update[] ) const ;
 *    void join( volatile       <podType> update[] ,
 *               volatile const <podType> input[] ) const ;
 *
 *    typedef true_type has_final ;
 *    void final( <podType> update[] ) const ;
 *  };
 * \endcode
 */
namespace Kokkos {
namespace Impl {

template< class T, class ReturnType , class ValueTraits>
struct ParallelReduceReturnValue;

template< class ReturnType , class FunctorType >
struct ParallelReduceReturnValue<typename std::enable_if<Kokkos::is_view<ReturnType>::value>::type, ReturnType, FunctorType> {
  typedef ReturnType return_type;
  static return_type& return_value(ReturnType& return_val, const FunctorType&) {
    return return_val;
  }
};

template< class ReturnType , class FunctorType>
struct ParallelReduceReturnValue<typename std::enable_if<!Kokkos::is_view<ReturnType>::value &&
                                 (!std::is_array<ReturnType>::value && !std::is_pointer<ReturnType>::value)>::type,
                                 ReturnType, FunctorType> {
  typedef Kokkos::View<  ReturnType
                       , Kokkos::HostSpace
                       , Kokkos::MemoryUnmanaged
      > return_type;

  static return_type return_value(ReturnType& return_val, const FunctorType&) {
    return return_type(&return_val);
  }
};

template< class ReturnType , class FunctorType>
struct ParallelReduceReturnValue<typename std::enable_if<!Kokkos::is_view<ReturnType>::value &&
                                 (is_array<ReturnType>::value || std::is_pointer<ReturnType>::value)>::type, ReturnType, FunctorType> {
  typedef Kokkos::View<  ReturnType
                       , Kokkos::HostSpace
                       , Kokkos::MemoryUnmanaged
      > return_type;

  static return_type return_value(ReturnType& return_val,
                                  const FunctorType& functor) {
    return return_type(return_val,functor.value_count);
  }
};
}

namespace Impl {
template< class T, class ReturnType , class FunctorType>
struct ParallelReducePolicyType;

template< class PolicyType , class FunctorType >
struct ParallelReducePolicyType<typename std::enable_if<Kokkos::Impl::is_execution_policy<PolicyType>::value>::type, PolicyType,FunctorType> {

  typedef PolicyType policy_type;
  static PolicyType policy(const PolicyType& policy_) {
    return policy_;
  }
};

template< class PolicyType , class FunctorType >
struct ParallelReducePolicyType<typename std::enable_if<std::is_integral<PolicyType>::value>::type, PolicyType,FunctorType> {
  typedef typename
    Impl::FunctorPolicyExecutionSpace< FunctorType , void >::execution_space
      execution_space ;

  typedef Kokkos::RangePolicy<execution_space> policy_type;

  static policy_type policy(const PolicyType& policy_) {
    return policy_type(0,policy_);
  }
};

}

namespace Impl {
  template< class T, class ReturnType , class FunctorType>
  struct ParallelReducePolicyType;
}
template< class LabelType, class PolicyType, class FunctorType, class ReturnType >
inline
void parallel_reduce_new(const LabelType& label,
                         const PolicyType& policy,
                         const FunctorType& functor,
                         ReturnType& return_value,
                         typename Impl::enable_if<
                           (Kokkos::Impl::is_label<LabelType>::value &&
                            Kokkos::Impl::is_execution_policy<
                              typename Impl::ParallelReducePolicyType<void,PolicyType,FunctorType>::policy_type
                            >::value)
                         >::type * = 0) {
  std::cout << label << std::endl;

  typedef typename Impl::ParallelReducePolicyType<void,PolicyType,FunctorType>::policy_type policy_type;
  typedef Impl::ParallelReduceReturnValue<void,ReturnType,FunctorType> return_value_adapter;

  auto return_view = return_value_adapter::return_value(return_value,functor);

  Kokkos::Impl::shared_allocation_tracking_claim_and_disable();
  Impl::ParallelReduce<FunctorType, policy_type >
     closure(functor,
             Impl::ParallelReducePolicyType<void,PolicyType,FunctorType>::policy(policy),
             return_view);
  Kokkos::Impl::shared_allocation_tracking_release_and_enable();
  closure.execute();
}

template< class PolicyType, class FunctorType, class ReturnType >
inline
void parallel_reduce_new(const PolicyType& policy,
                         const FunctorType& functor,
                         ReturnType& return_value,
                         typename Impl::enable_if<
                         Kokkos::Impl::is_execution_policy<
                           typename Impl::ParallelReducePolicyType<void,PolicyType,FunctorType>::policy_type>::value
                         >::type * = 0) {
  std::cout << "No Label" <<std::endl;

  typedef typename Impl::ParallelReducePolicyType<void,PolicyType,FunctorType>::policy_type policy_type;
  typedef Impl::ParallelReduceReturnValue<void,ReturnType,FunctorType> return_value_adapter;

  auto return_view = return_value_adapter::return_value(return_value,functor);

  Kokkos::Impl::shared_allocation_tracking_claim_and_disable();
  Impl::ParallelReduce<FunctorType, policy_type>
     closure(functor,
             Impl::ParallelReducePolicyType<void,PolicyType,FunctorType>::policy(policy),
             return_view);
  Kokkos::Impl::shared_allocation_tracking_release_and_enable();
  closure.execute();
}

template< class ExecPolicy , class FunctorType >
inline
void parallel_reduce( const ExecPolicy  & policy
                    , const FunctorType & functor
                    , const std::string& str = ""
                    , typename Impl::enable_if< ! Impl::is_integral< ExecPolicy >::value >::type * = 0
                    )
{
  // typedef typename
  //   Impl::FunctorPolicyExecutionSpace< FunctorType , ExecPolicy >::execution_space
  //     execution_space ;

  typedef Kokkos::Impl::FunctorValueTraits< FunctorType , typename ExecPolicy::work_tag >  ValueTraits ;

  typedef typename Kokkos::Impl::if_c< (ValueTraits::StaticValueSize != 0)
                                     , typename ValueTraits::value_type
                                     , typename ValueTraits::pointer_type
                                     >::type value_type ;

  Kokkos::View< value_type
              , HostSpace
              , Kokkos::MemoryUnmanaged
              >
    result_view ;

#if (KOKKOS_ENABLE_PROFILING)
  uint64_t kpID = 0;
     if(Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::beginParallelReduce("" == str ? typeid(FunctorType).name() : str, 0, &kpID);
     }
#endif

    Kokkos::Impl::shared_allocation_tracking_claim_and_disable();
    Impl::ParallelReduce< FunctorType , ExecPolicy > closure( functor , policy , result_view );
    Kokkos::Impl::shared_allocation_tracking_release_and_enable();

    closure.execute();

#if (KOKKOS_ENABLE_PROFILING)
     if(Kokkos::Profiling::profileLibraryLoaded()) {
  Kokkos::Profiling::endParallelReduce(kpID);
     }
#endif
}

// integral range policy
template< class FunctorType >
inline
void parallel_reduce( const size_t        work_count
                    , const FunctorType & functor
                    , const std::string& str = ""
                    )
{
  typedef typename
    Impl::FunctorPolicyExecutionSpace< FunctorType , void >::execution_space
      execution_space ;

  typedef RangePolicy< execution_space > policy ;

  typedef Kokkos::Impl::FunctorValueTraits< FunctorType , void >  ValueTraits ;

  typedef typename Kokkos::Impl::if_c< (ValueTraits::StaticValueSize != 0)
                                     , typename ValueTraits::value_type
                                     , typename ValueTraits::pointer_type
                                     >::type value_type ;

  Kokkos::View< value_type
              , HostSpace
              , Kokkos::MemoryUnmanaged
              >
    result_view ;

#if (KOKKOS_ENABLE_PROFILING)
  uint64_t kpID = 0;
     if(Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::beginParallelReduce("" == str ? typeid(FunctorType).name() : str, 0, &kpID);
     }
#endif

  Kokkos::Impl::shared_allocation_tracking_claim_and_disable();
  Impl::ParallelReduce< FunctorType , policy > closure( functor , policy(0,work_count) , result_view );
  Kokkos::Impl::shared_allocation_tracking_release_and_enable();

  closure.execute();

#if (KOKKOS_ENABLE_PROFILING)
     if(Kokkos::Profiling::profileLibraryLoaded()) {
  Kokkos::Profiling::endParallelReduce(kpID);
     }
#endif

}

// general policy and view ouput
template< class ExecPolicy , class FunctorType , class ViewType >
inline
void parallel_reduce( const ExecPolicy  & policy
                    , const FunctorType & functor
                    , const ViewType    & result_view
                    , const std::string& str = ""
                    , typename Impl::enable_if<
                      ( Kokkos::is_view<ViewType>::value && ! Impl::is_integral< ExecPolicy >::value
#ifdef KOKKOS_HAVE_CUDA
                        && ! Impl::is_same<typename ExecPolicy::execution_space,Kokkos::Cuda>::value
#endif
                      )>::type * = 0 )
{

#if (KOKKOS_ENABLE_PROFILING)
  uint64_t kpID = 0;
     if(Kokkos::Profiling::profileLibraryLoaded()) {
  Kokkos::Profiling::beginParallelReduce("" == str ? typeid(FunctorType).name() : str, 0, &kpID);
     }
#endif

  Kokkos::Impl::shared_allocation_tracking_claim_and_disable();
  Impl::ParallelReduce< FunctorType, ExecPolicy > closure( functor , policy , result_view );
  Kokkos::Impl::shared_allocation_tracking_release_and_enable();

  closure.execute();

#if (KOKKOS_ENABLE_PROFILING)
     if(Kokkos::Profiling::profileLibraryLoaded()) {
  Kokkos::Profiling::endParallelReduce(kpID);
     }
#endif

}

// general policy and pod or array of pod output
template< class ExecPolicy , class FunctorType >
void parallel_reduce( const ExecPolicy  & policy
                    , const FunctorType & functor
#ifdef KOKKOS_HAVE_CUDA
                    , typename Impl::enable_if<
                      ( ! Impl::is_integral< ExecPolicy >::value &&
                        ! Impl::is_same<typename ExecPolicy::execution_space,Kokkos::Cuda>::value )
                      , typename Kokkos::Impl::FunctorValueTraits< FunctorType , typename ExecPolicy::work_tag >::reference_type>::type result_ref
                      , const std::string& str = ""
                      , typename Impl::enable_if<! Impl::is_same<typename ExecPolicy::execution_space,Kokkos::Cuda>::value >::type* = 0
                      )
#else
                      , typename Impl::enable_if<
                        ( ! Impl::is_integral< ExecPolicy >::value)
                        , typename Kokkos::Impl::FunctorValueTraits< FunctorType , typename ExecPolicy::work_tag >::reference_type
                        >::type result_ref
                      , const std::string& str = ""
                        )
#endif
{
  typedef Kokkos::Impl::FunctorValueTraits< FunctorType , typename ExecPolicy::work_tag >  ValueTraits ;
  typedef Kokkos::Impl::FunctorValueOps<    FunctorType , typename ExecPolicy::work_tag >  ValueOps ;

  // Wrap the result output request in a view to inform the implementation
  // of the type and memory space.

  typedef typename Kokkos::Impl::if_c< (ValueTraits::StaticValueSize != 0)
                                     , typename ValueTraits::value_type
                                     , typename ValueTraits::pointer_type
                                     >::type value_type ;

  Kokkos::View< value_type
              , HostSpace
              , Kokkos::MemoryUnmanaged
              >
    result_view( ValueOps::pointer( result_ref )
               , ValueTraits::value_count( functor )
               );

#if (KOKKOS_ENABLE_PROFILING)
  uint64_t kpID = 0;
     if(Kokkos::Profiling::profileLibraryLoaded()) {
  Kokkos::Profiling::beginParallelReduce("" == str ? typeid(FunctorType).name() : str, 0, &kpID);
     }
#endif

  Kokkos::Impl::shared_allocation_tracking_claim_and_disable();
  Impl::ParallelReduce< FunctorType, ExecPolicy > closure( functor , policy , result_view );
  Kokkos::Impl::shared_allocation_tracking_release_and_enable();

  closure.execute();

#if (KOKKOS_ENABLE_PROFILING)
     if(Kokkos::Profiling::profileLibraryLoaded()) {
  Kokkos::Profiling::endParallelReduce(kpID);
     }
#endif

}

// integral range policy and view ouput
template< class FunctorType , class ViewType >
inline
void parallel_reduce( const size_t        work_count
                    , const FunctorType & functor
                    , const ViewType    & result_view
                    , const std::string& str = ""
                    , typename Impl::enable_if<( Kokkos::is_view<ViewType>::value
#ifdef KOKKOS_HAVE_CUDA
                        && ! Impl::is_same<
                          typename Impl::FunctorPolicyExecutionSpace< FunctorType , void >::execution_space,
                          Kokkos::Cuda>::value
#endif
                        )>::type * = 0 )
{
  typedef typename
    Impl::FunctorPolicyExecutionSpace< FunctorType , void >::execution_space
      execution_space ;

  typedef RangePolicy< execution_space > ExecPolicy ;

#if (KOKKOS_ENABLE_PROFILING)
  uint64_t kpID = 0;
     if(Kokkos::Profiling::profileLibraryLoaded()) {
  Kokkos::Profiling::beginParallelReduce("" == str ? typeid(FunctorType).name() : str, 0, &kpID);
     }
#endif

  Kokkos::Impl::shared_allocation_tracking_claim_and_disable();
  Impl::ParallelReduce< FunctorType, ExecPolicy > closure( functor , ExecPolicy(0,work_count) , result_view );
  Kokkos::Impl::shared_allocation_tracking_release_and_enable();

  closure.execute();

#if (KOKKOS_ENABLE_PROFILING)
     if(Kokkos::Profiling::profileLibraryLoaded()) {
  Kokkos::Profiling::endParallelReduce(kpID);
     }
#endif

}

// integral range policy and pod or array of pod output
template< class FunctorType >
inline
void parallel_reduce( const size_t        work_count
                    , const FunctorType & functor
                    , typename Kokkos::Impl::FunctorValueTraits<
                         typename Impl::if_c<Impl::is_execution_policy<FunctorType>::value ||
                                             Impl::is_integral<FunctorType>::value,
                            void,FunctorType>::type
                         , void >::reference_type result
                    , const std::string& str = ""
                    , typename Impl::enable_if< true
#ifdef KOKKOS_HAVE_CUDA
                              && ! Impl::is_same<
                             typename Impl::FunctorPolicyExecutionSpace< FunctorType , void >::execution_space,
                             Kokkos::Cuda>::value
#endif
                     >::type * = 0 )
{
  typedef Kokkos::Impl::FunctorValueTraits< FunctorType , void >  ValueTraits ;
  typedef Kokkos::Impl::FunctorValueOps<    FunctorType , void >  ValueOps ;

  typedef typename
    Kokkos::Impl::FunctorPolicyExecutionSpace< FunctorType , void >::execution_space
      execution_space ;

  typedef Kokkos::RangePolicy< execution_space > policy ;

  // Wrap the result output request in a view to inform the implementation
  // of the type and memory space.

  typedef typename Kokkos::Impl::if_c< (ValueTraits::StaticValueSize != 0)
                                     , typename ValueTraits::value_type
                                     , typename ValueTraits::pointer_type
                                     >::type value_type ;

  Kokkos::View< value_type
              , HostSpace
              , Kokkos::MemoryUnmanaged
              >
    result_view( ValueOps::pointer( result )
               , ValueTraits::value_count( functor )
               );

#if (KOKKOS_ENABLE_PROFILING)
  uint64_t kpID = 0;
     if(Kokkos::Profiling::profileLibraryLoaded()) {
  Kokkos::Profiling::beginParallelReduce("" == str ? typeid(FunctorType).name() : str, 0, &kpID);
     }
#endif

  Kokkos::Impl::shared_allocation_tracking_claim_and_disable();
  Impl::ParallelReduce< FunctorType , policy > closure( functor , policy(0,work_count) , result_view );
  Kokkos::Impl::shared_allocation_tracking_release_and_enable();

  closure.execute();

#if (KOKKOS_ENABLE_PROFILING)
     if(Kokkos::Profiling::profileLibraryLoaded()) {
  Kokkos::Profiling::endParallelReduce(kpID);
     }
#endif

}
#ifndef KOKKOS_HAVE_CUDA
template< class ExecPolicy , class FunctorType , class ResultType >
inline
void parallel_reduce( const std::string & str
                    , const ExecPolicy  & policy
                    , const FunctorType & functor
                    , ResultType * result)
{
  #if KOKKOS_ENABLE_DEBUG_PRINT_KERNEL_NAMES
  Kokkos::fence();
  std::cout << "KOKKOS_DEBUG Start parallel_reduce kernel: " << str << std::endl;
  #endif

  parallel_reduce(policy,functor,result,str);

  #if KOKKOS_ENABLE_DEBUG_PRINT_KERNEL_NAMES
  Kokkos::fence();
  std::cout << "KOKKOS_DEBUG End   parallel_reduce kernel: " << str << std::endl;
  #endif
  (void) str;
}

template< class ExecPolicy , class FunctorType , class ResultType >
inline
void parallel_reduce( const std::string & str
                    , const ExecPolicy  & policy
                    , const FunctorType & functor
                    , ResultType & result)
{
  #if KOKKOS_ENABLE_DEBUG_PRINT_KERNEL_NAMES
  Kokkos::fence();
  std::cout << "KOKKOS_DEBUG Start parallel_reduce kernel: " << str << std::endl;
  #endif

  parallel_reduce(policy,functor,result,str);

  #if KOKKOS_ENABLE_DEBUG_PRINT_KERNEL_NAMES
  Kokkos::fence();
  std::cout << "KOKKOS_DEBUG End   parallel_reduce kernel: " << str << std::endl;
  #endif
  (void) str;
}

template< class ExecPolicy , class FunctorType >
inline
void parallel_reduce( const std::string & str
                    , const ExecPolicy  & policy
                    , const FunctorType & functor)
{
  #if KOKKOS_ENABLE_DEBUG_PRINT_KERNEL_NAMES
  Kokkos::fence();
  std::cout << "KOKKOS_DEBUG Start parallel_reduce kernel: " << str << std::endl;
  #endif

  parallel_reduce(policy,functor,str);

  #if KOKKOS_ENABLE_DEBUG_PRINT_KERNEL_NAMES
  Kokkos::fence();
  std::cout << "KOKKOS_DEBUG End   parallel_reduce kernel: " << str << std::endl;
  #endif
  (void) str;
}
#endif

} // namespace Kokkos
