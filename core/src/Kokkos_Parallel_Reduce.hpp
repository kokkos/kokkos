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


template<class T, class Enable = void>
struct is_reducer_type {
  enum { value = 0 };
};


template<class T>
struct is_reducer_type<T,typename std::enable_if<
                       std::is_same<T,typename T::reducer_type>::value
                      >::type> {
  enum { value = 1 };
};
}


namespace Kokkos {
namespace Impl {

template< class T, class ReturnType , class ValueTraits>
struct ParallelReduceReturnValue;

template< class ReturnType , class FunctorType >
struct ParallelReduceReturnValue<typename std::enable_if<Kokkos::is_view<ReturnType>::value>::type, ReturnType, FunctorType> {
  typedef ReturnType return_type;
  typedef void* reducer_type;

  typedef typename return_type::value_type value_type_scalar;
  typedef typename return_type::value_type value_type_array[];

  typedef typename if_c<return_type::rank==0,value_type_scalar,value_type_array>::type value_type;

  static return_type& return_value(ReturnType& return_val, const FunctorType&) {
    return return_val;
  }
};

template< class ReturnType , class FunctorType>
struct ParallelReduceReturnValue<typename std::enable_if<
                                   !Kokkos::is_view<ReturnType>::value &&
                                  (!std::is_array<ReturnType>::value && !std::is_pointer<ReturnType>::value) &&
                                   !Kokkos::is_reducer_type<ReturnType>::value
                                 >::type, ReturnType, FunctorType> {
  typedef Kokkos::View<  ReturnType
                       , Kokkos::HostSpace
                       , Kokkos::MemoryUnmanaged
      > return_type;

  typedef void* reducer_type;

  typedef typename return_type::value_type value_type;

  static return_type return_value(ReturnType& return_val, const FunctorType&) {
    return return_type(&return_val);
  }
};

template< class ReturnType , class FunctorType>
struct ParallelReduceReturnValue<typename std::enable_if<
                                  (is_array<ReturnType>::value || std::is_pointer<ReturnType>::value)
                                >::type, ReturnType, FunctorType> {
  typedef Kokkos::View<  typename std::remove_const<ReturnType>::type
                       , Kokkos::HostSpace
                       , Kokkos::MemoryUnmanaged
      > return_type;

  typedef void* reducer_type;

  typedef typename return_type::value_type value_type[];

  static return_type return_value(ReturnType& return_val,
                                  const FunctorType& functor) {
    return return_type(return_val,functor.value_count);
  }
};

template< class ReturnType , class FunctorType>
struct ParallelReduceReturnValue<typename std::enable_if<
                                   Kokkos::is_reducer_type<ReturnType>::value
                                >::type, ReturnType, FunctorType> {
  typedef ReturnType return_type;
  typedef ReturnType reducer_type;
  typedef typename return_type::value_type value_type;

  static return_type return_value(ReturnType& return_val,
                                  const FunctorType& functor) {
    return return_val;
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
  template< class FunctorType, class ExecPolicy, class ValueType, class ExecutionSpace>
  struct ParallelReduceFunctorType {
    typedef FunctorType functor_type;
    static const functor_type& functor(const functor_type& functor) {
      return functor;
    }
  };
}

template<class Scalar>
struct Max {
  typedef Max reducer_type;
  typedef Scalar value_type;
  value_type min_value;
  value_type& result;

  Max(value_type& result_):result(result_) {}

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

template<class Scalar,class Space = HostSpace>
struct Add {
public:
  //Required
  typedef Add reducer_type;
  typedef Scalar value_type;

  typedef Kokkos::View<value_type, Space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > result_view_type;

private:
  result_view_type result;

public:

  Add(value_type& result_):result(&result_) {}

  //Required
  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src)  const {
    dest += src + 1;
  }

  KOKKOS_INLINE_FUNCTION
  void join(volatile value_type& dest, const volatile value_type& src) const {
    dest += src + 1;
  }

  //Optional
  KOKKOS_INLINE_FUNCTION
  void init( value_type& val)  const {
    val = value_type();
  }

  //Optional
  //KOKKOS_INLINE_FUNCTION
  //void final( value_type& val) {
  //}


  result_view_type result_view() const {
    return result;
  }
};

namespace Impl {

  template< class PolicyType, class FunctorType, class ReturnType >
  struct ParallelReduceAdaptor {
    typedef Impl::ParallelReduceReturnValue<void,ReturnType,FunctorType> return_value_adapter;
    typedef Impl::ParallelReduceFunctorType<FunctorType,PolicyType,
                                            typename return_value_adapter::value_type,
                                            typename PolicyType::execution_space> functor_adaptor;

    static inline
    void execute(const std::string& label,
        const PolicyType& policy,
        const FunctorType& functor,
        ReturnType& return_value) {
          #if (KOKKOS_ENABLE_PROFILING)
            uint64_t kpID = 0;
            if(Kokkos::Profiling::profileLibraryLoaded()) {
              Kokkos::Profiling::beginParallelReduce(std::string(label), 0, &kpID);
            }
          #endif

          Kokkos::Impl::shared_allocation_tracking_claim_and_disable();
          Impl::ParallelReduce<typename functor_adaptor::functor_type, PolicyType, typename return_value_adapter::reducer_type >
             closure(functor_adaptor::functor(functor),
                     policy,
                     return_value_adapter::return_value(return_value,functor));
          Kokkos::Impl::shared_allocation_tracking_release_and_enable();
          closure.execute();

          #if (KOKKOS_ENABLE_PROFILING)
            if(Kokkos::Profiling::profileLibraryLoaded()) {
              Kokkos::Profiling::endParallelReduce(kpID);
            }
          #endif
        }

  };
}

// ReturnValue is scalar or array: take by reference

template< class PolicyType, class FunctorType, class ReturnType >
inline
void parallel_reduce(const std::string& label,
                     const PolicyType& policy,
                     const FunctorType& functor,
                     ReturnType& return_value,
                     typename Impl::enable_if<
                       Kokkos::Impl::is_execution_policy<PolicyType>::value
                     >::type * = 0) {
  Impl::ParallelReduceAdaptor<PolicyType,FunctorType,ReturnType>::execute(label,policy,functor,return_value);
}

template< class PolicyType, class FunctorType, class ReturnType >
inline
void parallel_reduce(const PolicyType& policy,
                     const FunctorType& functor,
                     ReturnType& return_value,
                     typename Impl::enable_if<
                       Kokkos::Impl::is_execution_policy<PolicyType>::value
                     >::type * = 0) {
  Impl::ParallelReduceAdaptor<PolicyType,FunctorType,ReturnType>::execute("No Label",policy,functor,return_value);
}

template< class FunctorType, class ReturnType >
inline
void parallel_reduce(const size_t& policy,
                     const FunctorType& functor,
                     ReturnType& return_value) {
  typedef typename Impl::ParallelReducePolicyType<void,size_t,FunctorType>::policy_type policy_type;
  Impl::ParallelReduceAdaptor<policy_type,FunctorType,ReturnType>::execute("No Label",policy_type(0,policy),functor,return_value);
}

template< class FunctorType, class ReturnType >
inline
void parallel_reduce(const std::string& label,
                     const size_t& policy,
                     const FunctorType& functor,
                     ReturnType& return_value) {
  typedef typename Impl::ParallelReducePolicyType<void,size_t,FunctorType>::policy_type policy_type;
  Impl::ParallelReduceAdaptor<policy_type,FunctorType,ReturnType>::execute(label,policy_type(0,policy),functor,return_value);
}

// ReturnValue as View or Reducer: take by copy to allow for inline construction

template< class PolicyType, class FunctorType, class ReturnType >
inline
void parallel_reduce(const std::string& label,
                     const PolicyType& policy,
                     const FunctorType& functor,
                     const ReturnType& return_value,
                     typename Impl::enable_if<
                       Kokkos::Impl::is_execution_policy<PolicyType>::value
                     >::type * = 0) {
  Impl::ParallelReduceAdaptor<PolicyType,FunctorType,const ReturnType>::execute(label,policy,functor,return_value);
}

template< class PolicyType, class FunctorType, class ReturnType >
inline
void parallel_reduce(const PolicyType& policy,
                     const FunctorType& functor,
                     const ReturnType& return_value,
                     typename Impl::enable_if<
                       Kokkos::Impl::is_execution_policy<PolicyType>::value
                     >::type * = 0) {
  Impl::ParallelReduceAdaptor<PolicyType,FunctorType,const ReturnType>::execute("No Label",policy,functor,return_value);
}

template< class FunctorType, class ReturnType >
inline
void parallel_reduce(const size_t& policy,
                     const FunctorType& functor,
                     const ReturnType& return_value) {
  typedef typename Impl::ParallelReducePolicyType<void,size_t,FunctorType>::policy_type policy_type;

  Impl::ParallelReduceAdaptor<policy_type,FunctorType,const ReturnType>::execute("No Label",policy_type(0,policy),functor,return_value);
}

template< class FunctorType, class ReturnType >
inline
void parallel_reduce(const std::string& label,
                     const size_t& policy,
                     const FunctorType& functor,
                     const ReturnType& return_value) {
  typedef typename Impl::ParallelReducePolicyType<void,size_t,FunctorType>::policy_type policy_type;
  Impl::ParallelReduceAdaptor<policy_type,FunctorType,const ReturnType>::execute(label,policy_type(0,policy),functor,return_value);
}

// No Return Argument

template< class PolicyType, class FunctorType>
inline
void parallel_reduce(const std::string& label,
                     const PolicyType& policy,
                     const FunctorType& functor,
                     typename Impl::enable_if<
                       Kokkos::Impl::is_execution_policy<PolicyType>::value
                     >::type * = 0) {
  typedef Kokkos::Impl::FunctorValueTraits< FunctorType , void >  ValueTraits ;
  typedef typename Kokkos::Impl::if_c< (ValueTraits::StaticValueSize != 0)
                                     , typename ValueTraits::value_type
                                     , typename ValueTraits::pointer_type
                                     >::type value_type ;

  typedef Kokkos::View< value_type
              , Kokkos::HostSpace
              , Kokkos::MemoryUnmanaged
              > result_view_type;
  result_view_type result_view ;

  Impl::ParallelReduceAdaptor<PolicyType,FunctorType,result_view_type>::execute(label,policy,functor,result_view);
}

template< class PolicyType, class FunctorType >
inline
void parallel_reduce(const PolicyType& policy,
                     const FunctorType& functor,
                     typename Impl::enable_if<
                       Kokkos::Impl::is_execution_policy<PolicyType>::value
                     >::type * = 0) {
  typedef Kokkos::Impl::FunctorValueTraits< FunctorType , void >  ValueTraits ;
  typedef typename Kokkos::Impl::if_c< (ValueTraits::StaticValueSize != 0)
                                     , typename ValueTraits::value_type
                                     , typename ValueTraits::pointer_type
                                     >::type value_type ;

  typedef Kokkos::View< value_type
              , Kokkos::HostSpace
              , Kokkos::MemoryUnmanaged
              > result_view_type;
  result_view_type result_view ;

  Impl::ParallelReduceAdaptor<PolicyType,FunctorType,result_view_type>::execute("No Label",policy,functor,result_view);
}

template< class FunctorType >
inline
void parallel_reduce(const size_t& policy,
                     const FunctorType& functor) {
  typedef typename Impl::ParallelReducePolicyType<void,size_t,FunctorType>::policy_type policy_type;
  typedef Kokkos::Impl::FunctorValueTraits< FunctorType , void >  ValueTraits ;
  typedef typename Kokkos::Impl::if_c< (ValueTraits::StaticValueSize != 0)
                                     , typename ValueTraits::value_type
                                     , typename ValueTraits::pointer_type
                                     >::type value_type ;

  typedef Kokkos::View< value_type
              , Kokkos::HostSpace
              , Kokkos::MemoryUnmanaged
              > result_view_type;
  result_view_type result_view ;

  Impl::ParallelReduceAdaptor<policy_type,FunctorType,result_view_type>::execute("No Label",policy_type(0,policy),functor,result_view);
}

template< class FunctorType>
inline
void parallel_reduce(const std::string& label,
                     const size_t& policy,
                     const FunctorType& functor) {
  typedef typename Impl::ParallelReducePolicyType<void,size_t,FunctorType>::policy_type policy_type;
  typedef Kokkos::Impl::FunctorValueTraits< FunctorType , void >  ValueTraits ;
  typedef typename Kokkos::Impl::if_c< (ValueTraits::StaticValueSize != 0)
                                     , typename ValueTraits::value_type
                                     , typename ValueTraits::pointer_type
                                     >::type value_type ;

  typedef Kokkos::View< value_type
              , Kokkos::HostSpace
              , Kokkos::MemoryUnmanaged
              > result_view_type;
  result_view_type result_view ;

  Impl::ParallelReduceAdaptor<policy_type,FunctorType,result_view_type>::execute(label,policy_type(0,policy),functor,result_view);
}



} //namespace Kokkos
