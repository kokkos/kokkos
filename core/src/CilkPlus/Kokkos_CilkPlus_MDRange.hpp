#ifndef KOKKOS_CILKPLUS_MDRANGE_HPP
#define KOKKOS_CILKPLUS_MDRANGE_HPP

#include<CilkPlus/Kokkos_CilkPlus_Reduce.hpp>

namespace Kokkos {
namespace Impl {


template< class FunctorType , class ... Traits >
class ParallelFor< FunctorType ,
                   Kokkos::MDRangePolicy< Traits ... > ,
                   Kokkos::Experimental::CilkPlus
                 >
{
private:

  typedef Kokkos::MDRangePolicy< Traits ... > MDRangePolicy ;
  typedef typename MDRangePolicy::impl_range_policy Policy ;

  typedef typename Kokkos::Impl::HostIterateTile< MDRangePolicy, FunctorType, typename MDRangePolicy::work_tag, void > iterate_type;

  const FunctorType   m_functor ;
  const MDRangePolicy m_mdr_policy ;
  const Policy        m_policy ;

  void
  exec() const
    {
      const typename Policy::member_type e = m_policy.end();
      cilk_for ( typename Policy::member_type i = m_policy.begin() ; i < e ; ++i ) {
        iterate_type( m_mdr_policy, m_functor )( (const typename Policy::member_type)i );
      }
    }

public:

  inline
  void execute() const
    { this->exec(); }

  inline
  ParallelFor( const FunctorType   & arg_functor
             , const MDRangePolicy & arg_policy )
    : m_functor( arg_functor )
    , m_mdr_policy(  arg_policy )
    , m_policy( Policy(0, m_mdr_policy.m_num_tiles).set_chunk_size(1) )
    {}
};

template< class FunctorType , class ReducerType , class ... Traits >
class ParallelReduce< FunctorType
                    , Kokkos::MDRangePolicy< Traits ... >
                    , ReducerType
                    , Kokkos::Experimental::CilkPlus
                    >
{
private:

  typedef Kokkos::MDRangePolicy< Traits ... > MDRangePolicy ;
  typedef typename MDRangePolicy::impl_range_policy Policy ;

  typedef typename MDRangePolicy::work_tag                                  WorkTag ;

  typedef Kokkos::Impl::if_c< std::is_same<InvalidType,ReducerType>::value, FunctorType, ReducerType> ReducerConditional;
  typedef typename ReducerConditional::type ReducerTypeFwd;
  typedef typename Kokkos::Impl::if_c< std::is_same<InvalidType,ReducerType>::value, WorkTag, void>::type WorkTagFwd;

  typedef typename ReducerTypeFwd::value_type ValueType;

  typedef FunctorAnalysis< FunctorPatternInterface::REDUCE , Policy , FunctorType > Analysis ;
  typedef Kokkos::Impl::kokkos_cilk_reducer< ReducerTypeFwd, FunctorType, typename Analysis::value_type, WorkTagFwd > cilk_reducer_wrapper;

  typedef Kokkos::Impl::FunctorValueInit<   ReducerTypeFwd , WorkTagFwd >  ValueInit ;

  typedef typename Analysis::pointer_type    pointer_type ;
  typedef typename Analysis::reference_type  reference_type ;


  using iterate_type = typename Kokkos::Impl::HostIterateTile< MDRangePolicy
                                                                           , FunctorType
                                                                           , WorkTag
                                                                           , ValueType
                                                                           >;


  const FunctorType   m_functor ;
  const MDRangePolicy m_mdr_policy ;
  const Policy        m_policy ;
  const ReducerType   m_reducer ;
  const pointer_type  m_result_ptr ;

  inline
  void
  exec( reference_type update, const size_t l_alloc_bytes ) const
    {
      Kokkos::HostSpace space;
      const typename Policy::member_type e = m_policy.end();
      cilk_reducer_wrapper cilk_reducer(ReducerConditional::select(m_functor , m_reducer), l_alloc_bytes);      
      INITIALIZE_CILK_REDUCER( cilk_reducer_wrapper, cilk_reducer )
      size_t working_set = l_alloc_bytes * (m_policy.end() - m_policy.begin());
      void * w_ptr = NULL; 
      if (working_set > 0) {      
//         t rintf("calling alloc: %d \n", working_set);
         w_ptr = space.allocate( working_set );
         memset( w_ptr, 0, working_set );
      }
      cilk_for ( typename Policy::member_type i = m_policy.begin() ; i < e ; ++i ) {
         if (w_ptr != NULL) {
            void * const l_ptr = (void*)(((char*)w_ptr)+((i - m_policy.begin()) * l_alloc_bytes));
            reference_type lupdate = ValueInit::init(  ReducerConditional::select(m_functor , m_reducer) , l_ptr );
            iterate_type( m_mdr_policy, m_functor, lupdate )( (const typename Policy::member_type)i );
            cilk_reducer.join( lupdate );
         } 
      }
      cilk_reducer.update_value( update );
      cilk_reducer.release_resources();
      global_reducer = NULL;
      if (w_ptr != NULL) {
          //printf("freeing memory: %d \n", working_set);
          space.deallocate(w_ptr, working_set);
      }
    }

public:

  inline
  void execute() const
    {
      const size_t pool_reduce_size =
        Analysis::value_size( ReducerConditional::select(m_functor , m_reducer) );
      const size_t team_reduce_size  = 0 ; // Never shrinks
      const size_t team_shared_size  = 0 ; // Never shrinks
      const size_t thread_local_size = 0 ; // Never shrinks

      serial_resize_thread_team_data( pool_reduce_size
                                    , team_reduce_size
                                    , team_shared_size
                                    , thread_local_size );

      HostThreadTeamData & data = *serial_get_thread_team_data();

      pointer_type ptr =
        m_result_ptr ? m_result_ptr : pointer_type(data.pool_reduce_local());

      reference_type update =
        ValueInit::init(  ReducerConditional::select(m_functor , m_reducer) , ptr );

      this-> exec( update, pool_reduce_size );

      Kokkos::Impl::FunctorFinal< ReducerTypeFwd , WorkTagFwd >::
        final(  ReducerConditional::select(m_functor , m_reducer) , ptr );
    }

  template< class HostViewType >
  ParallelReduce( const FunctorType  & arg_functor ,
                  const MDRangePolicy       & arg_policy ,
                  const HostViewType & arg_result_view ,
                  typename std::enable_if<
                               Kokkos::is_view< HostViewType >::value &&
                              !Kokkos::is_reducer_type<ReducerType>::value
                  ,void*>::type = NULL)
    : m_functor( arg_functor )
    , m_mdr_policy( arg_policy )
    , m_policy( Policy(0, m_mdr_policy.m_num_tiles).set_chunk_size(1) )
    , m_reducer( InvalidType() )
    , m_result_ptr( arg_result_view.data() )
    {
      static_assert( Kokkos::is_view< HostViewType >::value
        , "Kokkos::Experimental::CilkPlus reduce result must be a View" );

      static_assert( std::is_same< typename HostViewType::memory_space , HostSpace >::value
        , "Kokkos::Experimental::CilkPlus reduce result must be a View in HostSpace" );
    }

  inline
  ParallelReduce( const FunctorType & arg_functor
                , MDRangePolicy       arg_policy
                , const ReducerType& reducer )
    : m_functor( arg_functor )
    , m_mdr_policy(  arg_policy )
    , m_policy( Policy(0, m_mdr_policy.m_num_tiles).set_chunk_size(1) )
    , m_reducer( reducer )
    , m_result_ptr(  reducer.view().data() )
    {
    }
};
}
}

#endif
