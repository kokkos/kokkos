#ifndef __CILKPLUS_RANGE_
#define __CILKPLUS_RANGE_

#ifdef KOKKOS_ENABLE_EMU
   #include<CilkPlus/Kokkos_CilkEmu_Reduce.hpp>
#else
   #include<CilkPlus/Kokkos_CilkPlus_Reduce.hpp>
#endif

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/
/* Parallel patterns for Kokkos::Experimental::CilkPlus with RangePolicy */

namespace Kokkos {
namespace Impl {

template< class FunctorType , class ... Traits >
class ParallelFor< FunctorType ,
                   Kokkos::RangePolicy< Traits ... > ,
                   Kokkos::Experimental::CilkPlus
                 >
{
private:

  typedef Kokkos::RangePolicy< Traits ... > Policy ;

  const FunctorType m_functor ;
  const Policy      m_policy ;

  template< class TagType >
  typename std::enable_if< std::is_same< TagType , void >::value >::type
  exec() const
    {
      
      const typename Policy::member_type e = m_policy.end();
      const typename Policy::member_type b = m_policy.begin();
      const typename Policy::member_type len = e-b;
      const typename Policy::member_type par_loop = len > 16 ? 16 : len;
      typename Policy::member_type int_loop = 1;
      if ( par_loop > 0 )
          int_loop = (len / par_loop) + ( ( (len % par_loop) == 0) ? 0 : 1 );
//      printf(" parallel for: b= %d, e = %d, l = %d, par = %d, int = %d \n", b, e, len, par_loop, int_loop);
      cilk_for (typename Policy::member_type i = 0 ; i < par_loop ; ++i ) {
        for ( typename Policy::member_type j = (int_loop * i); j < ( (int_loop * i) + int_loop); j++ ) {
//           printf(" parallel for: i = %d, j = %d \n", (const int)i, j);
           if (j < e)
              m_functor( (const typename Policy::member_type)j );
        }
      }
    }

  template< class TagType >
  typename std::enable_if< ! std::is_same< TagType , void >::value >::type
  exec() const
    {
      const TagType t{} ;
      const typename Policy::member_type e = m_policy.end();
      const typename Policy::member_type b = m_policy.begin();
      const typename Policy::member_type len = e-b;
      const typename Policy::member_type par_loop = len > 16 ? 16 : len;
      typename Policy::member_type int_loop = 1;
      if ( par_loop > 0 )
          int_loop = (len / par_loop) + ( ( (len % par_loop) == 0) ? 0 : 1 );
//      printf("T: parallel for: b= %d, e = %d, l = %d, par = %d, int = %d \n", b, e, len, par_loop, int_loop);
      cilk_for (typename Policy::member_type i = 0 ; i < par_loop ; ++i ) {
        for ( typename Policy::member_type j = (int_loop * i); j < ( (int_loop * i) + int_loop); j++ ) {
//           printf(" parallel for: i = %d, j = %d \n", (const int)i, j);
           if (j < e)
              m_functor( t , (const typename Policy::member_type)j );
        }
      }
    }

public:

  inline
  void execute() const
    { this-> template exec< typename Policy::work_tag >(); }

  inline
  ParallelFor( const FunctorType & arg_functor
             , const Policy      & arg_policy )
    : m_functor( arg_functor )
    , m_policy(  arg_policy )
    {}
};

template< class FunctorType , class ReducerType , class ... Traits >
class ParallelReduce< FunctorType
                    , Kokkos::RangePolicy< Traits ... >
                    , ReducerType
                    , Kokkos::Experimental::CilkPlus
                    >
{
private:

  typedef Kokkos::RangePolicy< Traits ... > Policy ;
  typedef typename Policy::work_tag                                  WorkTag ;
  typedef Kokkos::Experimental::CilkPlus exec_space;

  typedef Kokkos::Impl::if_c< std::is_same<InvalidType,ReducerType>::value, FunctorType, ReducerType> ReducerConditional;

  typedef typename ReducerConditional::type ReducerTypeFwd;
  typedef typename Kokkos::Impl::if_c< std::is_same<InvalidType,ReducerType>::value, WorkTag, void>::type WorkTagFwd;

  typedef FunctorAnalysis< FunctorPatternInterface::REDUCE , Policy , FunctorType > Analysis ;
  typedef Kokkos::Impl::kokkos_cilk_reducer< ReducerTypeFwd, FunctorType, typename Analysis::value_type, WorkTagFwd > cilk_reducer_wrapper;

  typedef Kokkos::Impl::FunctorValueInit<   ReducerTypeFwd , WorkTagFwd >  ValueInit ;

  typedef typename Analysis::pointer_type    pointer_type ;
  typedef typename Analysis::reference_type  reference_type ;

  const FunctorType   m_functor ;
  const Policy        m_policy ;
  const ReducerType   m_reducer ;
  const pointer_type  m_result_ptr ;

  void internal_reduce(const typename Policy::member_type e, const typename Policy::member_type b, int int_loop, int i, char** i_ptr) const {
     for ( typename Policy::member_type j = (int_loop * i); j < ( (int_loop * i) + int_loop); j++ ) {
        if (j < e && i_ptr != NULL) {
           int array_ndx = NODE_ID();
           void * const l_ptr = (void*) (i_ptr[array_ndx]);
           reference_type lupdate = ValueInit::init(  get_reducer<cilk_reducer_wrapper>()->f , l_ptr );
           get_reducer<cilk_reducer_wrapper>()->f( (const typename Policy::member_type)j , lupdate );
           get_reducer<cilk_reducer_wrapper>()->join( lupdate );
        }
     }
  }

  void initialize_cilk_reducer(const size_t l_alloc_bytes) const
  {      
      global_reducer = (void*)mw_malloc2d(NODELETS(), sizeof(cilk_reducer_wrapper));
      for (int i = 0; i < NODELETS(); i++) {
         cilk_reducer_wrapper * f = (cilk_reducer_wrapper*)global_reducer;
         new ((cilk_reducer_wrapper *)&(f[i])) cilk_reducer_wrapper(ReducerConditional::select(m_functor , m_reducer), l_alloc_bytes);
      }
  }

  template< class TagType >
  inline
  typename std::enable_if< std::is_same< TagType , void >::value >::type
  exec( reference_type update, const size_t l_alloc_bytes ) const
    {
      Kokkos::HostSpace space;
      initialize_cilk_reducer(l_alloc_bytes);
      void * w_ptr = mw_malloc2d(NODELETS(), l_alloc_bytes);
//      size_t working_set = l_alloc_bytes * (m_policy.end() - m_policy.begin());
//      void * w_ptr = NULL; 
//      if (working_set > 0) {               
//         w_ptr = space.allocate( working_set );
//         memset( w_ptr, 0, working_set );
//      }

      const typename Policy::member_type e = m_policy.end();
      const typename Policy::member_type b = m_policy.begin();
      const typename Policy::member_type len = e-b;
      const typename Policy::member_type par_loop = len > 16 ? 16 : len;
      typename Policy::member_type int_loop = 1;
      if ( par_loop > 0 )
          int_loop = (len / par_loop) + ( ( (len % par_loop) == 0) ? 0 : 1 );

//      printf("parallel reduce: b= %d, e = %d, l = %d, par = %d, int = %d \n", b, e, len, par_loop, int_loop);
//      cilk_for (typename Policy::member_type i = 0 ; i < par_loop ; ++i ) {
//      }

      for (int i = 0; i < par_loop; i++) {
         cilk_spawn_at(&(((char**)w_ptr)[i % NODELETS()][0]))  internal_reduce(e, b, int_loop, i, (char**)w_ptr);
      }
      get_reducer<cilk_reducer_wrapper>()->update_value(update);
      get_reducer<cilk_reducer_wrapper>()->release_resources();
      global_reducer = NULL;
      if (w_ptr != NULL) {
            mw_free(w_ptr);
//          space.deallocate(w_ptr, working_set);
      }
    }

  template< class TagType >
  inline
  typename std::enable_if< ! std::is_same< TagType , void >::value >::type
  exec( reference_type update, const size_t l_alloc_bytes ) const
    {
      const TagType t{} ;

      Kokkos::HostSpace space;
      cilk_reducer_wrapper cilk_reducer(ReducerConditional::select(m_functor , m_reducer), l_alloc_bytes);
      INITIALIZE_CILK_REDUCER( cilk_reducer_wrapper, cilk_reducer )
      size_t working_set = l_alloc_bytes * (m_policy.end() - m_policy.begin());
      void * w_ptr = NULL; 
      if (working_set > 0) {      
         w_ptr = space.allocate( working_set );
         memset( w_ptr, 0, working_set );
      }
      const typename Policy::member_type e = m_policy.end();
      const typename Policy::member_type b = m_policy.begin();
      const typename Policy::member_type len = e-b;
      const typename Policy::member_type par_loop = len > 16 ? 16 : len;
      typename Policy::member_type int_loop = 1;
      if ( par_loop > 0 )
          int_loop = (len / par_loop) + ( ( (len % par_loop) == 0) ? 0 : 1 );
//      printf("T: parallel reduce: b= %d, e = %d, l = %d, par = %d, int = %d \n", b, e, len, par_loop, int_loop);
      cilk_for (typename Policy::member_type i = 0 ; i < par_loop ; ++i ) {
        for ( typename Policy::member_type j = (int_loop * i); j < ( (int_loop * i) + int_loop); j++ ) {
           if (j < e && w_ptr != NULL) {
              void * const l_ptr = (void*)(((char*)w_ptr)+((i - m_policy.begin()) * l_alloc_bytes));
              reference_type lupdate = ValueInit::init(  ReducerConditional::select(m_functor , m_reducer) , l_ptr );
              m_functor( t, (const typename Policy::member_type)i , lupdate );
              cilk_reducer.join( lupdate );
           }
        }
      }
      cilk_reducer.update_value( update );
      cilk_reducer.release_resources();
      global_reducer = NULL;
      if (w_ptr != NULL) {
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

      this-> template exec< WorkTag >( update, pool_reduce_size );

      Kokkos::Impl::FunctorFinal< ReducerTypeFwd , WorkTagFwd >::
        final(  ReducerConditional::select(m_functor , m_reducer) , ptr );


    }

  template< class HostViewType >
  ParallelReduce( const FunctorType  & arg_functor ,
                  const Policy       & arg_policy ,
                  const HostViewType & arg_result_view ,
                  typename std::enable_if<
                               Kokkos::is_view< HostViewType >::value &&
                              !Kokkos::is_reducer_type<ReducerType>::value
                  ,void*>::type = NULL)
    : m_functor( arg_functor )
    , m_policy( arg_policy )
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
                , Policy       arg_policy
                , const ReducerType& reducer )
    : m_functor( arg_functor )
    , m_policy(  arg_policy )
    , m_reducer( reducer )
    , m_result_ptr(  reducer.view().data() )
    {
    }
};
}
}

#endif
