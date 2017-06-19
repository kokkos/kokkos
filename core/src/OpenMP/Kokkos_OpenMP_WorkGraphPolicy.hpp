#ifndef KOKKOS_OPENMP_WORKGRAPHPOLICY_HPP
#define KOKKOS_OPENMP_WORKGRAPHPOLICY_HPP

#if defined( KOKKOS_ENABLE_OPENMP )

namespace Kokkos {
namespace Impl {

template< class FunctorType , class ... Traits >
class ParallelFor< FunctorType ,
                   Kokkos::Experimental::WorkGraphPolicy< Traits ... > ,
                   Kokkos::OpenMP
                 >
  : public Kokkos::Experimental::Impl::
           WorkGraphExec< FunctorType,
                          Kokkos::OpenMP,
                          Traits ...
                        >
{
private:

  typedef Kokkos::Experimental::WorkGraphPolicy< Traits ... > Policy ;
  typedef Kokkos::Experimental::Impl::
          WorkGraphExec<FunctorType, Kokkos::OpenMP, Traits ... > Base ;

  template< class TagType >
  typename std::enable_if< std::is_same< TagType , void >::value >::type
  exec_one(const typename Policy::member_type& i) const {
    Base::m_functor( i );
  }

  template< class TagType >
  typename std::enable_if< ! std::is_same< TagType , void >::value >::type
  exec_one(const typename Policy::member_type& i) const {
    const TagType t{} ;
    Base::m_functor( t , i );
  }

public:

  inline
  void execute() const
  {
    const int pool_size = OpenMP::thread_pool_size();

    #pragma omp parallel num_threads(pool_size)
    {
      for (std::int32_t i; (-1 != (i = Base::before_work())); ) {
        exec_one< typename Policy::work_tag >( i );
        Base::after_work(i);
      }
    }
  }

  inline
  ParallelFor( const FunctorType & arg_functor
             , const Policy      & arg_policy )
    : Base( arg_functor, arg_policy )
  {
  }
};

} // namespace Impl
} // namespace Kokkos

#endif // defined( KOKKOS_ENABLE_OPENMP )
#endif /* #define KOKKOS_OPENMP_WORKGRAPHPOLICY_HPP */
