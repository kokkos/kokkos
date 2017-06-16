#ifndef KOKKOS_SERIAL_WORKGRAPHPOLICY_HPP
#define KOKKOS_SERIAL_WORKGRAPHPOLICY_HPP

#if defined( KOKKOS_ENABLE_SERIAL )

namespace Kokkos {
namespace Impl {

template< class FunctorType , class ... Traits >
class ParallelFor< FunctorType ,
                   Kokkos::Experimental::WorkGraphPolicy< Traits ... > ,
                   Kokkos::Serial
                 >
  : public Kokkos::Experimental::Impl::
           WorkGraphExec< FunctorType,
                          Kokkos::Serial,
                          Traits ...
                        >
{
private:

  typedef Kokkos::Experimental::WorkGraphPolicy< Traits ... > Policy ;
  typedef Kokkos::Experimental::Impl::
          WorkGraphExec<FunctorType, Kokkos::Serial, Traits ... > Base ;

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
    for (std::int32_t i; (-1 != (i = Base::before_work())); ) {
      exec_one< typename Policy::work_tag >( i );
      Base::after_work(i);
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

#endif // defined( KOKKOS_ENABLE_SERIAL )
#endif /* #define KOKKOS_SERIAL_WORKGRAPHPOLICY_HPP */
