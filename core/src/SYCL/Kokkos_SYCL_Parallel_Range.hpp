#include<SYCL/Kokkos_SYCL_KernelLaunch.hpp>

namespace Kokkos {
namespace Impl {

template< class FunctorType , class ... Traits >
class ParallelFor< FunctorType
                 , Kokkos::RangePolicy< Traits ... >
                 , Kokkos::Experimental::SYCL
                 >
{
public:
  typedef Kokkos::RangePolicy< Traits ... > Policy;
private:

  typedef typename Policy::member_type  Member ;
  typedef typename Policy::work_tag     WorkTag ;
  typedef typename Policy::launch_bounds LaunchBounds ;

  const FunctorType  m_functor ;
  const Policy       m_policy ;

  ParallelFor() = delete ;
  ParallelFor & operator = ( const ParallelFor & ) = delete ;

  template< class TagType >
  typename std::enable_if< std::is_same< TagType , void >::value >::type
  exec_range( const Member i ) const
    { m_functor( i ); }

  template< class TagType >
  typename std::enable_if< ! std::is_same< TagType , void >::value >::type
  exec_range( const Member i ) const
    { m_functor( TagType() , i ); }

public:

  typedef FunctorType functor_type ;

  inline
  void operator()(void) const
    {
    }

  inline
  void execute() const
    {
    }

  ParallelFor( const FunctorType  & arg_functor ,
               const Policy       & arg_policy )
    : m_functor( arg_functor )
    , m_policy(  arg_policy )
    { }

};

}
}
