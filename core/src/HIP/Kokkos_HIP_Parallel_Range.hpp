#include<HIP/Kokkos_HIP_KernelLaunch.hpp>

namespace Kokkos {
namespace Impl {

template< class FunctorType , class ... Traits >
class ParallelFor< FunctorType
                 , Kokkos::RangePolicy< Traits ... >
                 , Kokkos::Experimental::HIP
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
  inline __device__
  typename std::enable_if< std::is_same< TagType , void >::value >::type
  exec_range( const Member i ) const
    { m_functor( i ); }

  template< class TagType >
  inline __device__
  typename std::enable_if< ! std::is_same< TagType , void >::value >::type
  exec_range( const Member i ) const
    { m_functor( TagType() , i ); }

public:

  typedef FunctorType functor_type ;

  inline
  __device__
  void operator()(void) const
    {
      const Member work_stride = blockDim.y * gridDim.x ;
      const Member work_end    = m_policy.end();

      for ( Member
              iwork =  m_policy.begin() + threadIdx.y + blockDim.y * blockIdx.x ;
              iwork <  work_end ;
              iwork = iwork < work_end - work_stride ? iwork + work_stride : work_end) {
        this-> template exec_range< WorkTag >( iwork );
      }
    }

  inline
  void execute() const
    {
      const typename Policy::index_type nwork = m_policy.end() - m_policy.begin();

      const int block_size = 256; // Choose block_size better
      const dim3 block(  1 ,block_size , 1);
      const dim3 grid( typename Policy::index_type(( nwork + block.y - 1 ) / block.y) , 1 , 1);

      Kokkos::Experimental::Impl::HIPParallelLaunch< ParallelFor, LaunchBounds >( *this , grid , block , 0 , m_policy.space().impl_internal_space_instance() , false );
    }

  ParallelFor( const FunctorType  & arg_functor ,
               const Policy       & arg_policy )
    : m_functor( arg_functor )
    , m_policy(  arg_policy )
    { }

};

}
}
