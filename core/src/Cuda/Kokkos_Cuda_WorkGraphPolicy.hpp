#ifndef KOKKOS_CUDA_WORKGRAPHPOLICY_HPP
#define KOKKOS_CUDA_WORKGRAPHPOLICY_HPP

namespace Kokkos {
namespace Impl {

template< class FunctorType , class ... Traits >
class ParallelFor< FunctorType ,
                   Kokkos::Experimental::WorkGraphPolicy< Traits ... > ,
                   Kokkos::Cuda
                 >
  : public Kokkos::Experimental::Impl::
           WorkGraphExec< FunctorType,
                          Kokkos::Cuda,
                          Traits ...
                        >
{
private:

  typedef Kokkos::Experimental::WorkGraphPolicy< Traits ... >   Policy ;
  typedef Kokkos::Experimental::Impl::
          WorkGraphExec<FunctorType, Kokkos::Cuda, Traits ... > Base ;
  typedef ParallelFor<FunctorType, Policy, Kokkos::Cuda>        Self ;

  template< class TagType >
  __device__
  typename std::enable_if< std::is_same< TagType , void >::value >::type
  exec_one(const typename Policy::member_type& i) const {
    Base::m_functor( i );
  }

  template< class TagType >
  __device__
  typename std::enable_if< ! std::is_same< TagType , void >::value >::type
  exec_one(const typename Policy::member_type& i) const {
    const TagType t{} ;
    Base::m_functor( t , i );
  }

public:

  __device__
  inline
  void operator()() const {
    for (std::int32_t i; (-1 != (i = Base::before_work())); ) {
      exec_one< typename Policy::work_tag >( i );
      Base::after_work(i);
    }
  }

  inline
  void execute() const
  {
    const int warps_per_block = 4 ;
    const dim3 grid( Kokkos::Impl::cuda_internal_multiprocessor_count() , 1 , 1 );
    const dim3 block( 1 , Kokkos::Impl::CudaTraits::WarpSize , warps_per_block );
    const int shared = 0 ;
    const cudaStream_t stream = 0 ;

    Kokkos::Impl::CudaParallelLaunch<Self>(*this, grid, block, shared, stream);
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

#endif /* #define KOKKOS_CUDA_WORKGRAPHPOLICY_HPP */
