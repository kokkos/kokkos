#include<SYCL/Kokkos_SYCL_KernelLaunch.hpp>
#include <algorithm>
#include <functional>
namespace Kokkos {
namespace Impl {

template< typename FunctorType , class ... Traits >
struct ParallelFor< FunctorType
                 , Kokkos::RangePolicy< Traits ... >
                 , Kokkos::Experimental::SYCL
                 >
{
  typedef Kokkos::RangePolicy< Traits ... > Policy;
  typedef typename Policy::member_type  Member ;
  typedef typename Policy::work_tag     WorkTag ;
  typedef typename Policy::launch_bounds LaunchBounds ;

  FunctorType  m_functor ;
  Policy       m_policy ;

//  ParallelFor() = delete ;
//  ParallelFor& operator=( const ParallelFor & ) = delete ;

  template< class TagType >
  typename std::enable_if< std::is_same< TagType , void >::value >::type
  exec_range( const Member i ) const
    { m_functor( i ); }

  template< class TagType >
  typename std::enable_if< ! std::is_same< TagType , void >::value >::type
  exec_range( const Member i ) const
    { m_functor( TagType() , i ); }

//public:

  typedef FunctorType functor_type ;

  inline
  void operator()(cl::sycl::id<1> item) const
    {
      int idx = item[0];
      m_functor(idx);
    }

  inline
  void execute() const {
      int start = m_policy.begin();
      int end = m_policy.end();
      int extent = end - start;
      std::cerr << "Setting range = " << extent << std::endl;
      cl::sycl::range<1> dispatch_range(extent);

      m_policy.space().impl_internal_space_instance()->m_queue->submit([&] (cl::sycl::handler& cgh) {

			// cl::sycl::stream out(4096,1024,cgh);
          cgh.parallel_for(dispatch_range,
				  	  [=](cl::sycl::id<1> item) {
        	  	  	   	   // out << item[0] << sycl::endl;
        	  	  	   	   m_functor( static_cast<const int>(item[0]) );
          	  });
	        });

		m_policy.space().impl_internal_space_instance()->m_queue->wait();
    }

  ParallelFor( const FunctorType  & arg_functor ,
               const Policy       & arg_policy )
    : m_functor( arg_functor )
    , m_policy(  arg_policy )
    { }

};

}
}
