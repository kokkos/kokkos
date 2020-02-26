#include <SYCL/Kokkos_SYCL_Error.hpp>


namespace Kokkos {
namespace Experimental {
namespace Impl {


template<class Driver>
void sycl_launch(Driver driver) {
       driver.m_policy.space().impl_internal_space_instance()->m_queue->wait();
       driver.m_policy.space().impl_internal_space_instance()->m_queue->submit([&] (cl::sycl::handler& cgh) {

       Driver dr_copy(driver);
    	   cgh.parallel_for (//<class kokkos_sycl_functor> (
    			   cl::sycl::range<1>{driver.m_policy.end()-driver.m_policy.begin()},
				   [=] (cl::sycl::id<1> item) {
    				   int idx = item[0];
    				   dr_copy.m_functor(idx);
           });
      });
}


}
}
}
