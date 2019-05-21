#include <SYCL/Kokkos_SYCL_Error.hpp>

#define CONCAT_IMPL( x, y ) x##y
#define MACRO_CONCAT( x, y ) CONCAT_IMPL( x, y )
#define CGH_PARALLEL_FOR cgh.parallel_for<class MACRO_CONCAT( kokkos_sycl_functor, __COUNTER__ )>
namespace Kokkos {
namespace Experimental {
namespace Impl {

template<class Driver>
void sycl_launch(const Driver driver) {
       driver.m_policy.space().impl_internal_space_instance()->m_queue->submit([&] (cl::sycl::handler& cgh) {
         //CGH_PARALLEL_FOR (
         cgh.parallel_for<class kokkos_sycl_functor> (
            cl::sycl::range<1>(driver.m_policy.end()-driver.m_policy.begin()), [=] (cl::sycl::item<1> item) {
              int id = item.get_linear_id();
                driver.m_functor(id);        
         });
       });
}
}
}
}
