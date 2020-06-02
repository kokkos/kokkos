#include <SYCL/Kokkos_SYCL_Error.hpp>

template<class T>
class kokkos_sycl_functor;

#define CONCAT_IMPL( x, y ) x##y
#define MACRO_CONCAT( x, y ) CONCAT_IMPL( x, y )
#define CGH_PARALLEL_FOR cgh.parallel_for<class MACRO_CONCAT( kokkos_sycl_functor, __COUNTER__ )>
namespace Kokkos {
namespace Experimental {
namespace Impl {

template<class Driver>
void sycl_launch_bind(Driver tmp, cl::sycl::handler& cgh) {
  cgh.parallel_for(cl::sycl::range<1>(tmp.m_policy.end()-tmp.m_policy.begin()), tmp);
}

template<class Driver>
void sycl_launch(const Driver driver) {
       driver.m_policy.space().impl_internal_space_instance()->m_queue->wait();
       #ifndef SYCL_USE_BIND_LAUNCH
       driver.m_policy.space().impl_internal_space_instance()->m_queue->submit([&] (cl::sycl::handler& cgh) {
         //CGH_PARALLEL_FOR (
         #ifdef SYCL_JUST_DONT_NAME_KERNELS
         cgh.parallel_for (//<class kokkos_sycl_functor> (
         #else
         cgh.parallel_for <class kokkos_sycl_functor<Driver>> (
         #endif
            cl::sycl::range<1>(driver.m_policy.end()-driver.m_policy.begin()), [=] (cl::sycl::item<1> item) {
              int id = item.get_linear_id();
                driver.m_functor(id);        
         });
      });
      driver.m_policy.space().impl_internal_space_instance()->m_queue->wait();
      #else
      driver.m_policy.space().impl_internal_space_instance()->m_queue->submit(
        std::bind(Kokkos::Experimental::Impl::sycl_launch_bind<Driver>,driver,std::placeholders::_1));
      #endif
}


}
}
}
