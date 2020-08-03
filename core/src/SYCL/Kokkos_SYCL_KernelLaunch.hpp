#ifndef KOKKOS_SYCL_KERNELLAUNCH_HPP_
#define KOKKOS_SYCL_KERNELLAUNCH_HPP_

#include <SYCL/Kokkos_SYCL_Error.hpp>
#include <CL/sycl.hpp>

template <class T>
class kokkos_sycl_functor;

namespace Kokkos {
namespace Experimental {
namespace Impl {

template <class Driver>
void sycl_launch_bind(Driver tmp, cl::sycl::handler& cgh) {
  cgh.parallel_for(
      cl::sycl::range<1>(tmp.m_policy.end() - tmp.m_policy.begin()), tmp);
}


template <class Driver>
void sycl_launch(const Driver driver) {
  isTriviallyCopyable<Driver>();
  isTriviallyCopyable<decltype(driver.m_functor)>();
  driver.m_policy.space().impl_internal_space_instance()->m_queue->wait();
#ifndef SYCL_USE_BIND_LAUNCH
  driver.m_policy.space().impl_internal_space_instance()->m_queue->submit(
      [&](cl::sycl::handler& cgh) {
        cgh.parallel_for(
            cl::sycl::range<1>(driver.m_policy.end() - driver.m_policy.begin()),
            [=](cl::sycl::item<1> item) {
              int id = item.get_linear_id();
              driver.m_functor(id);
            });
      });
  driver.m_policy.space().impl_internal_space_instance()->m_queue->wait();
#else
  driver.m_policy.space().impl_internal_space_instance()->m_queue->submit(
      std::bind(Kokkos::Experimental::Impl::sycl_launch_bind<Driver>, driver,
                std::placeholders::_1));
#endif
}

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#endif  // KOKKOS_SYCL_KERNELLAUNCH_HPP_
