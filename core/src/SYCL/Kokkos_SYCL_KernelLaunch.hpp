#ifndef KOKKOS_SYCL_KERNELLAUNCH_HPP_
#define KOKKOS_SYCL_KERNELLAUNCH_HPP_

#include <SYCL/Kokkos_SYCL_Error.hpp>
#include <CL/sycl.hpp>

template <class T>
class kokkos_sycl_functor;

namespace Kokkos {
namespace Experimental {
namespace Impl {

template <typename Policy, typename Functor>
void sycl_direct_launch(const Policy& policy, const Functor& functor) {
  // Convenience references
  const Kokkos::Experimental::SYCL& space = policy.space();
  Kokkos::Experimental::Impl::SYCLInternal& instance =
      *space.impl_internal_space_instance();
  cl::sycl::queue& q = *instance.m_queue;

  q.wait();

  auto end   = policy.end();
  auto begin = policy.begin();
  cl::sycl::range<1> range(end - begin);

  q.submit([&](cl::sycl::handler& cgh) {
    cgh.parallel_for(range, [=](cl::sycl::item<1> item) {
      int id = item.get_linear_id();
      functor(id);
    });
  });

  q.wait();
}

// Indirectly launch a functor by explicitly creating it in USM shared memory
template <typename Policy, typename Functor>
void sycl_indirect_launch(const Policy& policy, const Functor& functor) {
  // Convenience references
  const Kokkos::Experimental::SYCL& space = policy.space();
  Kokkos::Experimental::Impl::SYCLInternal& instance =
      *space.impl_internal_space_instance();
  Kokkos::Experimental::Impl::SYCLInternal::IndirectKernelMem&
      indirectKernelMem = instance.m_indirectKernelMem;

  // Put a copy of the functor into USM
  //
  // Note:  if sycl::usm::alloc::device == indirectKernelMem.kind and functor
  // is neither an implicit-lifetime nor trivially-copyable type, dereferencing
  // the pointer returned technically invokes UB.  This is done because USM
  // device memory is presumably much faster than USM shared memory but we
  // cannot copy construct into the former.
  auto kernelFunctorPtr = indirectKernelMem.copy_from(functor);

  // Use reference_wrapper (because it is both trivially copyable and invocable)
  // to wrap the dereferenced pointer and launch it
  sycl_direct_launch(policy, std::reference_wrapper(*kernelFunctorPtr));
}

template <class Driver>
void sycl_launch(const Driver driver) {
  // if the functor is trivially copyable, we can launch it directly;
  // otherwise, we will launch it indirectly via explicitly creating
  // it in USM shared memory.
  if constexpr (std::is_trivially_copyable_v<decltype(driver.m_functor)>)
    sycl_direct_launch(driver.m_policy, driver.m_functor);
  else
    sycl_indirect_launch(driver.m_policy, driver.m_functor);
}

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#endif  // KOKKOS_SYCL_KERNELLAUNCH_HPP_

