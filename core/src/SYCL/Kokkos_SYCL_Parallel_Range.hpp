#ifndef KOKKOS_SYCL_PARALLEL_RANGE_HPP_
#define KOKKOS_SYCL_PARALLEL_RANGE_HPP_

#include <SYCL/Kokkos_SYCL_KernelLaunch.hpp>
#include <algorithm>
#include <functional>

template <class FunctorType, class ExecPolicy>
class Kokkos::Impl::ParallelFor<FunctorType, ExecPolicy,
                                Kokkos::Experimental::SYCL> {
 public:
  typedef ExecPolicy Policy;

 private:
  typedef typename Policy::member_type Member;
  typedef typename Policy::work_tag WorkTag;
  typedef typename Policy::launch_bounds LaunchBounds;

 public:
  const FunctorType m_functor;
  const Policy m_policy;

 private:
  ParallelFor()        = delete;
  ParallelFor& operator=(const ParallelFor&) = delete;

  template <class TagType>
  typename std::enable_if<std::is_same<TagType, void>::value>::type exec_range(
      const Member i) const {
    m_functor(i);
  }

  template <class TagType>
  typename std::enable_if<!std::is_same<TagType, void>::value>::type exec_range(
      const Member i) const {
    m_functor(TagType(), i);
  }

 public:
  typedef FunctorType functor_type;

  inline void operator()(cl::sycl::item<1> item) const {
    int id = item.get_linear_id();
    m_functor(id);
  }

  inline void execute() const {
    /*#ifdef SYCL_USE_BIND_LAUNCH
    m_policy.space().impl_internal_space_instance()->m_queue->submit(
      std::bind(Kokkos::Experimental::Impl::sycl_launch_bind<ParallelFor>,this,std::placeholders::_1));
    #else      */
    Kokkos::Experimental::Impl::sycl_launch(*this);
    //      #endif
  }

  ParallelFor(const FunctorType& arg_functor, const Policy& arg_policy)
      : m_functor(arg_functor), m_policy(arg_policy) {}
};

#endif  // KOKKOS_SYCL_PARALLEL_RANGE_HPP_

