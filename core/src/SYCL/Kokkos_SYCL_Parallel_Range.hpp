#ifndef KOKKOS_SYCL_PARALLEL_RANGE_HPP_
#define KOKKOS_SYCL_PARALLEL_RANGE_HPP_
#include <SYCL/Kokkos_SYCL_KernelLaunch.hpp>
#include <algorithm>
#include <functional>

template< class FunctorType , class ExecPolicy >
class Kokkos::Impl::ParallelFor< FunctorType
                 //NLIBER , Kokkos::RangePolicy< Traits ... >
                 , ExecPolicy
                 , Kokkos::Experimental::SYCL
                 >
{
public:
  //NLIBER typedef Kokkos::RangePolicy< Traits ... > Policy;
  typedef ExecPolicy Policy;
private:

  typedef typename Policy::member_type  Member ;
  typedef typename Policy::work_tag     WorkTag ;
  typedef typename Policy::launch_bounds LaunchBounds ;

public:
  const FunctorType  m_functor ;
  const Policy       m_policy ;

private:
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
  void operator()(cl::sycl::item<1> item) const
    {
      int id = item.get_linear_id();
      m_functor(id);
    }

  inline
  void execute() const
    {
      /*#ifdef SYCL_USE_BIND_LAUNCH
      m_policy.space().impl_internal_space_instance()->m_queue->submit(
        std::bind(Kokkos::Experimental::Impl::sycl_launch_bind<ParallelFor>,this,std::placeholders::_1));
      #else      */
      Kokkos::Experimental::Impl::sycl_launch(*this);
//      #endif
    }

  ParallelFor( const FunctorType  & arg_functor ,
               const Policy       & arg_policy )
    : m_functor( arg_functor )
    , m_policy(  arg_policy )
    { }

};

// NLIBER
#if 0
template <bool B>
class Kokkos::Impl::ParallelFor<
    Kokkos::Impl::ViewCopy<
        Kokkos::View<int*, ::Kokkos::LayoutLeft,
                     ::Kokkos::Device<::Kokkos::Experimental::SYCL,
                                      ::Kokkos::AnonymousSpace>,
                     ::Kokkos::MemoryTraits<0>>,
        Kokkos::View<const int*, ::Kokkos::LayoutLeft,
                     ::Kokkos::Device<::Kokkos::Experimental::SYCL,
                                      ::Kokkos::AnonymousSpace>,
                     ::Kokkos::MemoryTraits<0>>,
        Kokkos::LayoutLeft, Kokkos::Experimental::SYCL, 1, int, B>,
    Kokkos::RangePolicy<::Kokkos::Experimental::SYCL, ::Kokkos::IndexType<int>>,
    Kokkos::Experimental::SYCL> {
 public:
  using functor_type = Kokkos::Impl::ViewCopy<
      Kokkos::View<int*, ::Kokkos::LayoutLeft,
                   ::Kokkos::Device<::Kokkos::Experimental::SYCL,
                                    ::Kokkos::AnonymousSpace>,
                   ::Kokkos::MemoryTraits<0>>,
      Kokkos::View<const int*, ::Kokkos::LayoutLeft,
                   ::Kokkos::Device<::Kokkos::Experimental::SYCL,
                                    ::Kokkos::AnonymousSpace>,
                   ::Kokkos::MemoryTraits<0>>,
      Kokkos::LayoutLeft, Kokkos::Experimental::SYCL, 1, int, B>;

  using Policy = Kokkos::RangePolicy<::Kokkos::Experimental::SYCL,
                                     ::Kokkos::IndexType<int>>;

  void operator()(cl::sycl::item<1> item) const {
    int id = item.get_linear_id();
    m_functor(id);
  }

  void execute() const { Kokkos::Experimental::Impl::sycl_launch(*this); }

  ParallelFor(const functor_type& arg_functor, const Policy& arg_policy)
      : m_functor(arg_functor), m_policy(arg_policy) {}

  functor_type m_functor;
  Policy m_policy;
};
#endif


#endif // KOKKOS_SYCL_PARALLEL_RANGE_HPP_

