#ifndef KOKKOS_THREADS_WORKGRAPHPOLICY_HPP
#define KOKKOS_THREADS_WORKGRAPHPOLICY_HPP

namespace Kokkos {
namespace Impl {

template< class FunctorType , class ... Traits >
class ParallelFor< FunctorType ,
                   Kokkos::Experimental::WorkGraphPolicy< Traits ... > ,
                   Kokkos::Threads
                 >
  : public Kokkos::Experimental::Impl::
           WorkGraphExec< FunctorType,
                          Kokkos::Threads,
                          Traits ...
                        >
{
private:

  typedef Kokkos::Experimental::WorkGraphPolicy< Traits ... > Policy ;
  typedef Kokkos::Experimental::Impl::
          WorkGraphExec<FunctorType, Kokkos::Threads, Traits ... > Base ;
  typedef ParallelFor<FunctorType,
                      Kokkos::Experimental::WorkGraphPolicy<Traits ...>,
                      Kokkos::Threads> Self ;

  template< class TagType >
  typename std::enable_if< std::is_same< TagType , void >::value >::type
  exec_one(const typename Policy::member_type& i) const {
    Base::m_functor( i );
  }

  template< class TagType >
  typename std::enable_if< ! std::is_same< TagType , void >::value >::type
  exec_one(const typename Policy::member_type& i) const {
    const TagType t{} ;
    Base::m_functor( t , i );
  }

  inline void exec_one_thread() const {
    for (std::int32_t i; (-1 != (i = Base::before_work())); ) {
      exec_one< typename Policy::work_tag >( i );
      Base::after_work(i);
    }
  }

  static inline void thread_main( ThreadsExec&, const void* arg ) {
    const Self& self = *(static_cast<const Self*>(arg));
    self.exec_one_thread();
  }

public:

  inline
  void execute() const
  {
    ThreadsExec::start( & Self::thread_main, this );
    ThreadsExec::fence();
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

#endif /* #define KOKKOS_THREADS_WORKGRAPHPOLICY_HPP */
