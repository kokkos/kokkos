#ifndef KOKKOS_WORKGRAPHPOLICY_HPP
#define KOKKOS_WORKGRAPHPOLICY_HPP

namespace Kokkos {
namespace Experimental {

template< class ... Properties >
class WorkGraphPolicy
{
public:
  using self_type = WorkGraphPolicy<Properties ... >;
  using traits = Impl::PolicyTraits<Properties ... >;
  using index_type = typename traits::index_type;
  using execution_space = typename traits::execution_space;
  using memory_space = typename execution_space::memory_space;
  using graph_type = Kokkos::Crs<index_type, execution_space, void, index_type>;
  using member_type = index_type;

  graph_type graph;

private:

};

template< class functor_type , class execution_space, class ... policy_args >
class WorkGraphExec
{
private:

  using self_type = WorkGraphExec< functor_type, execution_space, policy_args ... >;
  using policy_type = Kokkos::WorkGraphPolicy< policy_args ... >;
  using member_type = typename policy_type::member_type;
  using memory_space = typename execution_space::memory_space;

  const functor_type m_functor;
  const policy_type  m_policy;
  const std::int32_t m_total_work;
  volatile std::int32_t * const m_counts;
  volatile std::int32_t * const m_queue;
  using range_type = Kokkos::pair<std::int32_t, std::int32_t>;
  volatile range_type * const m_ranges;

  struct TagZeroCounts {};
  KOKKOS_INLINE_FUNCTION
  void operator()(TagZeroCounts, std::int32_t i) const {
    m_counts[i] = 0;
  }
  void zero_counts() {
    using policy_type = RangePolicy<std::int32_t, execution_space, TagZeroCounts>;
    using closure_type = Impl::ParallelFor<self_type, policy_type>;
    const closure_type closure(*this, policy_type(0, m_total_work));
    closure.execute();
    execution_space::fence();
  }

  struct TagFillCounts {};
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFillCounts, std::int32_t i) const {
    atomic_increment( &m_counts[ m_policy.graph.entries(i) ] );
  }
  void fill_counts() {
    using policy_type = RangePolicy<std::int32_t, execution_space, TagFillCounts>;
    using closure_type = Impl::ParallelFor<self_type, policy_type>;
    const std::int32_t num_deps = graph.entries.dimension_0();
    const closure_type closure(*this, policy_type(0, num_deps));
    closure.execute();
    execution_space::fence();
  }

  struct TagInitQueue {};
  KOKKOS_INLINE_FUNCTION
  void operator()(TagInitQueue, std::int32_t i) const {
    m_queue[i] = -1;
  }
  void init_queue() {
    using policy_type = RangePolicy<std::int32_t, execution_space, TagInitQueue>;
    using closure_type = Impl::ParallelFor<self_type, policy_type>;
    const closure_type closure(*this, policy_type(0, m_total_work));
    closure.execute();
    execution_space::fence();
  }

  struct TagInitRanges {};
  KOKKOS_INLINE_FUNCTION
  void operator()(TagInitRanges, std::int32_t i) const {
    m_ranges[i] = range_type(0, 0);
  }
  void init_ranges() {
    using policy_type = RangePolicy<std::int32_t, execution_space, TagInitRanges>;
    using closure_type = Impl::ParallelFor<self_type, policy_type>;
    const closure_type closure(*this, policy_type(0, 1));
    closure.execute();
    execution_space::fence();
  }

  KOKKOS_INLINE_FUNCTION
  std::int32_t pop_work() const {
  }

  KOKKOS_INLINE_FUNCTION
  void after_work(std::int32_t i) const {
  }

public:

  KOKKOS_INLINE_FUNCTION
  std::int32_t before_work() const {
  }

  KOKKOS_INLINE_FUNCTION
  void after_work(std::int32_t i) const {
  }

  inline
  WorkGraphExec( const FunctorType & arg_functor
               , const Policy      & arg_policy )
    : m_functor( arg_functor )
    , m_policy(  arg_policy )
    , m_total_work( arg_policy.graph.numRows() )
    , m_counts( static_cast<decltype(m_counts)>(
          memory_space::allocate( m_total_work * sizeof( std::int32_t ) )))
    , m_queue( static_cast<decltype(m_queue)>(
          memory_space::allocate( m_total_work * sizeof( std::int32_t ) )))
    , m_ranges( static_cast<decltype(m_ranges)>(
          memory_space::allocate( sizeof( range_type ) ))) {
    zero_counts();
    fill_counts();
    init_queue();
    init_ranges();
  }

  inline
  void destroy() {
    memory_space::deallocate( m_counts, m_total_work * sizeof( std::int32_t ) );
    memory_space::deallocate( m_queue, m_total_work * sizeof( std::int32_t ) );
    memory_space::deallocate( m_ranges, sizeof( range_type ) );
  }
};

} // namespace Experimental
} // namespace Kokkos

#endif /* #define KOKKOS_WORKGRAPHPOLICY_HPP */
