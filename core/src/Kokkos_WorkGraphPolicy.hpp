#ifndef KOKKOS_WORKGRAPHPOLICY_HPP
#define KOKKOS_WORKGRAPHPOLICY_HPP

namespace Kokkos {
namespace Experimental {

template< class ... Properties >
class WorkGraphPolicy
{
public:
  using self_type = WorkGraphPolicy<Properties ... >;
  using traits = Kokkos::Impl::PolicyTraits<Properties ... >;
  using index_type = typename traits::index_type;
  using execution_space = typename traits::execution_space;
  using memory_space = typename execution_space::memory_space;
  using graph_type = Kokkos::Experimental::Crs<index_type, execution_space, void, index_type>;
  using member_type = index_type;

  graph_type graph;

};

namespace Impl {

template< class functor_type , class execution_space, class ... policy_args >
class WorkGraphExec
{
 private:

  using self_type = WorkGraphExec< functor_type, execution_space, policy_args ... >;
  using policy_type = Kokkos::Experimental::WorkGraphPolicy< policy_args ... >;
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
    using closure_type = Kokkos::Impl::ParallelFor<self_type, policy_type>;
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
    using closure_type = Kokkos::Impl::ParallelFor<self_type, policy_type>;
    const std::int32_t num_deps = m_policy.graph.entries.dimension_0();
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
    using closure_type = Kokkos::Impl::ParallelFor<self_type, policy_type>;
    const closure_type closure(*this, policy_type(0, m_total_work));
    closure.execute();
    execution_space::fence();
  }

  struct TagZeroRanges {};
  KOKKOS_INLINE_FUNCTION
  void operator()(TagZeroRanges, std::int32_t i) const {
    m_ranges[i] = range_type(0, 0);
  }
  void zero_ranges() {
    using policy_type = RangePolicy<std::int32_t, execution_space, TagZeroRanges>;
    using closure_type = Kokkos::Impl::ParallelFor<self_type, policy_type>;
    const closure_type closure(*this, policy_type(0, 1));
    closure.execute();
    execution_space::fence();
  }

  struct TagFillQueue {};
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFillQueue, std::int32_t i) const {
    if (m_counts[i] == 0) push_work(i);
  }
  void fill_queue() {
    using policy_type = RangePolicy<std::int32_t, execution_space, TagFillQueue>;
    using closure_type = Kokkos::Impl::ParallelFor<self_type, policy_type>;
    const closure_type closure(*this, policy_type(0, m_total_work));
    closure.execute();
    execution_space::fence();
  }

  KOKKOS_INLINE_FUNCTION
  std::int32_t pop_work() const {
    range_type w(-1,-1);
    while (true) {
      const range_type w_new( w.first + 1 , w.second );
      w = atomic_compare_exchange( m_ranges , w , w_new );
      if ( w.first < w.second ) { // there was work in the queue
        if ( w_new.first == w.first + 1 && w_new.second == w.second ) {
          // we got a work item
          std::int32_t i;
          // the push_work function may have incremented the end counter
          // but not yet written the work index into the queue.
          // wait until the entry is valid.
          while ( -1 == ( i = m_queue[ w.first ] ) );
          return i;
        } // we got a work item
      } else { // there was no work in the queue
        if (w.first == m_total_work) { // all work is done
          return -1;
        } // all work is done
      } // there was no work in the queue
    } // while (true)
  }

  KOKKOS_INLINE_FUNCTION
  void push_work(std::int32_t i) const {
    range_type w(-1,-1);
    while (true) {
      const range_type w_new( w.first , w.second + 1 );
      // try to increment the end counter
      w = atomic_compare_exchange( m_ranges , w , w_new );
      // stop trying if the increment was successful
      if ( w.first == w_new.first && w.second + 1 == w_new.second ) break;
    }
    // write the work index into the claimed spot in the queue
    m_queue[ w.second ] = i;
    // push this write out into the memory system
    memory_fence();
  }

 public:

  KOKKOS_INLINE_FUNCTION
  std::int32_t before_work() const {
    return pop_work();
  }

  KOKKOS_INLINE_FUNCTION
  void after_work(std::int32_t i) const {
    const std::int32_t begin = m_policy.graph.row_map( i );
    const std::int32_t end = m_policy.graph.row_map( i + 1 );
    for (std::int32_t j = begin; j < end; ++j) {
      const std::int32_t next = m_policy.graph.entries( j );
      const std::int32_t old_count =
        atomic_fetch_sub( &m_counts[ next ], std::int32_t( 1 ) );
      if ( old_count == 1 )  push_work( next );
    }
  }

  inline
  WorkGraphExec( const functor_type & arg_functor
               , const policy_type  & arg_policy )
    : m_functor( arg_functor )
    , m_policy(  arg_policy )
    , m_total_work( arg_policy.graph.numRows() )
    , m_counts( static_cast<decltype(m_counts)>(
          memory_space::allocate( m_total_work * sizeof( std::int32_t ) )))
    , m_queue( static_cast<decltype(m_queue)>(
          memory_space::allocate( m_total_work * sizeof( std::int32_t ) )))
    , m_ranges( static_cast<decltype(m_ranges)>(
          memory_space::allocate( sizeof( range_type ) )))
  {
    if (arg_policy.graph.numRows() > std::numeric_limits<std::int32_t>::max()) {
      Kokkos::abort("WorkGraphPolicy work must be indexable using int32_t");
    }
    zero_counts();
    fill_counts();
    init_queue();
    zero_ranges();
    fill_queue();
  }

  inline
  void destroy() {
    memory_space::deallocate( m_counts, m_total_work * sizeof( std::int32_t ) );
    memory_space::deallocate( m_queue, m_total_work * sizeof( std::int32_t ) );
    memory_space::deallocate( m_ranges, sizeof( range_type ) );
  }
};

} // namespace Impl
} // namespace Experimental
} // namespace Kokkos

#endif /* #define KOKKOS_WORKGRAPHPOLICY_HPP */
