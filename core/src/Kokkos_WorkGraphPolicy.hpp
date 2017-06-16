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
  using work_tag = typename traits::work_tag;
  using memory_space = typename execution_space::memory_space;
  using graph_type = Kokkos::Experimental::Crs<index_type, execution_space, void, index_type>;
  using member_type = index_type;

  graph_type graph;

  WorkGraphPolicy(graph_type arg_graph):graph(arg_graph) {}
};

namespace Impl {

template< class functor_type , class execution_space, class ... policy_args >
class WorkGraphExec
{
 public:

  using self_type = WorkGraphExec< functor_type, execution_space, policy_args ... >;
  using policy_type = Kokkos::Experimental::WorkGraphPolicy< policy_args ... >;
  using member_type = typename policy_type::member_type;
  using memory_space = typename execution_space::memory_space;

 protected:

  const functor_type m_functor;
  const policy_type  m_policy;

 private:

  using ints_type = View<std::int32_t*, execution_space>;
  using atomic_ints_type = View<std::int32_t*, execution_space, MemoryTraits<Atomic>>;
  using range_type = Kokkos::pair<std::int32_t, std::int32_t>;
  using ranges_type = View<range_type*, memory_space>;
  const std::int32_t m_total_work;
  ints_type m_counts;
  ints_type m_queue;
  ranges_type m_ranges;

  void init_queue() {
  }

 public:

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

 private:

  KOKKOS_INLINE_FUNCTION
  std::int32_t pop_work() const {
    std::cerr << "beginning of pop_work, m_queue[0] = " << m_queue[0] << '\n';
    range_type w(-1,-1);
    while (true) {
      const range_type w_new( w.first + 1 , w.second );
      w = atomic_compare_exchange( &m_ranges(0) , w , w_new );
      std::cerr << "after CAS, m_queue[0] = " << m_queue[0] << '\n';
      if ( w.first < w.second ) { // there was work in the queue
        if ( w_new.first == w.first + 1 && w_new.second == w.second ) {
          // we got a work item
          std::int32_t i;
          // the push_work function may have incremented the end counter
          // but not yet written the work index into the queue.
          // wait until the entry is valid.
          std::cerr << "before -1 spin, m_queue[0] = " << m_queue[0] << '\n';
          while ( -1 == ( i = m_queue[ w.first ] ) );
          std::cerr << "after -1 spin, m_queue[0] = " << m_queue[0] << '\n';
          std::cerr << "w.first was " << w.first << '\n';
          std::cerr << "popping work item " << i << '\n';
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
    std::cerr << "pushing work item " << i << '\n';
    range_type w(-1,-1);
    while (true) {
      const range_type w_new( w.first , w.second + 1 );
      // try to increment the end counter
      w = atomic_compare_exchange( &m_ranges(0) , w , w_new );
      // stop trying if the increment was successful
      if ( w.first == w_new.first && w.second + 1 == w_new.second ) break;
    }
    // write the work index into the claimed spot in the queue
    std::cerr << "m_queue[ w.second = " << w.second << " ] = i = " << i << '\n';
    m_queue[ w.second ] = i;
    // push this write out into the memory system
    memory_fence();
    std::cerr << "double check: m_queue[ " << w.second << " ] = " << m_queue[w.second] << '\n';
  }

 public:

  KOKKOS_INLINE_FUNCTION
  std::int32_t before_work() const {
    return pop_work();
  }

  KOKKOS_INLINE_FUNCTION
  void after_work(std::int32_t i) const {
    /* fence any writes that were done by the work item itself
       (usually writing its result to global memory) */
    memory_fence();
    const std::int32_t begin = m_policy.graph.row_map( i );
    const std::int32_t end = m_policy.graph.row_map( i + 1 );
    for (std::int32_t j = begin; j < end; ++j) {
      const std::int32_t next = m_policy.graph.entries( j );
      /* View is atomic, this is atomic_fetch_add(-1) */
      const std::int32_t new_count = (m_counts(next) -= 1);
      if ( new_count == 0 )  push_work( next );
    }
  }

  inline
  WorkGraphExec( const functor_type & arg_functor
               , const policy_type  & arg_policy )
    : m_functor( arg_functor )
    , m_policy(  arg_policy )
    , m_total_work( arg_policy.graph.numRows() )
    , m_queue(ViewAllocateWithoutInitializing("queue"), m_total_work)
  {
    std::cerr << "calling main WorkGraphExec constructor\n";
    if (arg_policy.graph.numRows() > std::numeric_limits<std::int32_t>::max()) {
      Kokkos::abort("WorkGraphPolicy work must be indexable using int32_t");
    }
    {
    //ints_type counts;
      get_crs_transpose_counts(m_counts, arg_policy.graph);
    //m_counts = counts;
    }
  //m_queue = ints_type("queue", m_total_work);
    std::cerr << "post-ctor, m_queue.size() " << m_queue.size() << " m_queue.use_count() " << m_queue.use_count() << '\n';
    deep_copy(m_queue, std::int32_t(-1));
    m_ranges = ranges_type("ranges", 1);
    fill_queue();
  }

};

} // namespace Impl
} // namespace Experimental
} // namespace Kokkos

#include "impl/Kokkos_Serial_WorkGraphPolicy.hpp"

#endif /* #define KOKKOS_WORKGRAPHPOLICY_HPP */
