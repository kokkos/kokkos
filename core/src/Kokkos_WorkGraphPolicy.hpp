#ifndef KOKKOS_WORKGRAPHPOLICY_HPP
#define KOKKOS_WORKGRAPHPOLICY_HPP

namespace Kokkos {

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

  graph_type graph;

private:

  using counts_type = Kokkos::View<index_type*, memory_space>;
  counts_type counts;

  struct TagComputeCounts {};
  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputeCounts, index_type i) const {
    atomic_increment( &counts(graph.entries(i)) );
  }
  void compute_counts() {
    const auto num_tasks = graph.numRows();
    const auto num_entries = graph.entries.dimension_0();
    counts = counts_type("WorkGraphPolicy:counts", num_tasks);
    using policy_type = RangePolicy<execution_space, TagComputeCounts>;
    using closure_type = Impl::ParallelFor<self_type, policy_type>;
    const closure_type closure(*this, policy_type(0, num_entries));
  }

};

}

#endif
