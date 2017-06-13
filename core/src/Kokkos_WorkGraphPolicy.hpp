#ifndef KOKKOS_WORKGRAPHPOLICY_HPP
#define KOKKOS_WORKGRAPHPOLICY_HPP

#include <Kokkos_Crs.hpp>

namespace Kokkos {

template< class ... Properties >
class WorkGraphPolicy : public Impl::PolicyTraits<Properties ... >
{
public:
  typedef Kokkos::Crs<index_type, execution_space, void, index_type> graph_type;
  typedef typename execution_space::memory_space memory_space;

  graph_type graph;
private:
  typedef Kokkos::View<index_type*, memory_space> counts_type;
  counts_type compute_counts() const {
  }
};

};

#endif
