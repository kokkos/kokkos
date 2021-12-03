#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cstdlib>

using ExecSpace = Kokkos::DefaultExecutionSpace;
using Device = Kokkos::Device<ExecSpace, typename ExecSpace::memory_space>;
using TeamPol = Kokkos::TeamPolicy<ExecSpace>;
using TeamMem = typename TeamPol::member_type;

struct TestFunctor {
  KOKKOS_INLINE_FUNCTION void operator()(const TeamMem t) const
  {
    int* x = (int*) t.team_shmem().get_shmem(sizeof(int));
    Kokkos::single(Kokkos::PerTeam(t),
    [=]()
    {
      *x = 0;
    });
    t.team_barrier();
    Kokkos::atomic_add(x, 1);
    t.team_barrier();
    Kokkos::single(Kokkos::PerTeam(t),
    [=]()
    {
      printf("Value after: %d\n", *x);
    });
  }
};

int main() {
  Kokkos::initialize();
  std::cout << "Testing shared mem atomics on " << ExecSpace::name() << '\n';
  //yes, this is more shared than needed for 1 int, but I got cudaErrorIllegalAddress when requesting 4 bytes
  Kokkos::parallel_for(TeamPol(1, 8).set_scratch_size(0, Kokkos::PerTeam(8)), TestFunctor());
  Kokkos::finalize();
  return 0;
}
