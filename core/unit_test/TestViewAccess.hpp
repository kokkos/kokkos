
#include <stdexcept>
#include <limits>
#include <math.h>

#include <Kokkos_Core.hpp>

namespace Kokkos {
namespace Experimental {
   extern void * ers;
   extern replicated EmuReplicatedSpace els;
   extern void initialize_memory_space();
}}

namespace test {

template< class Scalar, class ExecSpace >
void TestViewAccess( int N ) {
   Kokkos::Experimental::initialize_memory_space();

   Kokkos::View< Scalar*, Kokkos::HostSpace > hs_view( "host", N );

   for (int i = 0; i < N; i++) {
      hs_view(i) = i*2;
   }
   Kokkos::View< const Scalar*, Kokkos::HostSpace > cp_view( hs_view );
   for (int i = 0 ; i < N; i++) {
      printf("host view copy %d \n", cp_view(i) );
   }
   
   printf(" size of EmuLocalSpace is %ld \n", sizeof(Kokkos::Experimental::EmuLocalSpace) );
   printf(" size of EmuReplicatedSpace is %ld \n", sizeof(Kokkos::Experimental::EmuReplicatedSpace) );
   printf(" size of HostSpace is %ld \n", sizeof(Kokkos::HostSpace) );
   fflush(stdout);

   long * refPtr = Kokkos::Experimental::EmuReplicatedSpace::getRefAddr();
/*   for (int i = 0; i < Kokkos::Experimental::EmuReplicatedSpace::memory_zones(); i++) {
      MIGRATE(&refPtr[i]);
//      Kokkos::View< Scalar*, Kokkos::Experimental::EmuLocalSpace > local_space_view( Kokkos::ViewAllocateWithoutInitializing("local"), N );
      Kokkos::View< Scalar*, Kokkos::Experimental::EmuLocalSpace > local_space_view( "local", N );      
  
      if (mw_islocal(local_space_view.data())) {
         for (int i = 0; i < N; i++) {
            local_space_view(i) = (Scalar)i;
         }
      } else {
         printf("local mem check skipped, pointer is not local \n");
      }      
      int node_id = NODE_ID();
      printf("local mem current node: %d \n", node_id);
      fflush(stdout);
   }
   printf("local memory view test complete\n");
*/

   Kokkos::View< Scalar*, Kokkos::Experimental::EmuReplicatedSpace > replicated_space_view( "replicated", N );   
   Kokkos::View< Scalar*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::ForceRemote> > global_space_view( "global", N );
//   Kokkos::View< Scalar*, Kokkos::HostSpace, Kokkos::MemoryTraits<0> > global_space_view( "global", N );      

   printf("Testing access to replicated space view\n");
   fflush(stdout);

   for (int i = 0; i < N; i++) {
      replicated_space_view(i) = (Scalar)i;
   }
   printf("Testing access to global space view\n");
   fflush(stdout);

   for (int i = 0; i < N; i++) {      
      MIGRATE(&refPtr[i%NODELETS()]);
      global_space_view(i) = (Scalar)i;
   }


}



TEST_F( TEST_CATEGORY, view_access )
{
  TestViewAccess< long, TEST_EXECSPACE >( 10 );
}

}
