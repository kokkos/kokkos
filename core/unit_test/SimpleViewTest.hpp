#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <TestSharedAlloc.hpp>


namespace Test {


void test_local_memspace() {

//   test_shared_alloc< Kokkos::Experimental::EmuLocalSpace, TEST_EXECSPACE >();
   test_repl_shared_alloc< Kokkos::Experimental::EmuReplicatedSpace, TEST_EXECSPACE >();

}



void test_local_only() {





}


TEST_F( TEST_CATEGORY, local_only )
{
   test_local_memspace();
   test_local_only();
}



}
