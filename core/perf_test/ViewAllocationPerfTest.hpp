#include <Kokkos_Core.hpp>
#include <common/RAJAPerfSuite.hpp>
#include <common/Executor.hpp>
#include <common/QuickKernelBase.hpp>

namespace test {

        void BuildViewAllocationTest(rajaperf::Executor& exec) {
                exec.registerKernel("VIEW", rajaperf::make_kernel_base("ViewAllocationTest",
										exec.getRunParams(),
                                        // set up lambda
                                        // number of iterations , run_size
                                        KOKKOS_LAMBDA(const int, const int){

                                        },
                                        // run lambda
                                        // Iteration # tracker, run_size
                                        KOKKOS_LAMBDA(const int, const int run_size) {

                                        Kokkos::View<float*>TestAllocatedView("Test Uno", run_size);

                                        }));







        }



}
