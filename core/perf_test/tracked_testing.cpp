#include <common/Executor.hpp>
#include <common/RAJAPerfSuite.hpp>
#include <Kokkos_Core.hpp>
#include "ViewAllocationPerfTest.hpp"



int main(int argc, char* argv[]) {
  {

    // set up Executor
    rajaperf::Executor exec(0, argv);
    //rajaperf::Executor exec(argc, argv);
    rajaperf::RunParams run_params(0, argv);
    // Initialize Kokkos
    Kokkos::initialize(argc, argv);
   
  test::BuildViewAllocationTest(exec);


    exec.setupSuite();
	
    // STEP 3: Report suite run summary
    //         (enable users to catch errors before entire suite is run)
    exec.reportRunSummary(std::cout);

    // STEP 4: Execute suite
    exec.runSuite();

    // STEP 5: Generate suite execution reports
    exec.outputRunData();
  }
  Kokkos::finalize();
  std::cout << "\n\nDONE!!!...." << std::endl;
}



