/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/
#include <Kokkos_Core.hpp>

Kokkos::Tools::Experimental::ToolProgrammingInterface interface;

struct SampleFunctor {
  KOKKOS_FUNCTION void operator()(int) const {}
};
void next_cb(const char *, const uint32_t, uint64_t *) {
  interface.fence(0);
  std::cout << "Called the next callback\n";
};

int main(int argc, char *argv[]) {
  /**
   * Calling this function before initialize is necessary,
   * intialize is what invokes the callback we're setting
   */
  Kokkos::Tools::Experimental::set_provide_tool_programming_interface_callback(
      [](const unsigned int,
         Kokkos::Tools::Experimental::ToolProgrammingInterface
             provided_interface) { interface = provided_interface; });
  Kokkos::initialize(argc, argv);
  {
    Kokkos::Tools::Experimental::set_begin_parallel_for_callback(
        [](const char *, const uint32_t, uint64_t *) {
          interface.set_tool_hook({Kokkos_Tools_begin_parallel_for_event,
                                   reinterpret_cast<void *>(&next_cb)});
        });
    SampleFunctor samp;
    Kokkos::parallel_for("dummy_kernel", Kokkos::RangePolicy<>(0, 1), samp);
    Kokkos::parallel_for("dummy_kernel", Kokkos::RangePolicy<>(0, 1), samp);
  }
  Kokkos::finalize();
}
