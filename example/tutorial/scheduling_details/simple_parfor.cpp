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
#include <cstdio>
#include <string>

#if defined(KOKKOS_ENABLE_OPENMP)

using view_type = Kokkos::View<int*>;

struct SimpleFunctor {
  view_type view;

  SimpleFunctor(view_type view_) : view(view_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const { view(i) = omp_get_thread_num(); }
};

// This example should be removed before merging (it's for WIP debugging and
// visualization).
// Use KOKKOS_ENABLE_NATIVE_OPENMP to switch OpenMP implementation.
// You can run this with:
// ./example/tutorial/scheduling_details/KokkosExample_scheduling_details -N 129
// -d

int main(int argc, char* argv[]) {
  int N        = 1003;   // view size
  bool dynamic = false;  // use dynamic scheduling

  // Read command line arguments.
  for (int i = 0; i < argc; i++) {
    if (strcmp(argv[i], "-N") == 0) {
      N = atoi(argv[++i]);
      printf("  User N is %d\n", N);
    } else if (strcmp(argv[i], "-d") == 0) {
      dynamic = true;
    } else if ((strcmp(argv[i], "-h") == 0) ||
               (strcmp(argv[i], "-help") == 0)) {
      printf("Run a simple Kokkos::parallel_for example.\n\nOptions:\n");
      printf("  -N <int>:    range size\n");
      printf("  -d:    use dynamic schedule type\n");
      printf("  -h:    print this message\n\n");
      exit(1);
    }
  }

  Kokkos::initialize(argc, argv);

  {
    view_type view("scheduling-details", N);

    if (dynamic) {
      using policy_t = Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                                           Kokkos::Schedule<Kokkos::Dynamic> >;
      Kokkos::parallel_for(policy_t(0, N), SimpleFunctor(view));
    } else {
      Kokkos::parallel_for(N, SimpleFunctor(view));
    }

    // PRINT VISUALISATION OF WORK
    auto num_threads = Kokkos::OpenMP::impl_thread_pool_size();
    std::vector<std::string> strings(num_threads, std::string());

    for (int i = 0; i < num_threads; i++)  // rows represent threads
    {
      strings[i] = "[thread " + std::to_string(i) + "] ";
      for (int j = 0; j < N; j++)  // columns represent iterations (iwork)
      {
        if (view(j) == i) {
          strings[i] += "*";
        } else {
          strings[i] += " ";
        }
      }
    }

    for (auto& s : strings) {
      std::cout << s << "\n";
    }
  }

  Kokkos::finalize();
}

#endif
