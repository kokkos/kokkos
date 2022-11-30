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
#ifndef TESTVIEWSUBVIEWCREATION_HPP_
#define TESTVIEWSUBVIEWCREATION_HPP_
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

TEST(TEST_CATEGORY, subview_creation_timings) {
  TEST_EXECSPACE exec;
  Kokkos::View<int*, TEST_EXECSPACE> v("V", 1000);
  Kokkos::Timer t;
  for (int rep = 0; rep < 100; rep++) {
    Kokkos::parallel_for(
        "Reproducer", Kokkos::RangePolicy<TEST_EXECSPACE>(exec, 0, 10000000),
        KOKKOS_LAMBDA(int i) {
          auto vsub = Kokkos::subview(v, i % 1000);
          vsub()++;
        });
  }
  exec.fence();
  double elapsed_managed = t.seconds();
  std::cout << "Managed: " << elapsed_managed << " s.\n";
  Kokkos::View<int*, TEST_EXECSPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      vu = v;
  t.reset();
  for (int rep = 0; rep < 100; rep++) {
    Kokkos::parallel_for(
        "Reproducer", Kokkos::RangePolicy<TEST_EXECSPACE>(exec, 0, 10000000),
        KOKKOS_LAMBDA(int i) {
          auto vusub = Kokkos::subview(vu, i % 1000);
          vusub()++;
        });
  }
  exec.fence();
  double elapsed_unmanaged = t.seconds();
  std::cout << "Unmanaged: " << elapsed_unmanaged << " s.\n";
  ASSERT_LE(elapsed_managed, 1.2 * elapsed_unmanaged);
}

#endif
