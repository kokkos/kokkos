/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
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

#include <vector>

#include <Kokkos_Core.hpp>

namespace Test {

namespace {

template< class ExecSpace >
struct CountFillFunctor {
  KOKKOS_INLINE_FUNCTION
  std::int32_t operator()(std::int32_t row, std::int32_t* fill) const {
    auto n = (row % 4) + 1;
    if (fill) {
      for (std::int32_t j = 0; j < n; ++j) {
        fill[j] = j + 1;
      }
    }
    return n;
  }
};

template< class CrsType, class ExecSpace, class scalarType >
struct RunUpdateCrsTest {

  CrsType graph {};
  RunUpdateCrsTest( CrsType g_in ) : graph(g_in)
  {
  }

  void run_test() {
     parallel_for ("TestCrs1", Kokkos::RangePolicy<ExecSpace>(0,graph.numRows()),*this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const scalarType row) const {
     auto row_map = graph.row_map;
     auto entries = graph.entries;
     auto j_start = (row >= static_cast<decltype(row)>(row_map.extent(0))) ? 0 : row_map(row);
     auto j_end = (row+1 >= static_cast<decltype(row)>(row_map.extent(0))) ? 0 : row_map(row+1)-j_start;
     for (scalarType j = 0; j < j_end; ++j) {
        entries(j_start+j) = (j+1)*(j+1);
     }
  }
};

template< class ExecSpace >
void test_count_fill(std::int32_t nrows) {
  Kokkos::Crs<std::int32_t, ExecSpace, void, std::int32_t> graph;
  Kokkos::count_and_fill_crs(graph, nrows, CountFillFunctor<ExecSpace>());
  ASSERT_EQ(graph.numRows(), nrows); 
  auto row_map = Kokkos::create_mirror_view(graph.row_map);
  Kokkos::deep_copy(row_map, graph.row_map);
  auto entries = Kokkos::create_mirror_view(graph.entries);
  Kokkos::deep_copy(entries, graph.entries);
  for (std::int32_t row = 0; row < nrows; ++row) {
    auto n = (row % 4) + 1;
    ASSERT_EQ(row_map(row + 1) - row_map(row), n);
    for (std::int32_t j = 0; j < n; ++j) {
      ASSERT_EQ(entries(row_map(row) + j), j + 1);
    }
  }
}

// Test Crs Constructor / assignment operation by 
// using count and fill to create/populate initial graph,
// then use parallel_for with Crs directly to update content
// then verify results
template< class ExecSpace >
void test_constructor(std::int32_t nrows) {

  typedef Kokkos::Crs<std::int32_t, ExecSpace, void, std::int32_t> crs_int32;
  crs_int32 graph;
  Kokkos::count_and_fill_crs(graph, nrows, CountFillFunctor<ExecSpace>());
  ASSERT_EQ(graph.numRows(), nrows);

  RunUpdateCrsTest<crs_int32, ExecSpace, std::int32_t> crstest(graph);  
  crstest.run_test();

  auto row_map = Kokkos::create_mirror_view(graph.row_map);
  Kokkos::deep_copy(row_map, graph.row_map);
  auto entries = Kokkos::create_mirror_view(graph.entries);
  Kokkos::deep_copy(entries, graph.entries);

  for (std::int32_t row = 0; row < nrows; ++row) {
    auto n = (row % 4) + 1;
    ASSERT_EQ(row_map(row + 1) - row_map(row), n);    
    for (std::int32_t j = 0; j < n; ++j) {
      ASSERT_EQ(entries(row_map(row) + j), (j + 1)*(j+1));
    }
   }
}

} // anonymous namespace

TEST_F( TEST_CATEGORY, crs_count_fill )
{
  test_count_fill<TEST_EXECSPACE>(0);
  test_count_fill<TEST_EXECSPACE>(1);
  test_count_fill<TEST_EXECSPACE>(2);
  test_count_fill<TEST_EXECSPACE>(3);
  test_count_fill<TEST_EXECSPACE>(13);
  test_count_fill<TEST_EXECSPACE>(100);
  test_count_fill<TEST_EXECSPACE>(1000);
  test_count_fill<TEST_EXECSPACE>(10000);
}

TEST_F( TEST_CATEGORY, crs_copy_constructor )
{
  test_constructor<TEST_EXECSPACE>(0);
  test_constructor<TEST_EXECSPACE>(1);
  test_constructor<TEST_EXECSPACE>(2);
  test_constructor<TEST_EXECSPACE>(3);
  test_constructor<TEST_EXECSPACE>(13);
  test_constructor<TEST_EXECSPACE>(100);
  test_constructor<TEST_EXECSPACE>(1000);
  test_constructor<TEST_EXECSPACE>(10000);
}


} // namespace Test
