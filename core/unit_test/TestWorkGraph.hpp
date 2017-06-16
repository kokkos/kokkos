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
// Questions? Contact  H. Carter Edwards (hcedwar@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <vector>

#include <Kokkos_Core.hpp>

namespace Test {

namespace {

template< class ExecSpace >
struct TestWorkGraph {

  struct TagCompute {};
  
  using MemorySpace = typename ExecSpace::memory_space;
  using Policy = WorkGraphPolicy<std::int32_t, TagCompute, ExecSpace>;
  using Graph = typename Policy::graph_type;
  using RowMap = typename Graph::row_map_type;
  using Entries = typename Graph::entries_type;
  using Values = Kokkos::View<long*, MemorySpace>;

  long m_input;
  RowMap m_row_map;
  Entries m_entries;
  Values m_values;

  TestWorkGraph(std::int32_t arg_size):size(arg_size) {}

  inline
  long full_fibonacci( long n ) {
    constexpr long mask = 0x03;
    long fib[4] = { 0, 1, 1, 2 };
    for ( long i = 2; i <= n; ++i ) {
      fib[ i & mask ] = fib[ ( i - 1 ) & mask ] + fib[ ( i - 2 ) & mask ];
    }
    return fib[ n & mask ];
  }

  struct HostEntry {
    long input;
    std::int32_t parent;
  };
  std::vector<HostEntry> form_host_graph() {
    std::vector<HostEntry> g;
    g.push_back({ m_input , -1 });
    for (std::int32_t i = 0; i < std::int32_t(g.size()); ++i) {
      auto& e = g.at(std::size_t(i));
      if (e.input < 2) continue;
      g.push_back({ e.input - 1, i });
      g.push_back({ e.input - 2, i });
    }
    return g;
  }

  void form_graph() {
    auto hg = form_host_inverse_graph();
    m_row_map = RowMap("row_map", hg.size() + 1);
    m_entries = Entries("entries", hg.size() - 1); // all but the first have a parent
    auto h_row_map = Kokkos::create_mirror_view(m_row_map);
    auto h_entries = Kokkos::create_mirror_view(m_entries);
    h_row_map(0) = 0;
    for (std::int32_t i = 0; i < std::int32_t(hg.size()); ++i) {
      auto& e = hg.at(std::size_t(i));
      h_row_map(i + 1) = i;
      if (e.parent == -1) continue;
      h_entries(i - 1) = e.parent;
    }
    Kokkos::deep_copy(m_row_map, h_row_map);
    Kokkos::deep_copy(m_entries, h_entries);
  }

  void test_for() {
  }
};

} // anonymous namespace

TEST_F( TEST_CATEGORY, workgraph_for )
{
  { TestRange< TEST_EXECSPACE > f(0); f.test_for(); }
  { TestRange< TEST_EXECSPACE > f(1); f.test_for(); }
  { TestRange< TEST_EXECSPACE > f(3); f.test_for(); }
}

} // namespace Test

