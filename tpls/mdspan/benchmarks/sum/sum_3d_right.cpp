//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER
#include <mdspan/mdspan.hpp>

#include <memory>
#include <random>

#include "sum_3d_common.hpp"
#include "../fill.hpp"

//================================================================================

using index_type = int;

template <class T, size_t... Es>
using lmdspan = Kokkos::mdspan<T, Kokkos::extents<int, Es...>, Kokkos::layout_left>;
template <class T, size_t... Es>
using rmdspan = Kokkos::mdspan<T, Kokkos::extents<int, Es...>, Kokkos::layout_right>;

//================================================================================

template <class MDSpan, class... DynSizes>
void BM_MDSpan_Sum_3D_right(benchmark::State& state, MDSpan, DynSizes... dyn) {

  using value_type = typename MDSpan::value_type;
  auto buffer = std::make_unique<value_type[]>(
    MDSpan{nullptr, dyn...}.mapping().required_span_size()
  );

  auto s = MDSpan{buffer.get(), dyn...};
  mdspan_benchmark::fill_random(s);

  for (auto _ : state) {
    benchmark::DoNotOptimize(s);
    auto sdata = s.data_handle();
    benchmark::DoNotOptimize(sdata);
    value_type sum = 0;
    for(index_type i = 0; i < s.extent(0); ++i) {
      for (index_type j = 0; j < s.extent(1); ++j) {
        for (index_type k = 0; k < s.extent(2); ++k) {
          sum += s(i, j, k);
        }
      }
    }
    benchmark::DoNotOptimize(sum);
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(s.size() * sizeof(value_type) * state.iterations());
}
MDSPAN_BENCHMARK_ALL_3D(BM_MDSpan_Sum_3D_right, right_, rmdspan, 20, 20, 20);
MDSPAN_BENCHMARK_ALL_3D(BM_MDSpan_Sum_3D_right, left_, lmdspan, 20, 20, 20);
MDSPAN_BENCHMARK_ALL_3D(BM_MDSpan_Sum_3D_right, right_, rmdspan, 200, 200, 200);
MDSPAN_BENCHMARK_ALL_3D(BM_MDSpan_Sum_3D_right, left_, lmdspan, 200, 200, 200);

//================================================================================

BENCHMARK_CAPTURE(
  BM_Raw_Sum_3D_right, size_20_20_20, int(), size_t(20), size_t(20), size_t(20)
);
BENCHMARK_CAPTURE(
  BM_Raw_Sum_3D_right, size_200_200_200, int(), size_t(200), size_t(200), size_t(200)
);

//================================================================================

BENCHMARK_CAPTURE(
  BM_Raw_Static_Sum_3D_right, size_20_20_20, int(),
  std::integral_constant<size_t, 20>{},
  std::integral_constant<size_t, 20>{},
  std::integral_constant<size_t, 20>{}
);
BENCHMARK_CAPTURE(
  BM_Raw_Static_Sum_3D_right, size_200_200_200, int(),
  std::integral_constant<size_t, 200>{},
  std::integral_constant<size_t, 200>{},
  std::integral_constant<size_t, 200>{}
);


//================================================================================

BENCHMARK_CAPTURE(
  BM_Raw_Sum_1D, size_8000, int(), 8000
);

//================================================================================

BENCHMARK_MAIN();
