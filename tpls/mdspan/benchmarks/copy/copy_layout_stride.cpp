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

#include <benchmark/benchmark.h>

#include "fill.hpp"

using index_type = int;

MDSPAN_IMPL_INLINE_VARIABLE constexpr auto dyn = Kokkos::dynamic_extent;

template <class MDSpan, class... DynSizes>
void BM_MDSpan_Copy_2D_right(benchmark::State& state, MDSpan, DynSizes... dyn) {
  using value_type = typename MDSpan::value_type;
  auto buffer = std::make_unique<value_type[]>(
    MDSpan{nullptr, dyn...}.mapping().required_span_size()
  );
  auto buffer2 = std::make_unique<value_type[]>(
    MDSpan{nullptr, dyn...}.mapping().required_span_size()
  );
  auto s = MDSpan{buffer.get(), dyn...};
  mdspan_benchmark::fill_random(s);
  auto dest = MDSpan{buffer2.get(), dyn...};
  for (auto _ : state) {
    for(index_type i = 0; i < s.extent(0); ++i) {
      for (index_type j = 0; j < s.extent(1); ++j) {
          dest(i, j) = s(i, j);
      }
    }
    auto sdata = s.data_handle();
    auto ddata = dest.data_handle();
    benchmark::DoNotOptimize(sdata);
    benchmark::DoNotOptimize(ddata);
  }
  state.SetBytesProcessed(s.size() * sizeof(value_type) * state.iterations());
}

BENCHMARK_CAPTURE(
  BM_MDSpan_Copy_2D_right, size_100_100, Kokkos::mdspan<int, Kokkos::extents<index_type, 100, 100>>(nullptr)
);
BENCHMARK_CAPTURE(
  BM_MDSpan_Copy_2D_right, size_100_dyn, Kokkos::mdspan<int, Kokkos::extents<index_type, 100, dyn>>(), 100
);
BENCHMARK_CAPTURE(
  BM_MDSpan_Copy_2D_right, size_dyn_dyn, Kokkos::mdspan<int, Kokkos::dextents<index_type, 2>>(), 100, 100
);

//================================================================================

template <class MDSpan, class LayoutMapping>
void BM_MDSpan_Copy_2D_stride(benchmark::State& state, MDSpan, LayoutMapping map) {
  benchmark::DoNotOptimize(map);
  using value_type = typename MDSpan::value_type;
  auto buffer = std::make_unique<value_type[]>(
    map.required_span_size()
  );
  auto buffer2 = std::make_unique<value_type[]>(
    map.required_span_size()
  );
  auto s = MDSpan{buffer.get(), map};
  mdspan_benchmark::fill_random(s);
  auto dest = MDSpan{buffer2.get(), map};
  for (auto _ : state) {
    for(index_type i = 0; i < s.extent(0); ++i) {
      for (index_type j = 0; j < s.extent(1); ++j) {
        dest(i, j) = s(i, j);
      }
    }
    auto sdata = s.data_handle();
    auto ddata = dest.data_handle();
    benchmark::DoNotOptimize(sdata);
    benchmark::DoNotOptimize(ddata);
  }
  state.SetBytesProcessed(s.size() * sizeof(value_type) * state.iterations());
}

BENCHMARK_CAPTURE(
  BM_MDSpan_Copy_2D_stride, size_100_100,
  Kokkos::mdspan<int, Kokkos::extents<index_type, 100, 100>, Kokkos::layout_stride>(nullptr,
          Kokkos::layout_stride::mapping<Kokkos::extents<index_type, 100, 100>>()),
  Kokkos::layout_stride::template mapping<Kokkos::extents<index_type, 100, 100>>(
    Kokkos::extents<index_type, 100, 100>{},
    // layout right
    std::array<size_t, 2>{100, 1}
  )
);
BENCHMARK_CAPTURE(
  BM_MDSpan_Copy_2D_stride, size_100_100d,
  Kokkos::mdspan<int, Kokkos::extents<index_type, 100, dyn>, Kokkos::layout_stride>(),
  Kokkos::layout_stride::template mapping<Kokkos::extents<index_type, 100, dyn>>(
    Kokkos::extents<index_type, 100, dyn>{100},
    // layout right
    std::array<size_t, 2>{100, 1}
  )
);
BENCHMARK_CAPTURE(
  BM_MDSpan_Copy_2D_stride, size_100d_100,
  Kokkos::mdspan<int, Kokkos::extents<index_type, dyn, 100>, Kokkos::layout_stride>(),
  Kokkos::layout_stride::template mapping<Kokkos::extents<index_type, dyn, 100>>(
    Kokkos::extents<index_type, dyn, 100>{100},
    // layout right
    std::array<size_t, 2>{100, 1}
  )
);
BENCHMARK_CAPTURE(
  BM_MDSpan_Copy_2D_stride, size_100d_100d,
  Kokkos::mdspan<int, Kokkos::extents<index_type, dyn, dyn>, Kokkos::layout_stride>(),
  Kokkos::layout_stride::template mapping<Kokkos::extents<index_type, dyn, dyn>>(
    Kokkos::extents<index_type, dyn, dyn>{100, 100},
    // layout right
    std::array<size_t, 2>{100, 1}
  )
);

//================================================================================

template <class T, class Extents, class MapSrc, class MapDst>
void BM_MDSpan_Copy_2D_stride_diff_map(benchmark::State& state,
  T, Extents, MapSrc map_src, MapDst map_dest
) {
  using value_type = T;
  auto buff_src = std::make_unique<value_type[]>(
    map_src.required_span_size()
  );
  auto buff_dest = std::make_unique<value_type[]>(
    map_dest.required_span_size()
  );
  using map_stride_dyn = Kokkos::layout_stride;
  using mdspan_type = Kokkos::mdspan<T, Extents, map_stride_dyn>;
  auto src = mdspan_type{buff_src.get(), map_src};
  mdspan_benchmark::fill_random(src);
  auto dest = mdspan_type{buff_dest.get(), map_dest};
  for (auto _ : state) {
    for(index_type i = 0; i < src.extent(0); ++i) {
      for (index_type j = 0; j < src.extent(1); ++j) {
        dest(i, j) = src(i, j);
      }
    }
    auto sdata = src.data_handle();
    auto ddata = dest.data_handle();
    benchmark::DoNotOptimize(sdata);
    benchmark::DoNotOptimize(ddata);
  }
  state.SetBytesProcessed(src.extent(0) * src.extent(1) * sizeof(value_type) * state.iterations());
}

BENCHMARK_CAPTURE(
  BM_MDSpan_Copy_2D_stride_diff_map, size_100d_100d_bcast_0, int(),
  Kokkos::extents<index_type, dyn, dyn>{100, 100},
  Kokkos::layout_stride::template mapping<Kokkos::extents<index_type, dyn, dyn>>(
    Kokkos::extents<index_type, dyn, dyn>{100, 100},
    // layout right
    std::array<size_t, 2>{0, 1}
  ),
  Kokkos::layout_stride::template mapping<Kokkos::extents<index_type, dyn, dyn>>(
    Kokkos::extents<index_type, dyn, dyn>{100, 100},
    // layout right
    std::array<size_t, 2>{100, 1}
  )
);

BENCHMARK_CAPTURE(
  BM_MDSpan_Copy_2D_stride_diff_map, size_100d_100d_bcast_1, int(),
  Kokkos::extents<index_type, dyn, dyn>{100, 100},
  Kokkos::layout_stride::template mapping<Kokkos::extents<index_type, dyn, dyn>>(
    Kokkos::extents<index_type, dyn, dyn>{100, 100},
    // layout right
    std::array<size_t, 2>{1, 0}
  ),
  Kokkos::layout_stride::template mapping<Kokkos::extents<index_type, dyn, dyn>>(
    Kokkos::extents<index_type, dyn, dyn>{100, 100},
    // layout right
    std::array<size_t, 2>{100, 1}
  )
);

BENCHMARK_CAPTURE(
  BM_MDSpan_Copy_2D_stride_diff_map, size_100d_100d_bcast_both, int(),
  Kokkos::extents<index_type, dyn, dyn>{100, 100},
  Kokkos::layout_stride::template mapping<Kokkos::extents<index_type, dyn, dyn>>(
    Kokkos::extents<index_type, dyn, dyn>{100, 100},
    // layout right
    std::array<size_t, 2>{0, 0}
  ),
  Kokkos::layout_stride::template mapping<Kokkos::extents<index_type, dyn, dyn>>(
    Kokkos::extents<index_type, dyn, dyn>{100, 100},
    // layout right
    std::array<size_t, 2>{100, 1}
  )
);


//================================================================================

template <class T>
void BM_Raw_Copy_1D(benchmark::State& state, T, size_t size) {
  benchmark::DoNotOptimize(size);
  using value_type = T;
  auto buffer = std::make_unique<value_type[]>(size);
  {
    // just for setup...
    auto wrapped = Kokkos::mdspan<T, Kokkos::dextents<index_type, 1>>{buffer.get(), size};
    mdspan_benchmark::fill_random(wrapped);
  }
  value_type* src = buffer.get();
  auto buffer2 = std::make_unique<value_type[]>(size);
  value_type* dest = buffer2.get();
  for (auto _ : state) {
    for(size_t i = 0; i < size; ++i) {
      dest[i] = src[i];
    }
    benchmark::DoNotOptimize(src);
    benchmark::DoNotOptimize(dest);
  }
  state.SetBytesProcessed(size * sizeof(value_type) * state.iterations());
}

BENCHMARK_CAPTURE(
  BM_Raw_Copy_1D, size_10000, int(), 10000
);

//================================================================================

template <class T>
void BM_Raw_Copy_2D(benchmark::State& state, T, size_t x, size_t y) {
  using value_type = T;
  auto buffer = std::make_unique<value_type[]>(x * y);
  {
    // just for setup...
    auto wrapped = Kokkos::mdspan<T, Kokkos::dextents<index_type, 1>>{buffer.get(), x * y};
    mdspan_benchmark::fill_random(wrapped);
  }
  value_type* src = buffer.get();
  auto buffer2 = std::make_unique<value_type[]>(x * y);
  value_type* dest = buffer2.get();
  for (auto _ : state) {
    for(size_t i = 0; i < x; ++i) {
      for(size_t j = 0; j < y; ++j) {
        dest[i*y + j] = src[i*y + j];
      }
    }
    benchmark::DoNotOptimize(src);
    benchmark::DoNotOptimize(dest);
  }
  state.SetBytesProcessed(x * y * sizeof(value_type) * state.iterations());
}

BENCHMARK_CAPTURE(
  BM_Raw_Copy_2D, size_100_100, int(), 100, 100
);

//================================================================================

BENCHMARK_MAIN();
