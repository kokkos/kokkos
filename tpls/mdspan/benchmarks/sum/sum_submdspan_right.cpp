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
#include <iostream>

#include "sum_3d_common.hpp"

//================================================================================

using index_type = int;

template <class T, size_t... Es>
using lmdspan = Kokkos::mdspan<T, Kokkos::extents<index_type, Es...>, Kokkos::layout_left>;
template <class T, size_t... Es>
using rmdspan = Kokkos::mdspan<T, Kokkos::extents<index_type, Es...>, Kokkos::layout_right>;

//================================================================================

template <class MDSpan, class... DynSizes>
void BM_MDSpan_Sum_Subspan_3D_right(benchmark::State& state, MDSpan, DynSizes... dyn) {

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
    using index_type = typename MDSpan::index_type;
    for(index_type i = 0; i < s.extent(0); ++i) {
      auto sub_i = Kokkos::submdspan(s, i, Kokkos::full_extent, Kokkos::full_extent);
      for (index_type j = 0; j < s.extent(1); ++j) {
        auto sub_i_j = Kokkos::submdspan(sub_i, j, Kokkos::full_extent);
        for (index_type k = 0; k < s.extent(2); ++k) {
          sum += sub_i_j(k);
        }
      }
    }
    benchmark::DoNotOptimize(sum);
    benchmark::DoNotOptimize(sdata);
  }
  state.SetBytesProcessed(s.size() * sizeof(value_type) * state.iterations());
}
MDSPAN_BENCHMARK_ALL_3D(BM_MDSpan_Sum_Subspan_3D_right, right_, rmdspan, 20, 20, 20);
MDSPAN_BENCHMARK_ALL_3D(BM_MDSpan_Sum_Subspan_3D_right, left_, lmdspan, 20, 20, 20);
MDSPAN_BENCHMARK_ALL_3D(BM_MDSpan_Sum_Subspan_3D_right, right_, rmdspan, 200, 200, 200);
MDSPAN_BENCHMARK_ALL_3D(BM_MDSpan_Sum_Subspan_3D_right, left_, lmdspan, 200, 200, 200);

//================================================================================

BENCHMARK_CAPTURE(
  BM_Raw_Sum_3D_right, size_20_20_20, int(), size_t(20), size_t(20), size_t(20)
);
BENCHMARK_CAPTURE(
  BM_Raw_Sum_3D_right, size_200_200_200, int(), size_t(200), size_t(200), size_t(200)
);

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

BENCHMARK_CAPTURE(
  BM_Raw_Sum_3D_right_iter_left, size_20_20_20, int(), 20, 20, 20
);
BENCHMARK_CAPTURE(
  BM_Raw_Sum_3D_right_iter_left, size_200_200_200, int(), 200, 200, 200
);
BENCHMARK_CAPTURE(
  BM_Raw_Static_Sum_3D_right_iter_left, size_20_20_20, int(),
  std::integral_constant<size_t, 20>{},
  std::integral_constant<size_t, 20>{},
  std::integral_constant<size_t, 20>{}
);
BENCHMARK_CAPTURE(
  BM_Raw_Static_Sum_3D_right_iter_left, size_200_200_200, int(),
  std::integral_constant<size_t, 200>{},
  std::integral_constant<size_t, 200>{},
  std::integral_constant<size_t, 200>{}
);

//================================================================================

namespace _impl {

template <class, class T>
MDSPAN_FORCE_INLINE_FUNCTION
constexpr T&& _repeated_with(T&& v) noexcept { return std::forward<T>(v); }

template <class T, class... Rest>
MDSPAN_FORCE_INLINE_FUNCTION
MDSPAN_IMPL_CONSTEXPR_14 void _do_sum_submdspan(
  T& sum,
  Kokkos::mdspan<T, Kokkos::extents<index_type>, Rest...> s
)
{
  sum += s();
}

template <class T, size_t E, size_t... Es, class... Rest>
MDSPAN_FORCE_INLINE_FUNCTION
MDSPAN_IMPL_CONSTEXPR_14 void _do_sum_submdspan(
  T& sum,
  Kokkos::mdspan<T, Kokkos::extents<index_type, E, Es...>, Rest...> s
)
{
  for(index_type i = 0; i < s.extent(0); ++i) {
    _impl::_do_sum_submdspan(sum, Kokkos::submdspan(
      s, i, _repeated_with<decltype(Es)>(Kokkos::full_extent)...)
    );
  }
}

} // end namespace _impl

template <class MDSpan, class... DynSizes>
void BM_MDSpan_Sum_Subspan_MD_right(benchmark::State& state, MDSpan, DynSizes... dyn) {
  using value_type = typename MDSpan::value_type;
  auto buffer = std::make_unique<value_type[]>(
    MDSpan{nullptr, dyn...}.mapping().required_span_size()
  );
  auto s = MDSpan{buffer.get(), dyn...};
  mdspan_benchmark::fill_random(s);
  for (auto _ : state) {
    value_type sum = 0;
    _impl::_do_sum_submdspan(sum, s);
    benchmark::DoNotOptimize(sum);
    auto sdata = s.data_handle();
    benchmark::DoNotOptimize(sdata);
  }
  state.SetBytesProcessed(s.size() * sizeof(value_type) * state.iterations());
}

BENCHMARK_CAPTURE(
  BM_MDSpan_Sum_Subspan_MD_right, fixed_10D_size_1024,
  Kokkos::mdspan<int, Kokkos::extents<index_type, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2>>{nullptr}
);
BENCHMARK_CAPTURE(
  BM_MDSpan_Sum_Subspan_MD_right, dyn_10D_size_1024,
  Kokkos::mdspan<int, Kokkos::dextents<index_type, 10>>{},
  2, 2, 2, 2, 2,
  2, 2, 2, 2, 2
);
BENCHMARK_CAPTURE(
  BM_MDSpan_Sum_Subspan_MD_right, fixed5_dyn5_10D_alternate_size_1024,
  Kokkos::mdspan<int, Kokkos::extents<index_type, 2, Kokkos::dynamic_extent, 2, Kokkos::dynamic_extent, 2, Kokkos::dynamic_extent, 2, Kokkos::dynamic_extent, 2, Kokkos::dynamic_extent>>{},
  2, 2, 2, 2, 2
);
BENCHMARK_CAPTURE(
  BM_MDSpan_Sum_Subspan_MD_right, fixed8_dyn2_10D_alternate_size_1024,
  Kokkos::mdspan<int, Kokkos::extents<index_type, 2, 2, 2, 2, Kokkos::dynamic_extent, 2, 2, 2, 2, Kokkos::dynamic_extent>>{},
  2, 2
);
BENCHMARK_CAPTURE(
  BM_MDSpan_Sum_Subspan_MD_right, fixed_5D_size_1024,
  Kokkos::mdspan<int, Kokkos::extents<index_type, 4, 4, 4, 4, 4>>{nullptr}
);
BENCHMARK_CAPTURE(
  BM_MDSpan_Sum_Subspan_MD_right, dyn_5D_size_1024,
  Kokkos::mdspan<int, Kokkos::dextents<index_type, 5>>{},
  4, 4, 4, 4, 4
);
BENCHMARK_CAPTURE(
  BM_Raw_Sum_1D, size_1024, int(), 1024
);

//================================================================================

BENCHMARK_MAIN();
