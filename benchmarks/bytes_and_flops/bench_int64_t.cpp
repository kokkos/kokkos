// Copyright Contributors to the Kokkos project
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "bench.hpp"

template void run_stride_unroll<int64_t>(int N, int K, int R, int D, int U,
                                         int F, int T, int S, int B, int I);
