// Copyright Contributors to the Kokkos project
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Intentionally provides no Kokkos profiling interface symbols.
// Used to test that loading a library with no callbacks produces an error.

extern "C" {
void not_a_profiling_callback() {}
}
