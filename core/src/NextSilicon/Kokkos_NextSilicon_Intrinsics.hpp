// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_NEXTSILICON_INTRINSICS_HPP
#define KOKKOS_NEXTSILICON_INTRINSICS_HPP

// An internal-use-only intrinsic used to communicate from Kokkos C++ code to
// the MLIR compiler stack that a struct at the provided address can be
// considered immutable and thread invariant for the full duration of the
// microtask, and that consequently all its members are read only, and that the
// pointer to the struct and any derived pointer does not alias with anything
// else (pointers stored in the struct can still alias). This should _only_ be
// used to annotate the functor objects passed into Kokkos parallel constructs,
// and should not be used _anywhere_ outside of Kokkos. The internal nextapi
// header is guarded against accidental inclusion, but we override it here since
// the intrinsic it provides is specifically intended for use in Kokkos.
#define NEXTAPI_INCLUDE_INTERNAL_HEADER
#include <nextapi/internal/immutable_thread_invariant_parameter_struct.hpp>
#undef NEXTAPI_INCLUDE_INTERNAL_HEADER

#endif  // KOKKOS_NEXTSILICON_INTRINSICS_HPP
