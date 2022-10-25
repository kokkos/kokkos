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

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#endif

#include <HIP/Kokkos_HIP_DeepCopy.hpp>
#include <HIP/Kokkos_HIP_Error.hpp>  // HIP_SAFE_CALL
#include <HIP/Kokkos_HIP.hpp>

namespace Kokkos {
namespace Impl {
namespace {
hipStream_t get_deep_copy_stream() {
  static hipStream_t s = nullptr;
  if (s == nullptr) {
    KOKKOS_IMPL_HIP_SAFE_CALL(hipStreamCreate(&s));
  }
  return s;
}
}  // namespace

void DeepCopyHIP(void* dst, void const* src, size_t n) {
  KOKKOS_IMPL_HIP_SAFE_CALL(hipMemcpyAsync(dst, src, n, hipMemcpyDefault));
}

void DeepCopyAsyncHIP(const HIP& instance, void* dst, void const* src,
                      size_t n) {
  KOKKOS_IMPL_HIP_SAFE_CALL(
      hipMemcpyAsync(dst, src, n, hipMemcpyDefault, instance.hip_stream()));
}

void DeepCopyAsyncHIP(void* dst, void const* src, size_t n) {
  hipStream_t s = get_deep_copy_stream();
  KOKKOS_IMPL_HIP_SAFE_CALL(hipMemcpyAsync(dst, src, n, hipMemcpyDefault, s));
  Kokkos::Tools::Experimental::Impl::profile_fence_event<HIP>(
      "Kokkos::Impl::DeepCopyAsyncHIP: Post Deep Copy Fence on Deep-Copy "
      "stream",
      Kokkos::Tools::Experimental::SpecialSynchronizationCases::
          DeepCopyResourceSynchronization,
      [&]() { KOKKOS_IMPL_HIP_SAFE_CALL(hipStreamSynchronize(s)); });
}
}  // namespace Impl
}  // namespace Kokkos
