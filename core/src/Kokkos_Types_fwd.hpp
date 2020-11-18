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

#pragma once

#include "Kokkos_Core_fwd.hpp"

#include <type_traits>

namespace Kokkos {

//--------------------------------------------------------------------------------------//
//  this is used to mark execution or memory spaces as not available
//
template <typename Tp>
struct IsAvailable : std::true_type {};

#define KOKKOS_IMPL_DISABLE_TYPE(TYPE) \
  template <>                          \
  struct IsAvailable<TYPE> : std::false_type {};

//--------------------------------------------------------------------------------------//
//  declare any spaces that might not be available and mark them as unavailable
//
#if !defined(KOKKOS_ENABLE_SERIAL)
class Serial;
KOKKOS_IMPL_DISABLE_TYPE(Serial)
#endif

#if !defined(KOKKOS_ENABLE_THREADS)
class Threads;
class ThreadsExec;
KOKKOS_IMPL_DISABLE_TYPE(Threads)
KOKKOS_IMPL_DISABLE_TYPE(ThreadsExec)
#endif

#if !defined(KOKKOS_ENABLE_OPENMP)
class OpenMP;
class OpenMPExec;
KOKKOS_IMPL_DISABLE_TYPE(OpenMP)
KOKKOS_IMPL_DISABLE_TYPE(OpenMPExec)
#endif

#if !defined(KOKKOS_ENABLE_CUDA)
class Cuda;
class CudaSpace;
KOKKOS_IMPL_DISABLE_TYPE(Cuda)
KOKKOS_IMPL_DISABLE_TYPE(CudaSpace)
#endif

#if !defined(KOKKOS_ENABLE_CUDA_UVM)
class CudaUVMSpace;
KOKKOS_IMPL_DISABLE_TYPE(CudaUVMSpace)
#endif

#ifndef KOKKOS_ENABLE_HBWSPACE
namespace Experimental {
class HBWSpace;
}  // namespace Experimental
KOKKOS_IMPL_DISABLE_TYPE(Experimental::HBWSpace)
#endif

#if !defined(KOKKOS_ENABLE_OPENMPTARGET)
namespace Experimental {
class OpenMPTarget;
class OpenMPTargetSpace;
}  // namespace Experimental
KOKKOS_IMPL_DISABLE_TYPE(Experimental::OpenMPTarget)
KOKKOS_IMPL_DISABLE_TYPE(Experimental::OpenMPTargetSpace)
#endif

#if !defined(KOKKOS_ENABLE_HIP)
namespace Experimental {
class HIP;
class HIPSpace;
}  // namespace Experimental
KOKKOS_IMPL_DISABLE_TYPE(Experimental::HIP)
KOKKOS_IMPL_DISABLE_TYPE(Experimental::HIPSpace)
#endif

#if !defined(KOKKOS_ENABLE_SYCL)
namespace Experimental {
class SYCL;
class SYCLDeviceUSMSpace;
}  // namespace Experimental
KOKKOS_IMPL_DISABLE_TYPE(Experimental::SYCL)
KOKKOS_IMPL_DISABLE_TYPE(Experimental::SYCLDeviceUSMSpace)
#endif

}  // namespace Kokkos

#undef KOKKOS_IMPL_DISABLE_TYPE
