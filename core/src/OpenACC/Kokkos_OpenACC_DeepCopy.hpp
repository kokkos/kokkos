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

#ifndef KOKKOS_OPENACC_DEEP_COPY_HPP
#define KOKKOS_OPENACC_DEEP_COPY_HPP

#include <OpenACC/Kokkos_OpenACC.hpp>
#include <OpenACC/Kokkos_OpenACCSpace.hpp>

#include <Kokkos_Concepts.hpp>

#include <openacc.h>

struct Kokkos::Impl::DeepCopy<Kokkos::Experimental::OpenACCSpace,
                              Kokkos::Experimental::OpenACCSpace,
                              Kokkos::Experimental::OpenACC> {
  DeepCopy(void* dst, const void* src, size_t n) {
    // The behavior of acc_memcpy_device when bytes argument is zero is
    // clarified only in the latest OpenACC specification (V3.2), and thus the
    // value checking is added as a safeguard. (The current NVHPC (V22.5)
    // supports OpenACC V2.7.)
    if (n > 0) {
      acc_memcpy_device(dst, const_cast<void*>(src), n);
    }
  }
  DeepCopy(const Kokkos::Experimental::OpenACC& exec, void* dst,
           const void* src, size_t n) {
    if (n > 0) {
      acc_memcpy_device_async(dst, const_cast<void*>(src), n,
                              exec.acc_async_queue());
    }
  }
};

template <class ExecutionSpace>
struct Kokkos::Impl::DeepCopy<Kokkos::Experimental::OpenACCSpace,
                              Kokkos::Experimental::OpenACCSpace,
                              ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
    if (n > 0) {
      acc_memcpy_device(dst, const_cast<void*>(src), n);
    }
  }
  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence(
        "Kokkos::Impl::DeepCopy<OpenACCSpace, OpenACCSpace, "
        "ExecutionSpace>::DeepCopy: fence before copy");
    if (n > 0) {
      acc_memcpy_device_async(dst, const_cast<void*>(src), n,
                              exec.acc_async_queue());
    }
  }
};

struct Kokkos::Impl::DeepCopy<Kokkos::Experimental::OpenACCSpace,
                              Kokkos::HostSpace,
                              Kokkos::Experimental::OpenACC> {
  DeepCopy(void* dst, const void* src, size_t n) {
    if (n > 0) acc_memcpy_to_device(dst, const_cast<void*>(src), n);
  }
  DeepCopy(const Kokkos::Experimental::OpenACC& exec, void* dst,
           const void* src, size_t n) {
    if (n > 0)
      acc_memcpy_to_device_async(dst, const_cast<void*>(src), n,
                                 exec.acc_async_queue());
  }
};

template <class ExecutionSpace>
struct Kokkos::Impl::DeepCopy<Kokkos::Experimental::OpenACCSpace,
                              Kokkos::HostSpace, ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
    if (n > 0) {
      acc_memcpy_to_device(dst, const_cast<void*>(src), n);
    }
  }
  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence(
        "Kokkos::Impl::DeepCopy<OpenACCSpace, HostSpace, "
        "ExecutionSpace>::DeepCopy: fence before copy");
    if (n > 0) {
      acc_memcpy_to_device_async(dst, const_cast<void*>(src), n,
                                 exec.acc_async_queue());
    }
  }
};

struct Kokkos::Impl::DeepCopy<Kokkos::HostSpace,
                              Kokkos::Experimental::OpenACCSpace,
                              Kokkos::Experimental::OpenACC> {
  DeepCopy(void* dst, const void* src, size_t n) {
    if (n > 0) {
      acc_memcpy_from_device(dst, const_cast<void*>(src), n);
    }
  }
  DeepCopy(const Kokkos::Experimental::OpenACC& exec, void* dst,
           const void* src, size_t n) {
    if (n > 0) {
      acc_memcpy_from_device_async(dst, const_cast<void*>(src), n,
                                   exec.acc_async_queue());
    }
  }
};

template <class ExecutionSpace>
struct Kokkos::Impl::DeepCopy<
    Kokkos::HostSpace, Kokkos::Experimental::OpenACCSpace, ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
    if (n > 0) acc_memcpy_from_device_async(dst, const_cast<void*>(src), n);
  }
  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence(
        "Kokkos::Impl::DeepCopy<HostSpace, OpenACCSpace, "
        "ExecutionSpace>::DeepCopy: fence before copy");
    if (n > 0) {
      acc_memcpy_from_device_async(dst, const_cast<void*>(src), n,
                                   exec.acc_async_queue());
    }
  }
};

#endif
