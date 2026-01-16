// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_NEXTSILICON_DEEP_COPY_HPP
#define KOKKOS_NEXTSILICON_DEEP_COPY_HPP

#include <NextSilicon/Kokkos_NextSilicon.hpp>
#include <NextSilicon/Kokkos_NextSiliconSpace.hpp>

#include <Kokkos_Concepts.hpp>

namespace Kokkos {
namespace Impl {

void DeepCopySharedNextSilicon(void* dst, const void* src, size_t n);
void DeepCopyDeviceNextSilicon(void* dst, const void* src, size_t n);

template <>
struct DeepCopy<Kokkos::Experimental::NextSiliconSpace,
                Kokkos::Experimental::NextSiliconSpace,
                Kokkos::Experimental::NextSilicon> {
  DeepCopy(void* dst, const void* src, size_t n) {
    DeepCopyDeviceNextSilicon(dst, src, n);
  }
  DeepCopy(const Kokkos::Experimental::NextSilicon&, void* dst, const void* src,
           size_t n) {
    DeepCopyDeviceNextSilicon(dst, src, n);
  }
};

template <class ExecutionSpace>
struct DeepCopy<Kokkos::Experimental::NextSiliconSpace,
                Kokkos::Experimental::NextSiliconSpace, ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
    DeepCopyDeviceNextSilicon(dst, src, n);
  }
  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence(
        "Kokkos::Impl::DeepCopy<NextSiliconSpace, NextSiliconSpace, "
        "ExecutionSpace>::DeepCopy: fence before copy");
    DeepCopyDeviceNextSilicon(dst, src, n);
  }
};

template <>
struct DeepCopy<Kokkos::Experimental::NextSiliconSpace, Kokkos::HostSpace,
                Kokkos::Experimental::NextSilicon> {
  DeepCopy(void* dst, const void* src, size_t n) {
    DeepCopyDeviceNextSilicon(dst, src, n);
  }
  DeepCopy(const Kokkos::Experimental::NextSilicon&, void* dst, const void* src,
           size_t n) {
    DeepCopyDeviceNextSilicon(dst, src, n);
  }
};

template <class ExecutionSpace>
struct DeepCopy<Kokkos::Experimental::NextSiliconSpace, Kokkos::HostSpace,
                ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
    DeepCopyDeviceNextSilicon(dst, src, n);
  }
  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence(
        "Kokkos::Impl::DeepCopy<NextSiliconSpace, HostSpace, "
        "ExecutionSpace>::DeepCopy: fence before copy");
    DeepCopyDeviceNextSilicon(dst, src, n);
  }
};

template <>
struct DeepCopy<Kokkos::HostSpace, Kokkos::Experimental::NextSiliconSpace,
                Kokkos::Experimental::NextSilicon> {
  DeepCopy(void* dst, const void* src, size_t n) {
    DeepCopyDeviceNextSilicon(dst, src, n);
  }
  DeepCopy(const Kokkos::Experimental::NextSilicon&, void* dst, const void* src,
           size_t n) {
    DeepCopyDeviceNextSilicon(dst, src, n);
  }
};

template <class ExecutionSpace>
struct DeepCopy<Kokkos::HostSpace, Kokkos::Experimental::NextSiliconSpace,
                ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
    DeepCopySharedNextSilicon(dst, src, n);
  }
  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence(
        "Kokkos::Impl::DeepCopy<HostSpace, NextSiliconSpace, "
        "ExecutionSpace>::DeepCopy: fence before copy");
    DeepCopySharedNextSilicon(dst, src, n);
  }
};

template <>
struct DeepCopy<Kokkos::Experimental::NextSiliconSharedSpace,
                Kokkos::Experimental::NextSiliconSharedSpace,
                Kokkos::Experimental::NextSilicon> {
  DeepCopy(void* dst, const void* src, size_t n) {
    DeepCopySharedNextSilicon(dst, src, n);
  }
  DeepCopy(const Kokkos::Experimental::NextSilicon&, void* dst, const void* src,
           size_t n) {
    DeepCopySharedNextSilicon(dst, src, n);
  }
};

template <class ExecutionSpace>
struct DeepCopy<Kokkos::Experimental::NextSiliconSharedSpace,
                Kokkos::Experimental::NextSiliconSharedSpace, ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
    DeepCopySharedNextSilicon(dst, src, n);
  }
  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence(
        "Kokkos::Impl::DeepCopy<NextSiliconSpace, NextSiliconSpace, "
        "ExecutionSpace>::DeepCopy: fence before copy");
    DeepCopySharedNextSilicon(dst, src, n);
  }
};

template <>
struct DeepCopy<Kokkos::Experimental::NextSiliconSharedSpace, Kokkos::HostSpace,
                Kokkos::Experimental::NextSilicon> {
  DeepCopy(void* dst, const void* src, size_t n) {
    DeepCopySharedNextSilicon(dst, src, n);
  }
  DeepCopy(const Kokkos::Experimental::NextSilicon&, void* dst, const void* src,
           size_t n) {
    DeepCopySharedNextSilicon(dst, src, n);
  }
};

template <class ExecutionSpace>
struct DeepCopy<Kokkos::Experimental::NextSiliconSharedSpace, Kokkos::HostSpace,
                ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
    DeepCopySharedNextSilicon(dst, src, n);
  }
  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence(
        "Kokkos::Impl::DeepCopy<NextSiliconSpace, HostSpace, "
        "ExecutionSpace>::DeepCopy: fence before copy");
    DeepCopySharedNextSilicon(dst, src, n);
  }
};

template <>
struct DeepCopy<Kokkos::HostSpace, Kokkos::Experimental::NextSiliconSharedSpace,
                Kokkos::Experimental::NextSilicon> {
  DeepCopy(void* dst, const void* src, size_t n) {
    DeepCopySharedNextSilicon(dst, src, n);
  }
  DeepCopy(const Kokkos::Experimental::NextSilicon&, void* dst, const void* src,
           size_t n) {
    DeepCopySharedNextSilicon(dst, src, n);
  }
};

template <class ExecutionSpace>
struct DeepCopy<Kokkos::HostSpace, Kokkos::Experimental::NextSiliconSharedSpace,
                ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
    DeepCopySharedNextSilicon(dst, src, n);
  }
  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence(
        "Kokkos::Impl::DeepCopy<HostSpace, NextSiliconSpace, "
        "ExecutionSpace>::DeepCopy: fence before copy");
    DeepCopySharedNextSilicon(dst, src, n);
  }
};

template <class ExecutionSpace>
struct DeepCopy<Kokkos::Experimental::NextSiliconSpace,
                Kokkos::Experimental::NextSiliconSharedSpace, ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
    DeepCopySharedNextSilicon(dst, src, n);
  }
  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence(
        "Kokkos::Impl::DeepCopy<NextSiliconSpace, NextSiliconSharedSpace, "
        "ExecutionSpace>::DeepCopy: fence before copy");
    DeepCopySharedNextSilicon(dst, src, n);
  }
};

}  // namespace Impl
}  // namespace Kokkos

#endif
