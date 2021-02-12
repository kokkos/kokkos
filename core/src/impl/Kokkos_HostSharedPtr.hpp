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

#ifndef KOKKOS_IMPL_HOST_SHARED_PTR_HPP
#define KOKKOS_IMPL_HOST_SHARED_PTR_HPP

#include <Kokkos_Macros.hpp>
#include <Kokkos_Atomic.hpp>

#include <functional>

template <typename T>
class HostSharedPtr {
 public:
  HostSharedPtr(T* value_ptr, bool owning)
      : m_value_ptr(value_ptr), m_counter(owning ? (new int(1)) : nullptr) {}

  KOKKOS_FUNCTION HostSharedPtr(HostSharedPtr&& other) noexcept
      : m_value_ptr(other.m_value_ptr), m_counter(other.m_counter) {
    other.m_value_ptr = nullptr;
    other.m_counter   = nullptr;
  }

  KOKKOS_FUNCTION HostSharedPtr(const HostSharedPtr& other)
      : m_value_ptr(other.m_value_ptr), m_counter(other.m_counter) {
#ifdef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
    if (m_counter) Kokkos::atomic_add(m_counter, 1);
#endif
  }

  KOKKOS_FUNCTION HostSharedPtr& operator=(HostSharedPtr&& other) noexcept {
    if (&other != this) {
      cleanup();
      m_value_ptr       = other.m_value_ptr;
      other.m_value_ptr = nullptr;
      m_counter         = other.m_counter;
      other.m_counter   = nullptr;
    }
    return *this;
  }

  KOKKOS_FUNCTION HostSharedPtr& operator=(const HostSharedPtr& other) {
    if (&other != this) {
      cleanup();
      m_value_ptr = other.m_value_ptr;
      m_counter   = other.m_counter;
#ifdef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
      if (m_counter) Kokkos::atomic_add(m_counter, 1);
#endif
    }
    return *this;
  }

  KOKKOS_FUNCTION ~HostSharedPtr() { cleanup(); }

  KOKKOS_FUNCTION T* get() const noexcept { return m_value_ptr; }
  KOKKOS_FUNCTION T& operator*() const noexcept { return *get(); }
  KOKKOS_FUNCTION T* operator->() const noexcept { return get(); }

 private:
  KOKKOS_FUNCTION void cleanup() noexcept {
#ifdef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
    // If m_counter is set, then this instance is responsible for managing the
    // objects pointed to by m_counter and m_value_ptr.
    if (m_counter == nullptr) return;
    int const count = Kokkos::atomic_fetch_sub(m_counter, 1);
    if (count == 1) {
      delete m_counter;
      m_value_ptr->finalize();
      delete m_value_ptr;
    }
#endif
  }

  T* m_value_ptr;
  int* m_counter;
};

#endif
