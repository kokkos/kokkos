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

namespace Kokkos {
namespace Experimental {

template <typename T>
class MaybeReferenceCountedPtr {
 protected:
  MaybeReferenceCountedPtr(T* value_ptr)
      : m_value_ptr(value_ptr), m_control(nullptr) {}

  template <class Deleter>
  MaybeReferenceCountedPtr(T* value_ptr, Deleter deleter)
      : m_value_ptr(value_ptr), m_control(new Control{std::move(deleter), 1}) {}

 public:
  KOKKOS_FUNCTION MaybeReferenceCountedPtr(
      MaybeReferenceCountedPtr&& other) noexcept
      : m_value_ptr(other.m_value_ptr), m_control(other.m_control) {
    other.m_value_ptr = nullptr;
    other.m_control   = nullptr;
  }

  KOKKOS_FUNCTION MaybeReferenceCountedPtr(
      const MaybeReferenceCountedPtr& other) noexcept
      : m_value_ptr(other.m_value_ptr), m_control(other.m_control) {
#ifdef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
    if (m_control) Kokkos::atomic_add(&(m_control->m_counter), 1);
#endif
  }

  KOKKOS_FUNCTION MaybeReferenceCountedPtr& operator=(
      MaybeReferenceCountedPtr&& other) noexcept {
    if (&other != this) {
      cleanup();
      m_value_ptr       = other.m_value_ptr;
      other.m_value_ptr = nullptr;
      m_control         = other.m_control;
      other.m_control   = nullptr;
    }
    return *this;
  }

  KOKKOS_FUNCTION MaybeReferenceCountedPtr& operator=(
      const MaybeReferenceCountedPtr& other) noexcept {
    if (&other != this) {
      cleanup();
      m_value_ptr = other.m_value_ptr;
      m_control   = other.m_control;
#ifdef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
      if (is_reference_counted())
        Kokkos::atomic_add(&(m_control->m_counter), 1);
#endif
    }
    return *this;
  }

  KOKKOS_FUNCTION ~MaybeReferenceCountedPtr() { cleanup(); }

  KOKKOS_FUNCTION T* get() const noexcept { return m_value_ptr; }
  KOKKOS_FUNCTION T& operator*() const noexcept {
    KOKKOS_EXPECTS(bool(*this));
    return *get();
  }
  KOKKOS_FUNCTION T* operator->() const noexcept { return get(); }

  // checks if the stored pointer is not null
  KOKKOS_FUNCTION explicit operator bool() const noexcept {
    return get() != nullptr;
  }

  // checks whether the MaybeReferenceCountedPtr does reference counting
  // which implies managing the lifetime of the object
  KOKKOS_FUNCTION bool is_reference_counted() const noexcept {
    return m_control != nullptr;
  }

 private:
  KOKKOS_FUNCTION void cleanup() noexcept {
#ifdef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
    // If m_counter is set, then this instance is responsible for managing the
    // objects pointed to by m_counter and m_value_ptr.
    if (is_reference_counted()) {
      KOKKOS_EXPECTS(m_control->m_deleter);
      int const count = Kokkos::atomic_fetch_sub(&(m_control->m_counter), 1);
      if (count == 1) {
        (m_control->m_deleter)(m_value_ptr);
        m_value_ptr = nullptr;
        delete m_control;
        m_control = nullptr;
      }
    }
#endif
  }

  struct Control {
    std::function<void(T*)> m_deleter;
    int m_counter;
  };

  T* m_value_ptr;

 protected:
  Control* m_control;
};

template <class T>
class HostSharedPtr : public MaybeReferenceCountedPtr<T> {
 public:
  explicit HostSharedPtr(T* value_ptr = nullptr)
      : MaybeReferenceCountedPtr<T>(value_ptr, [](T* const t) { delete t; }) {}

  template <class Deleter>
  explicit HostSharedPtr(
      T* value_ptr = nullptr, Deleter deleter = [](T* const t) { delete t; })
      : MaybeReferenceCountedPtr<T>(value_ptr, std::move(deleter)) {}

  int use_count() const noexcept {
    return this->m_control ? this->m_control->m_counter : 0;
  }
};

template <class T>
class UnmanagedPtr : public MaybeReferenceCountedPtr<T> {
 public:
  explicit UnmanagedPtr(T* value_ptr = nullptr)
      : MaybeReferenceCountedPtr<T>(value_ptr) {}
};
}  // namespace Experimental
}  // namespace Kokkos

#endif
