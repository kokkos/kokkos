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
 public:
  using element_type = T;

 protected:
  MaybeReferenceCountedPtr(T* element_ptr)
      : m_element_ptr(element_ptr), m_control(nullptr) {}

  template <class Deleter>
  MaybeReferenceCountedPtr(T* element_ptr, const Deleter& deleter)
      : m_element_ptr(element_ptr) {
    try {
      m_control = new Control{deleter, 1};
    } catch (...) {
      deleter(element_ptr);
      throw;
    }
  }

 public:
  KOKKOS_FUNCTION MaybeReferenceCountedPtr(
      MaybeReferenceCountedPtr&& other) noexcept
      : m_element_ptr(other.m_element_ptr), m_control(other.m_control) {
    other.m_element_ptr = nullptr;
    other.m_control     = nullptr;
  }

  KOKKOS_FUNCTION MaybeReferenceCountedPtr(
      const MaybeReferenceCountedPtr& other) noexcept
      : m_element_ptr(other.m_element_ptr), m_control(other.m_control) {
#ifdef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
    if (m_control) Kokkos::atomic_add(&(m_control->m_counter), 1);
#endif
  }

  KOKKOS_FUNCTION MaybeReferenceCountedPtr& operator=(
      MaybeReferenceCountedPtr&& other) noexcept {
    if (&other != this) {
      cleanup();
      m_element_ptr       = other.m_element_ptr;
      other.m_element_ptr = nullptr;
      m_control           = other.m_control;
      other.m_control     = nullptr;
    }
    return *this;
  }

  KOKKOS_FUNCTION MaybeReferenceCountedPtr& operator=(
      const MaybeReferenceCountedPtr& other) noexcept {
    if (&other != this) {
      cleanup();
      m_element_ptr = other.m_element_ptr;
      m_control     = other.m_control;
#ifdef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
      if (is_reference_counted())
        Kokkos::atomic_add(&(m_control->m_counter), 1);
#endif
    }
    return *this;
  }

  KOKKOS_FUNCTION ~MaybeReferenceCountedPtr() { cleanup(); }

  KOKKOS_FUNCTION T* get() const noexcept { return m_element_ptr; }
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
    // objects pointed to by m_counter and m_element_ptr.
    if (is_reference_counted()) {
      int const count = Kokkos::atomic_fetch_sub(&(m_control->m_counter), 1);
      if (count == 1) {
        (m_control->m_deleter)(m_element_ptr);
        m_element_ptr = nullptr;
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

  T* m_element_ptr;

 protected:
  Control* m_control;
};

template <class T>
class HostSharedPtr : public MaybeReferenceCountedPtr<T> {
 public:
  // Objects that are default-constructed or initialized with an (explicit)
  // nullptr are not considered reference-counted.
  HostSharedPtr() noexcept : MaybeReferenceCountedPtr<T>(nullptr) {}
  HostSharedPtr(std::nullptr_t) noexcept : HostSharedPtr() {}

  explicit HostSharedPtr(T* element_ptr)
      : MaybeReferenceCountedPtr<T>(element_ptr, [](T* const t) { delete t; }) {
  }

  template <class Deleter>
  HostSharedPtr(T* element_ptr, const Deleter& deleter)
      : MaybeReferenceCountedPtr<T>(element_ptr, deleter) {}

  int use_count() const noexcept {
    return this->m_control ? this->m_control->m_counter : 0;
  }
};

template <class T>
class UnmanagedPtr : public MaybeReferenceCountedPtr<T> {
 public:
  explicit UnmanagedPtr(T* element_ptr = nullptr) noexcept
      : MaybeReferenceCountedPtr<T>(element_ptr) {}
};
}  // namespace Experimental
}  // namespace Kokkos

#endif
