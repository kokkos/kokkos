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

#ifndef KOKKOS_SYCL_INSTANCE_HPP_
#define KOKKOS_SYCL_INSTANCE_HPP_

#include <memory>
#include <CL/sycl.hpp>

namespace Kokkos {
namespace Experimental {
namespace Impl {

class SYCLInternal {
 public:
  using size_type = int;

  SYCLInternal() = default;
  ~SYCLInternal();

  SYCLInternal(const SYCLInternal&) = delete;
  SYCLInternal& operator=(const SYCLInternal&) = delete;
  SYCLInternal& operator=(SYCLInternal&&) = delete;
  SYCLInternal(SYCLInternal&&)            = delete;

  int m_syclDev             = -1;
  size_type* m_scratchSpace = nullptr;
  size_type* m_scratchFlags = nullptr;

  std::unique_ptr<sycl::queue> m_queue;

  // USMObjectMem is a reusable buffer for a single object
  // in USM memory
  template <sycl::usm::alloc Kind>
  class USMObjectMem {
   public:
    class Deleter {
     public:
      Deleter() = default;
      explicit Deleter(USMObjectMem* mem) : m_mem(mem) {}

      template <typename T>
      void operator()(T* p) const noexcept {
        assert(m_mem);
        assert(p == m_mem->m_data);
        assert(sizeof(T) == m_mem->m_size);

        if constexpr (sycl::usm::alloc::device == kind)
          // Only skipping the detor on trivially copyable types
          static_assert(std::is_trivially_copyable_v<T>);
        else
          p->~T();

        m_mem->m_size = 0;
      }

     private:
      USMObjectMem* m_mem = nullptr;
    };

    static constexpr sycl::usm::alloc kind = Kind;

    friend void swap(USMObjectMem& lhs, USMObjectMem& rhs) noexcept {
      assert(!lhs.m_size);
      assert(!rhs.m_size);

      using std::swap;
      //swap(lhs.m_q, rhs.m_q);
      swap(lhs.m_data, rhs.m_data);
      swap(lhs.m_size, rhs.m_size);
      swap(lhs.m_capacity, rhs.m_capacity);
    }

    USMObjectMem()                    = default;
    USMObjectMem(USMObjectMem const&) = delete;
    USMObjectMem& operator=(USMObjectMem const&) = delete;

    USMObjectMem(USMObjectMem&& that) noexcept : USMObjectMem() {
//	    swap(*this, that);
      m_q = that.m_q;
      m_data = that.m_data;
      m_size = that.m_size;
      m_capacity = that.m_capacity;
      
      that.m_data = nullptr;
      that.m_size = 0;
      that.m_capacity = 0;
    }

    USMObjectMem& operator=(USMObjectMem&& that) noexcept {
/*      m_q = that.m_q;
      m_data = that.m_data;
      m_size = that.m_size;
      m_capacity = that.m_capacity;

      that.m_data = nullptr;
      that.m_size = 0;
      that.m_capacity = 0;*/

      swap(*this, that);

      return *this;
    }

    ~USMObjectMem() {
      assert(!m_size);

      sycl::free(m_data, m_q);
    }

    explicit USMObjectMem(sycl::queue q) noexcept : m_q(std::move(q)) {}

    sycl::queue queue() const noexcept { return m_q; }

    void* data() noexcept { return m_data; }
    const void* data() const noexcept { return m_data; }

    size_t size() const noexcept { return m_size; }
    size_t capacity() const noexcept { return m_capacity; }

    // reserve() allocates space for at least n bytes
    // returns the new capacity
    size_t reserve(size_t n) {
      assert(!m_size);

      if (m_capacity < n) {
        // First free what we have (in case malloc can reuse it)
        sycl::free(m_data, m_q);

        m_data = sycl::malloc(n, m_q, kind);
        if (!m_data) {
          m_capacity = 0;
          throw std::bad_alloc();
        }

        m_capacity = n;
      }

      return m_capacity;
    }

   private:
    // This will memcpy an object T into memory held by this object
    // returns: a T* to that object
    //
    // Note:  it is UB to dereference this pointer with an object that is
    // not an implicit-lifetime nor trivially-copyable type, but presumably much
    // faster because we can use USM device memory
    template <typename T>
    std::unique_ptr<T, Deleter> memcpy_from(const T& t) {
      reserve(sizeof(T));
      sycl::event memcopied = m_q.memcpy(m_data, std::addressof(t), sizeof(T));
      memcopied.wait();

      std::unique_ptr<T, Deleter> ptr(reinterpret_cast<T*>(m_data),
                                      Deleter(this));
      m_size = sizeof(T);
      return ptr;
    }

    // This will copy-constuct an object T into memory held by this object
    // returns: a unique_ptr<T, destruct_delete> that will call the
    // destructor on the type when it goes out of scope.
    //
    // Note:  This will not work with USM device memory
    template <typename T>
    std::unique_ptr<T, Deleter> copy_construct_from(const T& t) {
      static_assert(kind != sycl::usm::alloc::device,
                    "Cannot copy construct into USM device memory");

      reserve(sizeof(T));

      std::unique_ptr<T, Deleter> ptr(new (m_data) T(t), Deleter(this));
      m_size = sizeof(T);
      return ptr;
    }

   public:
    // Performs either memcpy (for USM device memory) and returns a T*
    // (but is technically UB when dereferenced on an object that is not
    // an implicit-lifetime nor trivially-copyable type
    //
    // or
    //
    // performs copy construction (for other USM memory types) and returns a
    // unique_ptr<T, ...>
    template <typename T>
    std::unique_ptr<T, Deleter> copy_from(const T& t) {
      if constexpr (sycl::usm::alloc::device == kind)
        return memcpy_from(t);
      else
        return copy_construct_from(t);
    }

   private:
    template <typename T>
    T& memcpy_to(T& t) {
      assert(sizeof(T) == m_size);

      sycl::event memcopied = m_q.memcpy(std::addressof(t), m_data, sizeof(T));
      memcopied.wait();

      return t;
    }

    template <typename T>
    T& move_assign_to(T& t) {
      static_assert(kind != sycl::usm::alloc::device,
                    "Cannot move_assign_to from USM device memory");

      assert(sizeof(T) == m_size);

      t = std::move(*static_cast<T*>(m_data));

      return t;
    }

   public:
    template <typename T>
    T& transfer_to(T& t) {
      if constexpr (sycl::usm::alloc::device == kind)
        return memcpy_to(t);
      else
        return move_assign_to(t);
    }

   private:
    sycl::queue m_q;
    void* m_data      = nullptr;
    size_t m_size     = 0;
    size_t m_capacity = 0;
  };

  // An indirect kernel is one where the functor to be executed is explicitly
  // copied to USM device memory before being executed, to get around the
  // trivially copyable limitation of SYCL.
  using IndirectKernelMem = USMObjectMem<sycl::usm::alloc::shared>;
  IndirectKernelMem m_indirectKernelMem;

  using ReductionResultMem = USMObjectMem<sycl::usm::alloc::shared>;
  ReductionResultMem m_reductionResultMem;

  static int was_finalized;

  static SYCLInternal& singleton();

  int verify_is_initialized(const char* const label) const;

  void initialize(const sycl::device& d);

  int is_initialized() const { return m_queue != nullptr; }

  void finalize();
};

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos
#endif
