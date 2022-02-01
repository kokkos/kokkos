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

#ifndef KOKKOS_HIP_UNIQUE_TOKEN_HPP
#define KOKKOS_HIP_UNIQUE_TOKEN_HPP

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_HIP

#include <Kokkos_HIP_Space.hpp>
#include <Kokkos_UniqueToken.hpp>
#include <impl/Kokkos_SharedAlloc.hpp>

namespace Kokkos {
namespace Experimental {

// both global and instance Unique Tokens are implemented in the same way
template <>
class UniqueToken<HIP, UniqueTokenScope::Global> {
 protected:
  View<uint32_t*, HIPSpace> m_locks;
  uint32_t m_count;

 public:
  using execution_space = HIP;
  using size_type       = int32_t;

  explicit UniqueToken(execution_space const& = execution_space())
      : m_locks(View<uint32_t*, HIPSpace>("Kokkos::UniqueToken::m_locks",
                                          HIP().concurrency())),
        m_count(HIP().concurrency()){};

  KOKKOS_DEFAULTED_FUNCTION
  UniqueToken(const UniqueToken&) = default;

  KOKKOS_DEFAULTED_FUNCTION
  UniqueToken(UniqueToken&&) = default;

  KOKKOS_DEFAULTED_FUNCTION
  UniqueToken& operator=(const UniqueToken&) = default;

  KOKKOS_DEFAULTED_FUNCTION
  UniqueToken& operator=(UniqueToken&&) = default;

  /// \brief upper bound for acquired values, i.e. 0 <= value < size()
  KOKKOS_INLINE_FUNCTION
  size_type size() const noexcept { return m_count; }

  /// \brief acquire value such that 0 <= value < size()
  KOKKOS_INLINE_FUNCTION
  size_type acquire() const {
    KOKKOS_IF_ON_DEVICE(
        int idx = blockIdx.x * (blockDim.x * blockDim.y) +
                  threadIdx.y * blockDim.x + threadIdx.x;
        idx = idx % m_count; unsigned int active = __ballot(1);
        unsigned int done_active = 0; bool done = false;
        while (active != done_active) {
          if (!done) {
#ifdef KOKKOS_ENABLE_IMPL_DESUL_ATOMICS
            desul::atomic_thread_fence(desul::MemoryOrderAcquire(),
                                       desul::MemoryScopeDevice());
#else
            Kokkos::memory_fence();
#endif
            if (Kokkos::atomic_compare_exchange(&m_locks(idx), 0, 1) == 0) {
              done = true;
            } else {
              idx += blockDim.y * blockDim.x + 1;
              idx = idx % m_count;
            }
          }
          done_active = __ballot(done ? 1 : 0);
        } return idx;)
    KOKKOS_IF_ON_HOST(return 0;)
  }

  /// \brief release an acquired value
  KOKKOS_INLINE_FUNCTION
  void release(size_type idx) const noexcept {
#ifdef KOKKOS_ENABLE_IMPL_DESUL_ATOMICS
    desul::atomic_thread_fence(desul::MemoryOrderRelease(),
                               desul::MemoryScopeDevice());
#else
    Kokkos::memory_fence();
#endif
    (void)Kokkos::atomic_exchange(&m_locks(idx), 0);
  }
};

template <>
class UniqueToken<HIP, UniqueTokenScope::Instance>
    : public UniqueToken<HIP, UniqueTokenScope::Global> {
 private:
  View<uint32_t*, HIPSpace> m_buffer_view;

 public:
  explicit UniqueToken(execution_space const& arg = execution_space())
      : UniqueToken<HIP, UniqueTokenScope::Global>(arg) {}

  UniqueToken(size_type max_size, execution_space const& = execution_space()) {
    m_locks =
        View<uint32_t*, HIPSpace>("Kokkos::UniqueToken::m_locks", max_size);
    m_count = max_size;
  }
};

}  // namespace Experimental
}  // namespace Kokkos

#endif  // KOKKOS_ENABLE_HIP
#endif  // KOKKOS_HIP_UNIQUE_TOKEN_HPP
