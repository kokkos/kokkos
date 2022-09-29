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

#ifndef KOKKOS_RADIXSORT_HPP_
#define KOKKOS_RADIXSORT_HPP_
#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_RADIXSORT
#endif

#include <Kokkos_Core.hpp>

namespace Kokkos {
namespace Experimental {

// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda

template <typename T, typename IndexType = ::std::uint32_t>
class RadixSorter {
 public:

  static_assert(std::is_integral_v<T>, "Keys must be integral for now");
  
  static constexpr std::uint32_t num_bits = sizeof(T) * 8;

  RadixSorter() = default;
  explicit RadixSorter(std::size_t n)
      : m_key_scratch("radix_sort_key_scratch", n),
        m_index("radix_sort_index", n),
        m_index_scratch("radix_sort_index_scratch", n),
        m_scan("radix_sort_scan", n),
        m_bits("radix_sort_bits", n) {
    Kokkos::parallel_for(n, KOKKOS_LAMBDA(int i) { m_index(i) = i; });
  }

  template <class ExecutionSpace>
  void create_indirection_vector(const ExecutionSpace& exec, View<T*> keys) {
    using std::swap;
    
    for (int i = 0; i < num_bits; ++i) {
      step(exec, keys, m_index, i);

      swap(m_key_scratch, keys);
      swap(m_index_scratch, m_index);
      // Number of bits is always even, and we know on odd numbered
      // iterations we are reading from m_key_scratch and writing to keys
      // So when this loop ends, keys and m_index will contain the results
    }

    // Prepare to lift restriction that num_bits must be even
    if (num_bits % 2 == 1) {
      swap(m_key_scratch, keys);
      swap(m_index_scratch, m_index);
    }
  }

  //private:
  template <class ExecutionSpace>
  void step(const ExecutionSpace& exec, View<T*> keys, View<IndexType*> indices, std::uint32_t shift) {
    const auto n = keys.extent(0);
    RangePolicy<ExecutionSpace> policy(exec, 0, n);
    parallel_for(policy, [this, keys, shift] KOKKOS_FUNCTION(int i) {
      auto h    = keys(i) >> shift;

      // Handle signed 2's-complement
      if constexpr(std::is_signed_v<T>) {
        if (shift == num_bits - 1) {
          h = ~h;
        }
      }
      
      m_bits(i) = ~h & 0x1;
      m_scan(i) = m_bits(i);
    });

    exec.fence();

    parallel_scan(policy, KOKKOS_LAMBDA ( const int &i, T &_x, const bool &_final ){
      auto val = m_scan( i );

      if ( _final )
        m_scan( i ) = _x;

      _x += val;
    } );
    exec.fence();

    parallel_for(policy,
                         [this, keys, indices, n] KOKKOS_FUNCTION(int i) {
                           const auto total = m_scan(n - 1) + m_bits(n - 1);

                           auto t                   = i - m_scan(i) + total;
                           auto new_idx             = m_bits(i) ? m_scan(i) : t;
                           m_index_scratch(new_idx) = indices(i);
                           m_key_scratch(new_idx)   = keys(i);
                         });
    exec.fence();
  }

  View<T*> m_key_scratch;
  View<IndexType*> m_index;
  View<IndexType*> m_index_scratch;
  View<T*> m_scan;
  View<T*> m_bits;
};

}  // namespace Experimental
}  // namespace Kokkos

#ifdef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_RADIXSORT
#undef KOKKOS_IMPL_PUBLIC_INCLUDE
#undef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_RADIXSORT
#endif
#endif
