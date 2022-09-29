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

template <typename KeyView>
struct KeyFromView {
  using key_value_type = typename KeyView::value_type;
  static constexpr std::uint32_t num_bits = sizeof(key_value_type) * 8;
  
  KeyView keys;
  KeyFromView(KeyView k) : keys(k) {}

  // i: index of the key to get
  // bit: which bit, with 0 indicating the least-significant
  auto operator()(int i, std::uint32_t bit) const {
    auto h    = keys(i) >> bit;

    // Handle signed 2's-complement
    if constexpr(std::is_signed_v<key_value_type>) {
      if (bit == num_bits - 1) {
        h = ~h;
      }
    }
    return ~h & 0x1;
  }
};
  
template <typename T, typename IndexType = ::std::uint32_t>
class RadixSorter {
 public:

  static_assert(std::is_integral_v<T>, "Keys must be integral for now");
  
  static constexpr std::uint32_t num_bits = sizeof(T) * 8;

  RadixSorter() = default;
  explicit RadixSorter(std::size_t n)
      : m_key_scratch("radix_sort_key_scratch", n),
        m_index_old("radix_sort_index", n),
        m_index_new("radix_sort_index_scratch", n),
        m_scan("radix_sort_scan", n),
        m_bits("radix_sort_bits", n) {
    Kokkos::parallel_for(n, KOKKOS_LAMBDA(int i) { m_index_old(i) = i; });
  }

  template <class ExecutionSpace>
  void create_indirection_vector(ExecutionSpace const& exec, View<T*> keys) {
    const auto n = keys.extent(0);
    using std::swap;

    auto key_functor = KeyFromView{keys};
    
    for (int i = 0; i < num_bits; ++i) {
      step<false>(exec, n, key_functor, m_index_old, i);

      swap(m_index_new, m_index_old);
      // Number of bits is always even, and we know on odd numbered
      // iterations we are reading from m_key_scratch and writing to keys
      // So when this loop ends, keys and m_index_old will contain the results
    }

    // Prepare to lift restriction that num_bits must be even
    if (num_bits % 2 == 1) {
      swap(m_index_new, m_index_old);
    }
  }

  template <class ExecutionSpace>
  void apply_permutation(ExecutionSpace const& exec, View<T*> v) {
    parallel_for(RangePolicy<ExecutionSpace>(exec, 0, v.extent(0)),
                 KOKKOS_LAMBDA(int i) { m_key_scratch(i) = v(m_index_old(i)); });
    deep_copy(exec, v, m_key_scratch);
  }

  // 
  template <class ExecutionSpace>
  void sort(ExecutionSpace const& exec, View<T*> keys) {
    // Almost identical to create_indirection_array, except actually permute the input
    const auto n = keys.extent(0);
    using std::swap;

    for (int i = 0; i < num_bits; ++i) {
      auto key_functor = KeyFromView{keys};
    
      step<true>(exec, n, key_functor, m_index_old, i);
      permute_by_scan(exec, n, m_key_scratch, keys);

      swap(m_index_new, m_index_old);
      swap(m_key_scratch, keys);
      // Number of bits is always even, and we know on odd numbered
      // iterations we are reading from m_key_scratch and writing to keys
      // So when this loop ends, keys and m_index_old will contain the results
    }

    // Prepare to lift restriction that num_bits must be even
    if (num_bits % 2 == 1) {
      swap(m_index_new, m_index_old);
    }    
  }
  
  private:

  template <class ExecutionSpace, class U>
  void permute_by_scan(ExecutionSpace const& exec, size_t n, View<U*> out, View<U*> in) {
    RangePolicy<ExecutionSpace> policy(exec, 0, n);

    parallel_for(policy, KOKKOS_LAMBDA(int i) {
        const auto total = m_scan(n - 1) + m_bits(n - 1);
        auto t                   = i - m_scan(i) + total;
        auto new_idx             = m_bits(i) ? m_scan(i) : t;
        out(new_idx) = in(i);
      });
  }

  template <bool permute_input, class ExecutionSpace, class KeyFunctor>
  void step(ExecutionSpace const& exec, size_t n, KeyFunctor getKeyBit,
            View<IndexType*> indices, std::uint32_t shift) {
    RangePolicy<ExecutionSpace> policy(exec, 0, n);

    parallel_for(policy, KOKKOS_LAMBDA(int i) {
      auto key_bit = getKeyBit(permute_input ? i : indices(i), shift);

      m_bits(i) = key_bit;
      m_scan(i) = m_bits(i);
    });

    parallel_scan(policy, KOKKOS_LAMBDA ( const int &i, T &_x, const bool &_final ){
      auto val = m_scan( i );

      if ( _final )
        m_scan( i ) = _x;

      _x += val;
    } );

    permute_by_scan(exec, n, m_index_new, indices);
  }

  View<T*> m_key_scratch;
  View<IndexType*> m_index_old;
  View<IndexType*> m_index_new;
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
