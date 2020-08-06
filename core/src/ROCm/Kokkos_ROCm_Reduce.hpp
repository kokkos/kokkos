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

///////////////////////////////////////////////////////////////////////////////
// AMP REDUCE
//////////////////////////////////////////////////////////////////////////////

#if !defined(KOKKOS_ROCM_AMP_REDUCE_INL)
#define KOKKOS_ROCM_AMP_REDUCE_INL

#include <algorithm>
#include <numeric>
#include <cmath>
#include <type_traits>
#include <ROCm/Kokkos_ROCm_Tile.hpp>
#include <ROCm/Kokkos_ROCm_Invoke.hpp>
#include <ROCm/Kokkos_ROCm_Join.hpp>
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace Kokkos {
namespace Impl {

template <class T>
T* reduce_value(T* x, std::true_type) [[hc]] {
  return x;
}

template <class T>
T& reduce_value(T* x, std::false_type) [[hc]] {
  return *x;
}

#ifdef KOKKOS_IMPL_ROCM_CLANG_WORKAROUND
struct always_true {
  template <class... Ts>
  bool operator()(Ts&&...) const {
    return true;
  }
};
#endif

template <class Tag, class F, class ReducerType, class Invoker, class T>
void reduce_enqueue(const int szElements,  // size of the extent
                    const F& f, const ReducerType& reducer, Invoker invoke,
                    T* const output_result, int const output_length,
                    const int team_size = 64, const int vector_size = 1,
                    int const shared_size = 0) {
  using namespace hc;

  using ReducerConditional =
      Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value, F,
                         ReducerType>;
  using ReducerTypeFwd = typename ReducerConditional::type;
  using TagFwd =
      typename Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                                  Tag, void>::type;

  using ValueTraits = Kokkos::Impl::FunctorValueTraits<ReducerTypeFwd, TagFwd>;
  using ValueInit   = Kokkos::Impl::FunctorValueInit<ReducerTypeFwd, TagFwd>;
  using ValueJoin   = Kokkos::Impl::FunctorValueJoin<ReducerTypeFwd, TagFwd>;
  using ValueFinal  = Kokkos::Impl::FunctorFinal<ReducerTypeFwd, TagFwd>;

  using pointer_type   = typename ValueTraits::pointer_type;
  using reference_type = typename ValueTraits::reference_type;

  if (output_length < 1) return;

  const auto td = get_tile_desc<T>(szElements, output_length, team_size,
                                   vector_size, shared_size);

  // allocate host and device memory for the results from each team
  std::vector<T> result_cpu(td.num_tiles * output_length);
  hc::array<T> result(td.num_tiles * output_length);

  auto fut = tile_for<T[]>(
      td,
      [ =, &result ](hc::tiled_index<1> t_idx, tile_buffer<T[]> buffer) [[hc]] {
        const auto local  = t_idx.local[0];
        const auto global = t_idx.global[0];
        const auto tile   = t_idx.tile[0];

        buffer.action_at(local, [&](T* state) {
          ValueInit::init(ReducerConditional::select(f, reducer), state);
          invoke(make_rocm_invoke_fn<Tag>(f), t_idx, td,
                 reduce_value(state, std::is_pointer<reference_type>()));
        });
        t_idx.barrier.wait();

        for (std::size_t s = 1; s < buffer.size(); s *= 2) {
          const std::size_t index = 2 * s * local;
          if (index < buffer.size()) {
            buffer.action_at(index, index + s, [&](T* x, T* y) {
              ValueJoin::join(ReducerConditional::select(f, reducer), x, y);
            });
          }
          t_idx.barrier.wait();
        }

        // Store the tile result in the global memory.
        if (local == 0) {
#ifdef KOKKOS_IMPL_ROCM_CLANG_WORKAROUND
          // Workaround for assigning from LDS memory: std::copy should work
          // directly
          buffer.action_at(0, [&](T* x) {
#if ROCM15
            // new ROCM 15 address space changes aren't implemented in std
            // algorithms yet
            auto* src = reinterpret_cast<char*>(x);
            auto* dest =
                reinterpret_cast<char*>(result.data() + tile * output_length);
            for (int i = 0; i < sizeof(T) * output_length; i++)
              dest[i] = src[i];
#else
              // Workaround: copy_if used to avoid memmove
              std::copy_if(x, x+output_length, result.data()+tile*output_length, always_true{} );
#endif
          });
#else
          std::copy(buffer, buffer + output_length,
                    result.data() + tile * output_length);

#endif
        }
      });
  if (output_result != nullptr)
    ValueInit::init(ReducerConditional::select(f, reducer), output_result);
  fut.wait();
  copy(result, result_cpu.data());
  if (output_result != nullptr) {
    for (std::size_t i = 0; i < td.num_tiles; i++)
      ValueJoin::join(ReducerConditional::select(f, reducer), output_result,
                      result_cpu.data() + i * output_length);

    ValueFinal::final(ReducerConditional::select(f, reducer), output_result);
  }
}

}  // namespace Impl
}  // namespace Kokkos

#endif /* #if !defined( KOKKOS_ROCM_AMP_REDUCE_INL ) */
