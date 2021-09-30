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

#ifndef KOKKOS_KOKKOS_VIEWVALUEFUNCTOR_HPP
#define KOKKOS_KOKKOS_VIEWVALUEFUNCTOR_HPP

#include <impl/Kokkos_SharedAlloc.hpp>
#include <Kokkos_Parallel_fwd.hpp>

namespace Kokkos {
namespace Impl {

struct ViewValueConstructTag {};
struct ViewValueDestroyTag {};

template <class Space, class ValueType>
struct ViewValueFunctorStorage {
  using execution_space = typename Space::execution_space;

  execution_space space;
  // TODO @mdspan shouldn't this be the pointer type from the View so that it
  //              includes things like alignment and restrict?
  ValueType* ptr;
  size_t n;
  std::string name;

  ViewValueFunctorStorage() = default;

  ViewValueFunctorStorage(const ViewValueFunctorStorage&) = default;

  ViewValueFunctorStorage(ViewValueFunctorStorage&&) = default;

  ViewValueFunctorStorage& operator=(const ViewValueFunctorStorage&) = default;

  ViewValueFunctorStorage& operator=(ViewValueFunctorStorage&&) = default;

  ~ViewValueFunctorStorage() = default;

  ViewValueFunctorStorage(execution_space arg_space, ValueType* const arg_ptr,
                          size_t const arg_n, std::string arg_name)
      : space(std::move(arg_space)),
        ptr(arg_ptr),
        n(arg_n),
        name(std::move(arg_name)) {}
};

//==============================================================================
// <editor-fold desc="ViewValueDestroy: specialize destruction behavior"> {{{1

// General case: we need to call the destructor
template <class Space, class ValueType, class Enable = void>
struct ViewValueDestroy : ViewValueFunctorStorage<Space, ValueType> {
  using base_t = ViewValueFunctorStorage<Space, ValueType>;
  using base_t::base_t;

  KOKKOS_INLINE_FUNCTION
  void operator()(ViewValueDestroyTag const&, const size_t i) const {
    (this->ptr + i)->~ValueType();
    // Old comment left here for posterity:
    // KOKKOS_IMPL_CUDA_CLANG_WORKAROUND this line causes ptax error
    // __cxa_begin_catch in nested_view unit-test
  }

  void destroy_shared_allocation() {
    Kokkos::parallel_for(
        "Kokkos::View::destruction [" + this->name + "]",
        RangePolicy<typename base_t::execution_space, ViewValueDestroyTag,
                    IndexType<int64_t>>(0, this->n),
        *this);
    // TODO add a view trait that makes this fence optional?
    this->space.fence();
  }
};

// Trivially destructible: don't need to launch a kernel to destroy the value
template <class Space, class ValueType>
struct ViewValueDestroy<
    Space, ValueType,
    std::enable_if_t<std::is_trivially_destructible<ValueType>::value>>
    : ViewValueFunctorStorage<Space, ValueType> {
  using base_t = ViewValueFunctorStorage<Space, ValueType>;
  using base_t::base_t;

  void destroy_shared_allocation() {
    // Note: this is new code after the View refactor, but after discussion we
    // decided that deciding not to fence just because something happens to be
    // trivially destructible could introduce obscure and difficult to detect
    // and find bugs. A user could make a change in one place that all of a
    // sudden causes a type to be trivially destructible and a fence that they
    // were (probably accidentally) relying on in a completely different part
    // of the code goes away, leading to race conditions that are hard to debug.
    // This is extremely counterintuitive behavior, so it's probably worth the
    // cost of always fencing here.
    this->space.fence();
  }
};

// </editor-fold> end ViewValueDestroy: specialize destruction behavior }}}1
//==============================================================================

/*
 *  The construction, assignment to default, and destruction
 *  are merged into a single functor.
 *  Primarily to work around an unresolved CUDA back-end bug
 *  that would lose the destruction cuda device function when
 *  called from the shared memory tracking destruction.
 *  Secondarily to have two fewer partial specializations.
 */
template <class Space, class ValueType>
struct ViewValueFunctor : ViewValueDestroy<Space, ValueType> {
  using base_t = ViewValueDestroy<Space, ValueType>;
  using base_t::base_t;

  KOKKOS_INLINE_FUNCTION
  void operator()(ViewValueConstructTag const&, size_t i) const {
    new (this->ptr + i) ValueType();
  }

  void construct_shared_allocation() {
#ifdef KOKKOS_ENABLE_CUDA
    if (std::is_same<ExecSpace, Kokkos::Cuda>::value) {
      Kokkos::Impl::cuda_prefetch_pointer(space, ptr, sizeof(ValueType) * n,
                                          true);
    }
#endif
    Kokkos::parallel_for(
        "Kokkos::View::initialization [" + this->name + "]",
        RangePolicy<typename base_t::execution_space, ViewValueConstructTag,
                    IndexType<int64_t>>{0, static_cast<int64_t>(this->n)},
        *this);
    // TODO add a view trait that makes this fence optional?
    this->space.fence();
  }
};

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_KOKKOS_VIEWVALUEFUNCTOR_HPP
