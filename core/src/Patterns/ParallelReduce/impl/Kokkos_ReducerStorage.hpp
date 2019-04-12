/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
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

#ifndef KOKKOS_PATTERNS_PARALLEL_REDUCE_KOKKOS_REDUCERSTORAGE_HPP
#define KOKKOS_PATTERNS_PARALLEL_REDUCE_KOKKOS_REDUCERSTORAGE_HPP

#include <Kokkos_Macros.hpp>

#include <utility>

namespace Kokkos {
namespace Impl {

/** A helper class for combining "early" and "late" construction of Reducers.
 *
 * Many internally constructed reducers need to store pointers to the
 * ReductionFunctor and/or the ExecutionPolicy, which are stored in the
 * `Impl::ParallelReduce` specialization, so they can't be constructed until
 * the `Impl::ParallelReduce` specialization is constructed.  User-defined
 * Reducers, however, are constructed in user code, and thus only need to be
 * moved into the storage in the `Impl::ParallelReduce` specialization.  To
 * avoid entagling these concerns with the rest of the specialization,
 * implementations of `Impl::ParallelReduce` specializations should use this
 * helper type to construct and store the Reducer.
 *
 * @tparam ReducerType A type that meets the requirements of Reducer
 */
template <class ReducerType>
class ReducerStorage
{
public:
  using reducer = ReducerType;

private:

  reducer m_reducer;

public:

  //----------------------------------------------------------------------------

  /**
   *  Construct the reducer from references to a Functor, an ExecutionPolicy,
   *  and a pointer to the result.  These references and pointers must be valid
   *  for the lifetime of the reducer!
   *
   */
  template <
    class FunctorType,
    class PolicyType,
    class ViewType
  >
  KOKKOS_INLINE_FUNCTION
  ReducerStorage(
    FunctorType const& arg_functor,
    PolicyType const& arg_policy,
    ViewType arg_view
  ) noexcept(noexcept(ReducerType(arg_functor, arg_policy, arg_view)))
    : m_reducer(arg_functor, arg_policy, arg_view)
  { }

  explicit
  KOKKOS_INLINE_FUNCTION
  ReducerStorage(
    ReducerType arg_reducer
  ) noexcept
    : m_reducer(std::move(arg_reducer))
  { }

  //----------------------------------------------------------------------------

  reducer const& get_reducer() const { return m_reducer; }

};

} // end namespace Impl
} // end namespace Kokkos

#endif //KOKKOS_PATTERNS_PARALLEL_REDUCE_KOKKOS_REDUCERSTORAGE_HPP
