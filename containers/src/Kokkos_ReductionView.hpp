/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
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
// Questions? Contact  H. Carter Edwards (hcedwar@sandia.gov)
//
// ************************************************************************
//@HEADER
*/


/// \file Kokkos_ReductionView.hpp
/// \brief Declaration and definition of Kokkos::ReductionView.
///
/// This header file declares and defines Kokkos::ReductionView and its
/// related nonmember functions.

#ifndef KOKKOS_REDUCTIONVIEW_HPP
#define KOKKOS_REDUCTIONVIEW_HPP

#include <Kokkos_Core.hpp>
#include <utility>

namespace Kokkos {
namespace Experimental {

enum : int {
  ReductionSum,
};

enum : int {
  ReductionDuplicated,
  ReductionNonDuplicated
};

enum : int {
  ReductionAtomic,
  ReductionNonAtomic
};

}} // Kokkos::Experimental

namespace Kokkos {
namespace Impl {
namespace Experimental {

template <typename ExecSpace>
struct DefaultReductionImpl;

#ifdef KOKKOS_ENABLE_SERIAL
template <>
struct DefaultReductionImpl<Kokkos::Serial> {
  enum : int { duplication = Kokkos::Experimental::ReductionNonDuplicated };
  enum : int { contribution = Kokkos::Experimental::ReductionNonAtomic };
};
#endif

#ifdef KOKKOS_ENABLE_OPENMP
template <>
struct DefaultReductionImpl<Kokkos::OpenMP> {
  enum : int { duplication = Kokkos::Experimental::ReductionDuplicated };
  enum : int { contribution = Kokkos::Experimental::ReductionNonAtomic };
};
#endif

#ifdef KOKKOS_ENABLE_THREADS
template <>
struct DefaultReductionImpl<Kokkos::Threads> {
  enum : int { duplication = Kokkos::Experimental::ReductionDuplicated };
  enum : int { contribution = Kokkos::Experimental::ReductionNonAtomic };
};
#endif

#ifdef KOKKOS_ENABLE_CUDA
template <>
struct DefaultReductionImpl<Kokkos::Cuda> {
  enum : int { duplication = Kokkos::Experimental::ReductionNonDuplicated };
  enum : int { contribution = Kokkos::Experimental::ReductionAtomic };
};
#endif

template <typename ValueType, int Op, int contribution>
struct ReductionValue;

template <typename ValueType>
struct ReductionValue<ValueType, Kokkos::Experimental::ReductionSum, Kokkos::Experimental::ReductionNonAtomic> {
  public:
    KOKKOS_FORCEINLINE_FUNCTION ReductionValue(ValueType& value_in) : value( value_in ) {}
    KOKKOS_FORCEINLINE_FUNCTION void operator+=(ValueType const& rhs) {
      value += rhs;
    }
  private:
    ValueType& value;
};

template <typename ValueType>
struct ReductionValue<ValueType, Kokkos::Experimental::ReductionSum, Kokkos::Experimental::ReductionAtomic> {
  public:
    KOKKOS_FORCEINLINE_FUNCTION ReductionValue(ValueType& value_in) : value( value_in ) {}
    KOKKOS_FORCEINLINE_FUNCTION void operator+=(ValueType const& rhs) {
      Kokkos::atomic_add(&value, rhs);
    }
  private:
    ValueType& value;
};

template <typename T, typename Layout>
struct DuplicatedDataType;

template <typename T>
struct DuplicatedDataType<T, Kokkos::LayoutRight> {
  typedef T* value_type; // For LayoutRight, add a star all the way on the left
};

template <typename T, size_t N>
struct DuplicatedDataType<T[N], Kokkos::LayoutRight> {
  typedef typename DuplicatedDataType<T, Kokkos::LayoutRight>::value_type value_type[N];
};

template <typename T>
struct DuplicatedDataType<T[], Kokkos::LayoutRight> {
  typedef typename DuplicatedDataType<T, Kokkos::LayoutRight>::value_type value_type[];
};

template <typename T>
struct DuplicatedDataType<T*, Kokkos::LayoutRight> {
  typedef typename DuplicatedDataType<T, Kokkos::LayoutRight>::value_type* value_type;
};

template <typename T>
struct DuplicatedDataType<T, Kokkos::LayoutLeft> {
  typedef T* value_type;
};

template <typename T, size_t N>
struct DuplicatedDataType<T[N], Kokkos::LayoutLeft> {
  typedef typename DuplicatedDataType<T, Kokkos::LayoutLeft>::value_type* value_type;
};

template <typename T>
struct DuplicatedDataType<T[], Kokkos::LayoutLeft> {
  typedef typename DuplicatedDataType<T, Kokkos::LayoutLeft>::value_type* value_type;
};

template <typename T>
struct DuplicatedDataType<T*, Kokkos::LayoutLeft> {
  typedef typename DuplicatedDataType<T, Kokkos::LayoutLeft>::value_type* value_type;
};

}}} // Kokkos::Impl::Experimental

namespace Kokkos {
namespace Experimental {

template <typename DataType
         ,int Op = ReductionSum
         ,typename ExecSpace = Kokkos::DefaultExecutionSpace
         ,typename Layout = Kokkos::DefaultExecutionSpace::array_layout
         ,int contribution = Kokkos::Impl::Experimental::DefaultReductionImpl<ExecSpace>::contribution
         ,int duplication = Kokkos::Impl::Experimental::DefaultReductionImpl<ExecSpace>::duplication
         >
class ReductionView;

template <typename DataType
         ,int Op
         ,typename ExecSpace
         ,typename Layout
         ,int contribution
         ,int duplication
         >
class ReductionAccess;

// non-duplicated implementation
template <typename DataType
         ,int Op
         ,typename ExecSpace
         ,typename Layout
         ,int contribution
         >
class ReductionView<DataType
                   ,Op
                   ,ExecSpace
                   ,Layout
                   ,contribution
                   ,ReductionNonDuplicated>
{
public:
  typedef Kokkos::View<DataType, Layout, ExecSpace> original_view_type;
  typedef typename original_view_type::value_type original_value_type;
  typedef Kokkos::Impl::Experimental::ReductionValue<
      original_value_type, Op, contribution> value_type;
  typedef ReductionAccess<DataType, Op, ExecSpace, Layout, contribution, ReductionNonDuplicated> access_type;

  ReductionView()
  {
  }

  template <typename RT, typename ... RP >
  ReductionView(View<RT, RP...> const& original_view)
  : internal_view(original_view)
  {
  }

  access_type access() const {
    return access_type(*this);
  }

protected:
  template <typename ... Args>
  KOKKOS_FORCEINLINE_FUNCTION
  value_type at(Args ... args) const {
    return internal_view(args...);
  }
private:
  typedef original_view_type internal_view_type;
  internal_view_type internal_view;
};

template <typename DataType
         ,int Op
         ,typename ExecSpace
         ,typename Layout
         ,int contribution
         >
class ReductionAccess<DataType
                   ,Op
                   ,ExecSpace
                   ,Layout
                   ,contribution
                   ,ReductionNonDuplicated>
      : public ReductionView<DataType, Op, ExecSpace, Layout, contribution, ReductionNonDuplicated>
{
public:
  typedef ReductionView<DataType, Op, ExecSpace, Layout, contribution> Base;
  using typename Base::value_type;

  KOKKOS_INLINE_FUNCTION
  ReductionAccess(Base const& base)
    : Base(base)
  {
  }

  template <typename ... Args>
  KOKKOS_FORCEINLINE_FUNCTION
  value_type operator()(Args ... args) const {
    return Base::at(args...);
  }
};

// duplicated implementation
// LayoutLeft and LayoutRight are different enough that we'll just specialize each

template <typename DataType
         ,int Op
         ,typename ExecSpace
         ,int contribution
         >
class ReductionView<DataType
                   ,Op
                   ,ExecSpace
                   ,Kokkos::LayoutRight
                   ,contribution
                   ,ReductionDuplicated>
{
public:
  typedef Kokkos::View<DataType, Kokkos::LayoutRight, ExecSpace> original_view_type;
  typedef typename original_view_type::value_type original_value_type;
  typedef Kokkos::Impl::Experimental::ReductionValue<
      original_value_type, Op, contribution> value_type;
  typedef ReductionAccess<DataType, Op, ExecSpace, Kokkos::LayoutRight, contribution, ReductionDuplicated> access_type;

  ReductionView()
  {
  }

  template <typename RT, typename ... RP >
  ReductionView(View<RT, RP...> const& original_view)
  : unique_token()
  , internal_view(original_view.labe(),
                  unique_token.size(),
                  original_view.dimensions_0(),
                  original_view.dimensions_1(),
                  original_view.dimensions_2(),
                  original_view.dimensions_3(),
                  original_view.dimensions_4(),
                  original_view.dimensions_5(),
                  original_view.dimensions_6())
  {
  }

  KOKKOS_INLINE_FUNCTION
  access_type access() const {
    return access_type(*this);
  }

protected:
  template <typename ... Args>
  KOKKOS_FORCEINLINE_FUNCTION
  value_type at(int rank, Args ... args) const {
    return internal_view(rank, args...);
  }

private:
  typedef typename Kokkos::Impl::Experimental::DuplicatedDataType<DataType, Kokkos::LayoutRight> data_type_info;
  typedef typename data_type_info::value_type internal_data_type;
  typedef Kokkos::View<internal_data_type, Kokkos::LayoutRight, ExecSpace> internal_view_type;
  typedef Kokkos::Experimental::UniqueToken<
      ExecSpace, Kokkos::Experimental::UniqueTokenScope::Instance> unique_token_type;

  unique_token_type unique_token;
  internal_view_type internal_view;
};

template <typename DataType
         ,int Op
         ,typename ExecSpace
         ,int contribution
         >
class ReductionView<DataType
                   ,Op
                   ,ExecSpace
                   ,Kokkos::LayoutLeft
                   ,contribution
                   ,ReductionDuplicated>
{
public:
  typedef Kokkos::View<DataType, Kokkos::LayoutLeft, ExecSpace> original_view_type;
  typedef typename original_view_type::value_type original_value_type;
  typedef Kokkos::Impl::Experimental::ReductionValue<
      original_value_type, Op, contribution> value_type;
  typedef ReductionAccess<DataType, Op, ExecSpace, Kokkos::LayoutLeft, contribution, ReductionDuplicated> access_type;

  ReductionView()
  {
  }

  template <typename RT, typename ... RP >
  ReductionView(View<RT, RP...> const& original_view)
  : unique_token()
  {
    size_t arg_N[8] = {
      original_view.dimension_0(),
      original_view.dimension_1(),
      original_view.dimension_2(),
      original_view.dimension_3(),
      original_view.dimension_4(),
      original_view.dimension_5(),
      original_view.dimension_6(),
      0
    };
    for (int i = 0; i < 8; ++i) {
      if (arg_N[i] == 0) {
        arg_N[i] = unique_token.size();
        break;
      }
    }
    internal_view = internal_view_type(
        original_view.name(),
        arg_N[0], arg_N[1], arg_N[2], arg_N[3],
        arg_N[4], arg_N[5], arg_N[6], arg_N[7]);
  }

  KOKKOS_INLINE_FUNCTION
  access_type access() const {
    return access_type(*this);
  }

protected:
  template <typename ... Args>
  KOKKOS_FORCEINLINE_FUNCTION
  value_type at(int rank, Args ... args) const {
    return internal_view(args..., rank);
  }

private:
  typedef typename Kokkos::Impl::Experimental::DuplicatedDataType<DataType, Kokkos::LayoutLeft> data_type_info;
  typedef typename data_type_info::value_type internal_data_type;
  typedef Kokkos::View<internal_data_type, Kokkos::LayoutLeft, ExecSpace> internal_view_type;
  typedef Kokkos::Experimental::UniqueToken<
      ExecSpace, Kokkos::Experimental::UniqueTokenScope::Instance> unique_token_type;

  unique_token_type unique_token;
  internal_view_type internal_view;
};


/* This object has to be separate in order to store the thread rank, which cannot
   be obtained until one is inside a parallel construct, and may be relatively
   expensive to obtain at every contribution
   (calls a non-inlined function, looks up a thread-local variable).
   Due to the expense, it is sensible to query it at most once per parallel iterate
   (ideally once per thread, but parallel_for doesn't expose that)
   and then store it in a stack variable.
   ReductionAccess serves as a non-const object on the stack which can store the thread rank */

template <typename DataType
         ,int Op
         ,typename ExecSpace
         ,typename Layout
         ,int contribution
         >
class ReductionAccess<DataType
                   ,Op
                   ,ExecSpace
                   ,Layout
                   ,contribution
                   ,ReductionDuplicated>
      : public ReductionView<DataType, Op, ExecSpace, Layout, contribution, ReductionDuplicated>
{
public:
  typedef ReductionView<DataType, Op, ExecSpace, Layout, contribution, ReductionDuplicated> Base;
  using typename Base::value_type;

  KOKKOS_INLINE_FUNCTION
  ReductionAccess(Base const& base)
    : Base(base)
    , rank(Base::unique_token.acquire()) {
  }

  KOKKOS_INLINE_FUNCTION
  ~ReductionAccess() {
    Base::unique_token.release(rank);
  }

  template <typename ... Args>
  KOKKOS_FORCEINLINE_FUNCTION
  value_type operator()(Args ... args) const {
    return Base::at(rank, args...);
  }

private:

  // simplify RAII by disallowing copies
  ReductionAccess(ReductionAccess const& other) = delete;
  ReductionAccess(ReductionAccess&& other) = delete;
  ReductionAccess& operator=(ReductionAccess const& other) = delete;
  ReductionAccess& operator=(ReductionAccess&& other) = delete;

  using typename Base::unique_token_type;
  typedef typename unique_token_type::size_type rank_type;
  rank_type rank;
};

template <int Op = Kokkos::Experimental::ReductionSum, typename RT, typename ... RP>
ReductionView<
  RT,
  Op,
  typename ViewTraits<RT, RP...>::execution_space,
  typename ViewTraits<RT, RP...>::array_layout>
create_reduction_view(View<RT, RP...> const& original_view) {
  return original_view;
}

}} // namespace Kokkos::Experimental

#endif
