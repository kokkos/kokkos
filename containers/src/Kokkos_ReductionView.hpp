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
    ReductionValue(ValueType& value_in) : value( value_in ) {}
    void operator+=(ValueType const& rhs) {
      value += rhs;
    }
  private:
    ValueType& value;
};

template <typename ValueType>
struct ReductionValue<ValueType, Kokkos::Experimental::ReductionSum, Kokkos::Experimental::ReductionAtomic> {
  public:
    ReductionValue(ValueType& value_in) : value( value_in ) {}
    void operator+=(ValueType const& rhs) {
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
  template <typename ... Args>
  KOKKOS_FORCEINLINE_FUNCTION
  value_type operator()(Args ... args) const {
    return internal_view(args...);
  }
private:
  typedef original_view_type internal_view_type;
  internal_view_type internal_view;
};

// duplicated implementation
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
                   ,ReductionDuplicated>
{
public:
  static_assert(std::is_same<Layout, Kokkos::LayoutRight>::value,
                "duplicated ReductionView only supports LayoutRight");
  typedef Kokkos::View<DataType, Layout, ExecSpace> original_view_type;
  typedef typename original_view_type::value_type original_value_type;
  typedef Kokkos::Impl::Experimental::ReductionValue<
      original_value_type, Op, contribution> value_type;

  ReductionView()
  {
  }

  template <typename ... Args>
  ReductionView(
      typename std::enable_if<std::is_same<Layout, LayoutRight>::value,
                              std::string const&>::value_type name,
      Args ... args)
    : internal_view(name, args..., unique_token.size())
  {
  }

  template <typename ... Args>
  ReductionView(
      typename std::enable_if<std::is_same<Layout, LayoutLeft>::value,
                              std::string const&>::value_type name,
      Args ... args)
    : internal_view(name, unique_token.size(), args...)
  {
  }

  ReductionView(ReductionView const& other)
    : internal_view(other)
    , rank(unique_token.acquire())
  {
  }

  ReductionView(ReductionView&& other)
    : internal_view(std::move(other.internal_view))
    , rank(unique_token.acquire())
  {
  }

  ~ReductionView() {
    unique_token.release(rank);
  }

  ReductionView& operator=(ReductionView&& other) {
    internal_view = std::move(other.internal_view);
    return *this;
  }

  ReductionView& operator=(ReductionView const& other) {
    internal_view = other.internal_view;
    return *this;
  }

  template <typename ... Args>
  KOKKOS_FORCEINLINE_FUNCTION
  typename std::enable_if<
      std::is_same<Layout, Kokkos::LayoutRight>::value,
      value_type>::value_type
  operator()(Args ... args) const {
    return internal_view(rank, args...);
  }

  template <typename ... Args>
  KOKKOS_FORCEINLINE_FUNCTION
  typename std::enable_if<
      std::is_same<Layout, Kokkos::LayoutLeft>::value,
      value_type>::value_type
  operator()(Args ... args) const {
    return internal_view(args..., rank);
  }

private:
  typedef typename Kokkos::Impl::Experimental::DuplicatedDataType<DataType, Layout> data_type_info;
  typedef typename data_type_info::value_type internal_data_type;
  typedef Kokkos::View<internal_data_type, Layout, ExecSpace> internal_view_type;
  typedef Kokkos::Experimental::UniqueToken<
      ExecSpace, Kokkos::Experimental::UniqueTokenScope::Instance> unique_token_type;
  typedef typename unique_token_type::size_type rank_type;

  unique_token_type unique_token;
  internal_view_type internal_view;
  rank_type rank;
};

}} // namespace Kokkos::Experimental

#endif
