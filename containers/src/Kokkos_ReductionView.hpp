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
  ReductionNonDuplicated = 0,
  ReductionDuplicated    = 1
};

enum : int {
  ReductionNonAtomic = 0,
  ReductionAtomic    = 1
};

}} // Kokkos::Experimental

namespace Kokkos {
namespace Impl {
namespace Experimental {

template <typename ExecSpace>
struct DefaultDuplication;

template <typename ExecSpace, int duplication>
struct DefaultContribution;

#ifdef KOKKOS_ENABLE_SERIAL
template <>
struct DefaultDuplication<Kokkos::Serial> {
  enum : int { value = Kokkos::Experimental::ReductionNonDuplicated };
};
template <>
struct DefaultContribution<Kokkos::Serial, Kokkos::Experimental::ReductionNonDuplicated> {
  enum : int { value = Kokkos::Experimental::ReductionNonAtomic };
};
template <>
struct DefaultContribution<Kokkos::Serial, Kokkos::Experimental::ReductionDuplicated> {
  enum : int { value = Kokkos::Experimental::ReductionNonAtomic };
};
#endif

#ifdef KOKKOS_ENABLE_OPENMP
template <>
struct DefaultDuplication<Kokkos::OpenMP> {
  enum : int { value = Kokkos::Experimental::ReductionDuplicated };
};
template <>
struct DefaultContribution<Kokkos::OpenMP, Kokkos::Experimental::ReductionNonDuplicated> {
  enum : int { value = Kokkos::Experimental::ReductionAtomic };
};
template <>
struct DefaultContribution<Kokkos::OpenMP, Kokkos::Experimental::ReductionDuplicated> {
  enum : int { value = Kokkos::Experimental::ReductionNonAtomic };
};
#endif

#ifdef KOKKOS_ENABLE_THREADS
template <>
struct DefaultDuplication<Kokkos::Threads> {
  enum : int { value = Kokkos::Experimental::ReductionDuplicated };
};
template <>
struct DefaultContribution<Kokkos::Threads, Kokkos::Experimental::ReductionNonDuplicated> {
  enum : int { value = Kokkos::Experimental::ReductionAtomic };
};
template <>
struct DefaultContribution<Kokkos::Threads, Kokkos::Experimental::ReductionDuplicated> {
  enum : int { value = Kokkos::Experimental::ReductionNonAtomic };
};
#endif

#ifdef KOKKOS_ENABLE_CUDA
template <>
struct DefaultDuplication<Kokkos::Cuda> {
  enum : int { value = Kokkos::Experimental::ReductionNonDuplicated };
};
template <>
struct DefaultContribution<Kokkos::Cuda, Kokkos::Experimental::ReductionNonDuplicated> {
  enum : int { value = Kokkos::Experimental::ReductionAtomic };
};
template <>
struct DefaultContribution<Kokkos::Cuda, Kokkos::Experimental::ReductionDuplicated> {
  enum : int { value = Kokkos::Experimental::ReductionAtomic };
};
#endif

/* ReductionValue is the object returned by the access operator() of ReductionAccess,
   similar to that returned by an Atomic View, it wraps Kokkos::atomic_add with convenient
   operator+=, etc. */
template <typename ValueType, int Op, int contribution>
struct ReductionValue;

template <typename ValueType>
struct ReductionValue<ValueType, Kokkos::Experimental::ReductionSum, Kokkos::Experimental::ReductionNonAtomic> {
  public:
    KOKKOS_FORCEINLINE_FUNCTION ReductionValue(ValueType& value_in) : value( value_in ) {}
    KOKKOS_FORCEINLINE_FUNCTION ReductionValue(ReductionValue&& other) : value( other.value ) {}
    KOKKOS_FORCEINLINE_FUNCTION void operator+=(ValueType const& rhs) {
      value += rhs;
    }
    KOKKOS_FORCEINLINE_FUNCTION void operator-=(ValueType const& rhs) {
      value -= rhs;
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
    KOKKOS_FORCEINLINE_FUNCTION void operator-=(ValueType const& rhs) {
      Kokkos::atomic_add(&value, -rhs);
    }
  private:
    ValueType& value;
};

/* DuplicatedDataType, given a View DataType, will create a new DataType
   that has a new runtime dimension which becomes the largest-stride dimension.
   In the case of LayoutLeft, due to the limitation induced by the design of DataType
   itself, it must convert any existing compile-time dimensions into runtime dimensions. */
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

/* Slice is just responsible for stuffing the correct number of Kokkos::ALL
   arguments on the correct side of the index in a call to subview() to get a
   subview where the index specified is the largest-stride one. */
template <typename Layout, int rank, typename V, typename ... Args>
struct Slice {
  typedef Slice<Layout, rank - 1, V, Kokkos::Impl::ALL_t, Args...> next;
  typedef typename next::value_type value_type;

  static
  value_type get(V const& src, const size_t i, Args ... args) {
    return next::get(src, i, Kokkos::ALL, args...);
  }
};

template <typename V, typename ... Args>
struct Slice<Kokkos::LayoutRight, 1, V, Args...> {
  typedef typename Kokkos::Impl::ViewMapping
                          < void
                          , V
                          , const size_t
                          , Args ...
                          >::type value_type;
  static
  value_type get(V const& src, const size_t i, Args ... args) {
    return Kokkos::subview(src, i, args...);
  }
};

template <typename V, typename ... Args>
struct Slice<Kokkos::LayoutLeft, 1, V, Args...> {
  typedef typename Kokkos::Impl::ViewMapping
                          < void
                          , V
                          , Args ...
                          , const size_t
                          >::type value_type;
  static
  value_type get(V const& src, const size_t i, Args ... args) {
    return Kokkos::subview(src, args..., i);
  }
};

template <typename ExecSpace, typename ValueType, int Op>
struct ReduceDuplicates;

template <typename ExecSpace, typename ValueType, int Op>
struct ReduceDuplicatesBase {
  typedef ReduceDuplicates<ExecSpace, ValueType, Op> Derived;
  ValueType* ptr_in;
  ValueType* ptr_out;
  size_t stride;
  size_t n;
  ReduceDuplicatesBase(ValueType* ptr_in, ValueType* ptr_out, size_t stride_in, size_t n_in, std::string const& name)
    : ptr_in(ptr_in)
    , ptr_out(ptr_out)
    , stride(stride_in)
    , n(n_in)
  {
#if defined(KOKKOS_ENABLE_PROFILING)
    uint64_t kpID = 0;
    if(Kokkos::Profiling::profileLibraryLoaded()) {
      Kokkos::Profiling::beginParallelFor(std::string("reduce_") + name, 0, &kpID);
    }
#endif
    typedef RangePolicy<ExecSpace, size_t> policy_type;
    typedef Kokkos::Impl::ParallelFor<Derived, policy_type> closure_type;
    const closure_type closure(*(static_cast<Derived*>(this)), policy_type(0, stride));
    closure.execute();
#if defined(KOKKOS_ENABLE_PROFILING)
    if(Kokkos::Profiling::profileLibraryLoaded()) {
      Kokkos::Profiling::endParallelFor(kpID);
    }
#endif
  }
};

template <typename ExecSpace, typename ValueType>
struct ReduceDuplicates<ExecSpace, ValueType, Kokkos::Experimental::ReductionSum> :
  public ReduceDuplicatesBase<ExecSpace, ValueType, Kokkos::Experimental::ReductionSum>
{
  typedef ReduceDuplicatesBase<ExecSpace, ValueType, Kokkos::Experimental::ReductionSum> Base;
  ReduceDuplicates(ValueType* ptr_in, ValueType* ptr_out, size_t stride_in, size_t n_in, std::string const& name):
    Base(ptr_in, ptr_out, stride_in, n_in, name)
  {}
  KOKKOS_FORCEINLINE_FUNCTION void operator()(size_t i) const {
    for (size_t j = 0; j < Base::n; ++j) {
      Base::ptr_out[i] += Base::ptr_in[i + Base::stride * j];
    }
  }
};

template <typename ExecSpace, typename ValueType, int Op>
struct ResetDuplicates;

template <typename ExecSpace, typename ValueType, int Op>
struct ResetDuplicatesBase {
  typedef ResetDuplicates<ExecSpace, ValueType, Op> Derived;
  ValueType* ptr_in;
  ResetDuplicatesBase(ValueType* ptr_in, size_t size_in, std::string const& name)
    : ptr_in(ptr_in)
  {
#if defined(KOKKOS_ENABLE_PROFILING)
    uint64_t kpID = 0;
    if(Kokkos::Profiling::profileLibraryLoaded()) {
      Kokkos::Profiling::beginParallelFor(std::string("reduce_") + name, 0, &kpID);
    }
#endif
    typedef RangePolicy<ExecSpace, size_t> policy_type;
    typedef Kokkos::Impl::ParallelFor<Derived, policy_type> closure_type;
    const closure_type closure(*(static_cast<Derived*>(this)), policy_type(0, size_in));
    closure.execute();
#if defined(KOKKOS_ENABLE_PROFILING)
    if(Kokkos::Profiling::profileLibraryLoaded()) {
      Kokkos::Profiling::endParallelFor(kpID);
    }
#endif
  }
};

template <typename ExecSpace, typename ValueType>
struct ResetDuplicates<ExecSpace, ValueType, Kokkos::Experimental::ReductionSum> :
  public ResetDuplicatesBase<ExecSpace, ValueType, Kokkos::Experimental::ReductionSum>
{
  typedef ResetDuplicatesBase<ExecSpace, ValueType, Kokkos::Experimental::ReductionSum> Base;
  ResetDuplicates(ValueType* ptr_in, size_t size_in, std::string const& name):
    Base(ptr_in, size_in, name)
  {}
  KOKKOS_FORCEINLINE_FUNCTION void operator()(size_t i) const {
    Base::ptr_in[i] = Kokkos::reduction_identity<ValueType>::sum();
  }
};

}}} // Kokkos::Impl::Experimental

namespace Kokkos {
namespace Experimental {

template <typename DataType
         ,int Op = ReductionSum
         ,typename ExecSpace = Kokkos::DefaultExecutionSpace
         ,typename Layout = Kokkos::DefaultExecutionSpace::array_layout
         ,int duplication = Kokkos::Impl::Experimental::DefaultDuplication<ExecSpace>::value
         ,int contribution = Kokkos::Impl::Experimental::DefaultContribution<ExecSpace, duplication>::value
         >
class ReductionView;

template <typename DataType
         ,int Op
         ,typename ExecSpace
         ,typename Layout
         ,int duplication
         ,int contribution
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
                   ,ReductionNonDuplicated
                   ,contribution>
{
public:
  typedef Kokkos::View<DataType, Layout, ExecSpace> original_view_type;
  typedef typename original_view_type::value_type original_value_type;
  typedef Kokkos::Impl::Experimental::ReductionValue<
      original_value_type, Op, contribution> value_type;
  friend class ReductionAccess<DataType, Op, ExecSpace, Layout, ReductionNonDuplicated, ReductionNonAtomic>;
  friend class ReductionAccess<DataType, Op, ExecSpace, Layout, ReductionNonDuplicated, ReductionAtomic>;

  ReductionView()
  {
  }

  template <typename RT, typename ... RP>
  ReductionView(View<RT, RP...> const& original_view)
  : internal_view(original_view)
  {
  }

  template <int access_contrib = contribution>
  KOKKOS_FORCEINLINE_FUNCTION
  ReductionAccess<DataType, Op, ExecSpace, Layout, ReductionNonDuplicated, access_contrib>
  access() const {
    return ReductionAccess<DataType, Op, ExecSpace, Layout, ReductionNonDuplicated, access_contrib>{*this};
  }

  template <typename ... RP>
  void contribute_into(View<DataType, RP...> const& dest) const
  {
    typedef View<DataType, RP...> dest_type;
    static_assert(std::is_same<
        typename dest_type::array_layout,
        Layout>::value,
        "ReductionView contribute destination has different layout");
    static_assert(Kokkos::Impl::VerifyExecutionCanAccessMemorySpace<
        typename ExecSpace::memory_space,
        typename dest_type::memory_space>::value,
        "ReductionView contribute destination memory space not accessible");
    if (dest.data() == internal_view.data()) return;
    Kokkos::Impl::Experimental::ReduceDuplicates<ExecSpace, original_value_type, Op>(
        internal_view.data(),
        dest.data(),
        0,
        1,
        internal_view.label());
  }

  void reset() {
    Kokkos::Impl::Experimental::ResetDuplicates<ExecSpace, original_value_type, Op>(
        internal_view.data(),
        internal_view.size(),
        internal_view.label());
  }
  void reset_duplicates() {}

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
                   ,ReductionNonDuplicated
                   ,contribution>
{
public:
  typedef ReductionView<DataType, Op, ExecSpace, Layout, ReductionNonDuplicated, contribution> view_type;
  typedef typename view_type::value_type value_type;

  KOKKOS_INLINE_FUNCTION
  ReductionAccess(view_type const& view_in)
    : view(view_in)
  {
  }

  template <typename ... Args>
  KOKKOS_FORCEINLINE_FUNCTION
  value_type operator()(Args ... args) const {
    return view.at(args...);
  }
private:
  view_type const& view;
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
                   ,ReductionDuplicated
                   ,contribution>
{
public:
  typedef Kokkos::View<DataType, Kokkos::LayoutRight, ExecSpace> original_view_type;
  typedef typename original_view_type::value_type original_value_type;
  typedef Kokkos::Impl::Experimental::ReductionValue<
      original_value_type, Op, contribution> value_type;
  friend class ReductionAccess<DataType, Op, ExecSpace, Kokkos::LayoutRight, ReductionDuplicated, ReductionNonAtomic>;
  friend class ReductionAccess<DataType, Op, ExecSpace, Kokkos::LayoutRight, ReductionDuplicated, ReductionAtomic>;
  typedef typename Kokkos::Impl::Experimental::DuplicatedDataType<DataType, Kokkos::LayoutRight> data_type_info;
  typedef typename data_type_info::value_type internal_data_type;
  typedef Kokkos::View<internal_data_type, Kokkos::LayoutRight, ExecSpace> internal_view_type;

  ReductionView()
  {
  }

  template <typename RT, typename ... RP >
  ReductionView(View<RT, RP...> const& original_view)
  : unique_token()
  , internal_view(Kokkos::ViewAllocateWithoutInitializing(
                    std::string("duplicated_") + original_view.label()),
                  unique_token.size(),
                  original_view.dimension_0(),
                  original_view.dimension_1(),
                  original_view.dimension_2(),
                  original_view.dimension_3(),
                  original_view.dimension_4(),
                  original_view.dimension_5(),
                  original_view.dimension_6())
  {
    reset();
  }

  template <int access_contrib = contribution>
  inline
  ReductionAccess<DataType, Op, ExecSpace, Kokkos::LayoutRight, ReductionDuplicated, access_contrib>
  access() const {
    return ReductionAccess<DataType, Op, ExecSpace, Kokkos::LayoutRight, ReductionDuplicated, access_contrib>{*this};
  }

  typename Kokkos::Impl::Experimental::Slice<
    Kokkos::LayoutRight, internal_view_type::rank, internal_view_type>::value_type
  subview() const
  {
    return Kokkos::Impl::Experimental::Slice<
      Kokkos::LayoutRight, internal_view_type::Rank, internal_view_type>::get(internal_view, 0);
  }

  template <typename ... RP>
  void deep_copy_from(View<DataType, RP...> const& src)
  {
    //
  //Kokkos::deep_copy(this->subview(), src);
  }

  template <typename ... RP>
  void contribute_into(View<DataType, RP...> const& dest) const
  {
    typedef View<DataType, RP...> dest_type;
    static_assert(std::is_same<
        typename dest_type::array_layout,
        Kokkos::LayoutRight>::value,
        "ReductionView deep_copy destination has different layout");
    static_assert(Kokkos::Impl::VerifyExecutionCanAccessMemorySpace<
        typename ExecSpace::memory_space,
        typename dest_type::memory_space>::value,
        "ReductionView deep_copy destination memory space not accessible");
    size_t strides[8];
    internal_view.stride(strides);
    if (dest.data() == internal_view.data()) {
      Kokkos::Impl::Experimental::ReduceDuplicates<ExecSpace, original_value_type, Op>(
          internal_view.data() + strides[0],
          dest.data(),
          strides[0],
          internal_view.dimension(0) - 1,
          internal_view.label());
    } else {
      Kokkos::Impl::Experimental::ReduceDuplicates<ExecSpace, original_value_type, Op>(
          internal_view.data(),
          dest.data(),
          strides[0],
          internal_view.dimension(0),
          internal_view.label());
    }
  }

  void reset() {
    Kokkos::Impl::Experimental::ResetDuplicates<ExecSpace, original_value_type, Op>(
        internal_view.data(),
        internal_view.size(),
        internal_view.label());
  }
  void reset_duplicates() {
    size_t strides[8];
    internal_view.stride(strides);
    Kokkos::Impl::Experimental::ResetDuplicates<ExecSpace, original_value_type, Op>(
        internal_view.data() + strides[0],
        internal_view.size() - strides[0],
        internal_view.label());
  }

protected:
  template <typename ... Args>
  KOKKOS_FORCEINLINE_FUNCTION
  value_type at(int rank, Args ... args) const {
    return internal_view(rank, args...);
  }

protected:
  typedef Kokkos::Experimental::UniqueToken<
      ExecSpace, Kokkos::Experimental::UniqueTokenScope::Global> unique_token_type;

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
                   ,ReductionDuplicated
                   ,contribution>
{
public:
  typedef Kokkos::View<DataType, Kokkos::LayoutLeft, ExecSpace> original_view_type;
  typedef typename original_view_type::value_type original_value_type;
  typedef Kokkos::Impl::Experimental::ReductionValue<
      original_value_type, Op, contribution> value_type;
  friend class ReductionAccess<DataType, Op, ExecSpace, Kokkos::LayoutLeft, ReductionDuplicated, ReductionNonAtomic>;
  friend class ReductionAccess<DataType, Op, ExecSpace, Kokkos::LayoutLeft, ReductionDuplicated, ReductionAtomic>;
  typedef typename Kokkos::Impl::Experimental::DuplicatedDataType<DataType, Kokkos::LayoutLeft> data_type_info;
  typedef typename data_type_info::value_type internal_data_type;
  typedef Kokkos::View<internal_data_type, Kokkos::LayoutLeft, ExecSpace> internal_view_type;

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
        Kokkos::ViewAllocateWithoutInitializing(
          std::string("duplicated_") + original_view.name()),
        arg_N[0], arg_N[1], arg_N[2], arg_N[3],
        arg_N[4], arg_N[5], arg_N[6], arg_N[7]);
    reset();
  }

  template <int access_contrib = contribution>
  inline
  ReductionAccess<DataType, Op, ExecSpace, Kokkos::LayoutLeft, ReductionDuplicated, access_contrib>
  access() const {
    return ReductionAccess<DataType, Op, ExecSpace, Kokkos::LayoutLeft, ReductionDuplicated, access_contrib>{*this};
  }

  typename Kokkos::Impl::Experimental::Slice<
    Kokkos::LayoutLeft, internal_view_type::rank, internal_view_type>::value_type
  subview() const
  {
    return Kokkos::Impl::Experimental::Slice<
      Kokkos::LayoutLeft, internal_view_type::rank, internal_view_type>::get(internal_view, 0);
  }

  template <typename ... RP>
  void contribute_into(View<DataType, RP...> const& dest) const
  {
    typedef View<DataType, RP...> dest_type;
    static_assert(std::is_same<
        typename dest_type::array_layout,
        Kokkos::LayoutLeft>::value,
        "ReductionView deep_copy destination has different layout");
    static_assert(Kokkos::Impl::VerifyExecutionCanAccessMemorySpace<
        typename ExecSpace::memory_space,
        typename dest_type::memory_space>::value,
        "ReductionView deep_copy destination memory space not accessible");
    size_t strides[8];
    internal_view.stride(strides);
    size_t stride = strides[internal_view_type::rank - 1];
    auto dimension = internal_view.dimension(
        internal_view_type::rank - 1);
    if (dest.data() == internal_view.data()) {
      Kokkos::Impl::Experimental::ReduceDuplicates<ExecSpace, original_value_type, Op>(
          internal_view.data() + stride,
          dest.data(),
          stride,
          dimension - 1,
          internal_view.label());
    } else {
      Kokkos::Impl::Experimental::ReduceDuplicates<ExecSpace, original_value_type, Op>(
          internal_view.data(),
          dest.data(),
          stride,
          dimension,
          internal_view.label());
    }
  }

  void reset() {
    Kokkos::Impl::Experimental::ResetDuplicates<ExecSpace, original_value_type, Op>(
        internal_view.data(),
        internal_view.size(),
        internal_view.label());
  }
  void reset_duplicates() {
    size_t strides[8];
    internal_view.stride(strides);
    size_t stride = strides[internal_view_type::rank - 1];
    Kokkos::Impl::Experimental::ResetDuplicates<ExecSpace, original_value_type, Op>(
        internal_view.data() + stride,
        internal_view.size() - stride,
        internal_view.label());
  }

protected:
  template <typename ... Args>
  inline value_type at(int rank, Args ... args) const {
    return internal_view(args..., rank);
  }

protected:
  typedef Kokkos::Experimental::UniqueToken<
      ExecSpace, Kokkos::Experimental::UniqueTokenScope::Global> unique_token_type;

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
                   ,ReductionDuplicated
                   ,contribution>
{
public:
  typedef ReductionView<DataType, Op, ExecSpace, Layout, ReductionDuplicated, contribution> view_type;
  typedef typename view_type::value_type value_type;

  inline ReductionAccess(view_type const& view_in)
    : view(view_in)
    , rank(view_in.unique_token.acquire()) {
  }

  inline ~ReductionAccess() {
    if (rank != ~rank_type(0)) view.unique_token.release(rank);
  }

  template <typename ... Args>
  KOKKOS_FORCEINLINE_FUNCTION
  value_type operator()(Args ... args) const {
    return view.at(rank, args...);
  }

private:

  view_type const& view;

  // simplify RAII by disallowing copies
  ReductionAccess(ReductionAccess const& other) = delete;
  ReductionAccess& operator=(ReductionAccess const& other) = delete;
  ReductionAccess& operator=(ReductionAccess&& other) = delete;

public:
  // do need to allow moves though, for the common
  // auto b = a.access();
  // that assignments turns into a move constructor call 
  inline ReductionAccess(ReductionAccess&& other)
    : view(other.view)
    , rank(other.rank)
  {
    other.rank = ~rank_type(0);
  }

private:

  typedef typename view_type::unique_token_type unique_token_type;
  typedef typename unique_token_type::size_type rank_type;
  rank_type rank;
};

template <int Op = Kokkos::Experimental::ReductionSum,
          int duplication = -1,
          int contribution = -1,
          typename RT, typename ... RP>
ReductionView
  < RT
  , Op
  , typename ViewTraits<RT, RP...>::execution_space
  , typename ViewTraits<RT, RP...>::array_layout
  /* just setting defaults if not specified... things got messy because the view type
     does not come before the duplication/contribution settings in the
     template parameter list */
  , duplication == -1 ? Kokkos::Impl::Experimental::DefaultDuplication<typename ViewTraits<RT, RP...>::execution_space>::value : duplication
  , contribution == -1 ?
      Kokkos::Impl::Experimental::DefaultContribution<
                        typename ViewTraits<RT, RP...>::execution_space,
                        (duplication == -1 ?
                           Kokkos::Impl::Experimental::DefaultDuplication<
                             typename ViewTraits<RT, RP...>::execution_space
                             >::value
                                           : duplication
                        )
                        >::value
                       : contribution
  >
create_reduction_view(View<RT, RP...> const& original_view) {
  return original_view; // implicit ReductionView constructor call
}

}} // namespace Kokkos::Experimental

namespace Kokkos {
namespace Experimental {

template <typename DT, int OP, typename ES, typename LY, int CT, int DP, typename ... VP>
void
contribute(View<DT, VP...>& dest, Kokkos::Experimental::ReductionView<DT, OP, ES, LY, CT, DP> const& src)
{
  src.contribute_into(dest);
}

}} // namespace Kokkos::Experimental

#endif
