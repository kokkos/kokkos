//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#include <Kokkos_Macros.hpp>
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif
#ifndef KOKKOS_PARALLEL_REDUCE_HPP
#define KOKKOS_PARALLEL_REDUCE_HPP

#include <Kokkos_ReductionIdentity.hpp>
#include <Kokkos_View.hpp>
#include <impl/Kokkos_FunctorAnalysis.hpp>
#include <impl/Kokkos_Tools_Generic.hpp>
#include <type_traits>

namespace Kokkos {

// \brief Class offering functionalities common to all reducers
//
// In order to be a valid reducer, a class must implement the functions and
// define the types offered in this class.
// To facilitate implementation, a new reducer class can simply inherit from
// BaseReducer.
namespace Impl {
template <class Scalar, class Space>
struct BaseReducer {
 public:
  // Following types need to be available for the reducer to be valid
  using value_type       = std::remove_cv_t<Scalar>;
  using result_view_type = Kokkos::View<value_type, Space>;

  static_assert(!std::is_pointer_v<value_type> && !std::is_array_v<value_type>);

 protected:
  // Contains the value of the reduction
  result_view_type value;
  // Whether the reducer returns its value through a Kokkos::View or a scalar
  bool references_scalar_v;

 public:
  // Construct from a scalar value
  KOKKOS_INLINE_FUNCTION
  BaseReducer(value_type& value_) : value(&value_), references_scalar_v(true) {}

  // Construct from a View
  KOKKOS_INLINE_FUNCTION
  BaseReducer(const result_view_type& value_)
      : value(value_), references_scalar_v(false) {}

  // Reducers also need to implement the two following functions:
  // KOKKOS_INLINE_FUNCTION
  // void join(value_type& dest, const value_type& src) const {
  //    // Do the reduction operation here
  // }

  // KOKKOS_INLINE_FUNCTION
  // void init(value_type& val) const {
  //   // Return the neutral value for the reduction operation, for instance
  //   // FLOAT_MIN if searching for the max, 0 for a sum or 1 for a product).
  // }
  //

  // Needed accessors
  KOKKOS_INLINE_FUNCTION
  value_type& reference() const { return *value.data(); }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const { return value; }

  KOKKOS_INLINE_FUNCTION
  bool references_scalar() const { return references_scalar_v; }
};
}  // namespace Impl

template <class Scalar, class Space>
struct Sum : Impl::BaseReducer<Scalar, Space> {
 private:
  using parent_type = Impl::BaseReducer<Scalar, Space>;

 public:
  using reducer    = Sum<Scalar, Space>;
  using value_type = typename parent_type::value_type;

  // Inherit constructors
  using parent_type::parent_type;

  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const { dest += src; }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val = reduction_identity<value_type>::sum();
  }
};

template <typename Scalar, typename... Properties>
KOKKOS_DEDUCTION_GUIDE Sum(View<Scalar, Properties...> const&)
    -> Sum<Scalar, typename View<Scalar, Properties...>::memory_space>;

template <class Scalar, class Space>
struct Prod : Impl::BaseReducer<Scalar, Space> {
 private:
  using parent_type = Impl::BaseReducer<Scalar, Space>;

 public:
  using reducer    = Prod<Scalar, Space>;
  using value_type = typename parent_type::value_type;

  // Inherit constructors
  using parent_type::parent_type;

  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const { dest *= src; }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val = reduction_identity<value_type>::prod();
  }
};

template <typename Scalar, typename... Properties>
KOKKOS_DEDUCTION_GUIDE Prod(View<Scalar, Properties...> const&)
    -> Prod<Scalar, typename View<Scalar, Properties...>::memory_space>;

template <class Scalar, class Space>
struct Min : Impl::BaseReducer<Scalar, Space> {
 private:
  using parent_type = Impl::BaseReducer<Scalar, Space>;

 public:
  using reducer    = Min<Scalar, Space>;
  using value_type = typename parent_type::value_type;

  // Inherit constructors
  using parent_type::parent_type;

  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    if (src < dest) dest = src;
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val = reduction_identity<value_type>::min();
  }
};

template <typename Scalar, typename... Properties>
KOKKOS_DEDUCTION_GUIDE Min(View<Scalar, Properties...> const&)
    -> Min<Scalar, typename View<Scalar, Properties...>::memory_space>;

template <class Scalar, class Space>
struct Max : Impl::BaseReducer<Scalar, Space> {
 private:
  using parent_type = Impl::BaseReducer<Scalar, Space>;

 public:
  using reducer    = Max<Scalar, Space>;
  using value_type = typename parent_type::value_type;

  // Inherit constructors
  using parent_type::parent_type;

  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    if (src > dest) dest = src;
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val = reduction_identity<value_type>::max();
  }
};

template <typename Scalar, typename... Properties>
KOKKOS_DEDUCTION_GUIDE Max(View<Scalar, Properties...> const&)
    -> Max<Scalar, typename View<Scalar, Properties...>::memory_space>;

template <class Scalar, class Space>
struct LAnd : Impl::BaseReducer<Scalar, Space> {
 private:
  using parent_type = Impl::BaseReducer<Scalar, Space>;

 public:
  using reducer    = LAnd<Scalar, Space>;
  using value_type = typename parent_type::value_type;

  // Inherit constructors
  using parent_type::parent_type;

  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    dest = dest && src;
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val = reduction_identity<value_type>::land();
  }
};

template <typename Scalar, typename... Properties>
KOKKOS_DEDUCTION_GUIDE LAnd(View<Scalar, Properties...> const&)
    -> LAnd<Scalar, typename View<Scalar, Properties...>::memory_space>;

template <class Scalar, class Space>
struct LOr : Impl::BaseReducer<Scalar, Space> {
 private:
  using parent_type = Impl::BaseReducer<Scalar, Space>;

 public:
  using reducer    = LOr<Scalar, Space>;
  using value_type = typename parent_type::value_type;

  // Inherit constructors
  using parent_type::parent_type;

  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    dest = dest || src;
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val = reduction_identity<value_type>::lor();
  }
};

template <typename Scalar, typename... Properties>
KOKKOS_DEDUCTION_GUIDE LOr(View<Scalar, Properties...> const&)
    -> LOr<Scalar, typename View<Scalar, Properties...>::memory_space>;

template <class Scalar, class Space>
struct BAnd : Impl::BaseReducer<Scalar, Space> {
 private:
  using parent_type = Impl::BaseReducer<Scalar, Space>;

 public:
  using reducer    = BAnd<Scalar, Space>;
  using value_type = typename parent_type::value_type;

  // Inherit constructors
  using parent_type::parent_type;

  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    dest = dest & src;
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val = reduction_identity<value_type>::band();
  }
};

template <typename Scalar, typename... Properties>
KOKKOS_DEDUCTION_GUIDE BAnd(View<Scalar, Properties...> const&)
    -> BAnd<Scalar, typename View<Scalar, Properties...>::memory_space>;

template <class Scalar, class Space>
struct BOr : Impl::BaseReducer<Scalar, Space> {
 private:
  using parent_type = Impl::BaseReducer<Scalar, Space>;

 public:
  using reducer    = BOr<Scalar, Space>;
  using value_type = typename parent_type::value_type;

  // Inherit constructors
  using parent_type::parent_type;

  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    dest = dest | src;
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val = reduction_identity<value_type>::bor();
  }
};

template <typename Scalar, typename... Properties>
KOKKOS_DEDUCTION_GUIDE BOr(View<Scalar, Properties...> const&)
    -> BOr<Scalar, typename View<Scalar, Properties...>::memory_space>;

template <class Scalar, class Index>
struct ValLocScalar {
  Scalar val;
  Index loc;
};

template <class Scalar, class Index, class Space>
struct MinLoc
    : Impl::BaseReducer<
          ValLocScalar<std::remove_cv_t<Scalar>, std::remove_cv_t<Index>>,
          Space> {
 private:
  using scalar_type = std::remove_cv_t<Scalar>;
  using index_type  = std::remove_cv_t<Index>;
  using parent_type =
      Impl::BaseReducer<ValLocScalar<scalar_type, index_type>, Space>;

  static_assert(!std::is_pointer_v<scalar_type> &&
                !std::is_array_v<scalar_type>);

 public:
  using reducer    = MinLoc<Scalar, Index, Space>;
  using value_type = typename parent_type::value_type;

  // Inherit constructors
  using parent_type::parent_type;

  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    if (src.val < dest.val) {
      dest = src;
    } else if (src.val == dest.val &&
               dest.loc == reduction_identity<index_type>::min()) {
      dest.loc = src.loc;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val.val = reduction_identity<scalar_type>::min();
    val.loc = reduction_identity<index_type>::min();
  }
};

template <typename Scalar, typename Index, typename... Properties>
KOKKOS_DEDUCTION_GUIDE
MinLoc(View<ValLocScalar<Scalar, Index>, Properties...> const&) -> MinLoc<
    Scalar, Index,
    typename View<ValLocScalar<Scalar, Index>, Properties...>::memory_space>;

template <class Scalar, class Index, class Space>
struct MaxLoc
    : Impl::BaseReducer<
          ValLocScalar<std::remove_cv_t<Scalar>, std::remove_cv_t<Index>>,
          Space> {
 private:
  using scalar_type = std::remove_cv_t<Scalar>;
  using index_type  = std::remove_cv_t<Index>;
  using parent_type =
      Impl::BaseReducer<ValLocScalar<scalar_type, index_type>, Space>;

  static_assert(!std::is_pointer_v<scalar_type> &&
                !std::is_array_v<scalar_type>);

 public:
  using value_type = typename parent_type::value_type;

  using reducer = MaxLoc<Scalar, Index, Space>;

  // Inherit constructors
  using parent_type::parent_type;

  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    if (src.val > dest.val) {
      dest = src;
    } else if (src.val == dest.val &&
               dest.loc == reduction_identity<index_type>::min()) {
      dest.loc = src.loc;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val.val = reduction_identity<scalar_type>::max();
    val.loc = reduction_identity<index_type>::min();
  }
};

template <typename Scalar, typename Index, typename... Properties>
KOKKOS_DEDUCTION_GUIDE
MaxLoc(View<ValLocScalar<Scalar, Index>, Properties...> const&) -> MaxLoc<
    Scalar, Index,
    typename View<ValLocScalar<Scalar, Index>, Properties...>::memory_space>;

template <class Scalar>
struct MinMaxScalar {
  Scalar min_val, max_val;
};

template <class Scalar, class Space>
struct MinMax
    : Impl::BaseReducer<MinMaxScalar<std::remove_cv_t<Scalar>>, Space> {
 private:
  using scalar_type = std::remove_cv_t<Scalar>;
  using parent_type =
      Impl::BaseReducer<MinMaxScalar<std::remove_cv_t<Scalar>>, Space>;

  static_assert(!std::is_pointer_v<scalar_type> &&
                !std::is_array_v<scalar_type>);

 public:
  using value_type = typename parent_type::value_type;

  using reducer = MinMax<Scalar, Space>;

  // Inherit constructors
  using parent_type::parent_type;

  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    if (src.min_val < dest.min_val) {
      dest.min_val = src.min_val;
    }
    if (src.max_val > dest.max_val) {
      dest.max_val = src.max_val;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val.max_val = reduction_identity<scalar_type>::max();
    val.min_val = reduction_identity<scalar_type>::min();
  }
};

template <typename Scalar, typename... Properties>
KOKKOS_DEDUCTION_GUIDE MinMax(View<MinMaxScalar<Scalar>, Properties...> const&)
    -> MinMax<Scalar,
              typename View<MinMaxScalar<Scalar>, Properties...>::memory_space>;

template <class Scalar, class Index>
struct MinMaxLocScalar {
  Scalar min_val, max_val;
  Index min_loc, max_loc;
};

template <class Scalar, class Index, class Space>
struct MinMaxLoc
    : Impl::BaseReducer<
          MinMaxLocScalar<std::remove_cv_t<Scalar>, std::remove_cv_t<Index>>,
          Space> {
 private:
  using scalar_type = std::remove_cv_t<Scalar>;
  using index_type  = std::remove_cv_t<Index>;
  using parent_type =
      Impl::BaseReducer<MinMaxLocScalar<scalar_type, index_type>, Space>;

  static_assert(!std::is_pointer_v<scalar_type> &&
                !std::is_array_v<scalar_type>);

 public:
  using reducer    = MinMaxLoc<Scalar, Index, Space>;
  using value_type = typename parent_type::value_type;

  // Inherit constructors
  using parent_type::parent_type;

  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    if (src.min_val < dest.min_val) {
      dest.min_val = src.min_val;
      dest.min_loc = src.min_loc;
    } else if (dest.min_val == src.min_val &&
               dest.min_loc == reduction_identity<index_type>::min()) {
      dest.min_loc = src.min_loc;
    }
    if (src.max_val > dest.max_val) {
      dest.max_val = src.max_val;
      dest.max_loc = src.max_loc;
    } else if (dest.max_val == src.max_val &&
               dest.max_loc == reduction_identity<index_type>::min()) {
      dest.max_loc = src.max_loc;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val.max_val = reduction_identity<scalar_type>::max();
    val.min_val = reduction_identity<scalar_type>::min();
    val.max_loc = reduction_identity<index_type>::min();
    val.min_loc = reduction_identity<index_type>::min();
  }
};

template <typename Scalar, typename Index, typename... Properties>
KOKKOS_DEDUCTION_GUIDE MinMaxLoc(
    View<MinMaxLocScalar<Scalar, Index>, Properties...> const&)
    -> MinMaxLoc<Scalar, Index,
                 typename View<MinMaxLocScalar<Scalar, Index>,
                               Properties...>::memory_space>;

// --------------------------------------------------
// reducers added to support std algorithms
// --------------------------------------------------

//
// MaxFirstLoc
//
template <class Scalar, class Index, class Space>
struct MaxFirstLoc
    : Impl::BaseReducer<
          ValLocScalar<std::remove_cv_t<Scalar>, std::remove_cv_t<Index>>,
          Space> {
 private:
  using scalar_type = std::remove_cv_t<Scalar>;
  using index_type  = std::remove_cv_t<Index>;
  using parent_type =
      Impl::BaseReducer<ValLocScalar<scalar_type, index_type>, Space>;

  static_assert(!std::is_pointer_v<scalar_type> &&
                !std::is_array_v<scalar_type>);
  static_assert(std::is_integral_v<index_type>);

 public:
  using reducer    = MaxFirstLoc<Scalar, Index, Space>;
  using value_type = typename parent_type::value_type;

  // Inherit constructors
  using parent_type::parent_type;

  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    if (dest.val < src.val) {
      dest = src;
    } else if (!(src.val < dest.val)) {
      dest.loc = (src.loc < dest.loc) ? src.loc : dest.loc;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val.val = reduction_identity<scalar_type>::max();
    val.loc = reduction_identity<index_type>::min();
  }
};

template <typename Scalar, typename Index, typename... Properties>
KOKKOS_DEDUCTION_GUIDE MaxFirstLoc(
    View<ValLocScalar<Scalar, Index>, Properties...> const&)
    -> MaxFirstLoc<Scalar, Index,
                   typename View<ValLocScalar<Scalar, Index>,
                                 Properties...>::memory_space>;

//
// MaxFirstLocCustomComparator
// recall that comp(a,b) returns true is a < b
//
template <class Scalar, class Index, class ComparatorType, class Space>
struct MaxFirstLocCustomComparator
    : Impl::BaseReducer<
          ValLocScalar<std::remove_cv_t<Scalar>, std::remove_cv_t<Index>>,
          Space> {
 private:
  using scalar_type = std::remove_cv_t<Scalar>;
  using index_type  = std::remove_cv_t<Index>;
  using parent_type =
      Impl::BaseReducer<ValLocScalar<scalar_type, index_type>, Space>;

  static_assert(!std::is_pointer_v<scalar_type> &&
                !std::is_array_v<scalar_type>);
  static_assert(std::is_integral_v<index_type>);

 public:
  using reducer =
      MaxFirstLocCustomComparator<Scalar, Index, ComparatorType, Space>;
  using value_type       = typename parent_type::value_type;
  using result_view_type = typename parent_type::result_view_type;

 private:
  ComparatorType m_comp;

 public:
  KOKKOS_INLINE_FUNCTION
  MaxFirstLocCustomComparator(value_type& value_, ComparatorType comp_)
      : parent_type(value_), m_comp(comp_) {}

  KOKKOS_INLINE_FUNCTION
  MaxFirstLocCustomComparator(const result_view_type& value_,
                              ComparatorType comp_)
      : parent_type(value_), m_comp(comp_) {}

  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    if (m_comp(dest.val, src.val)) {
      dest = src;
    } else if (!m_comp(src.val, dest.val)) {
      dest.loc = (src.loc < dest.loc) ? src.loc : dest.loc;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val.val = reduction_identity<scalar_type>::max();
    val.loc = reduction_identity<index_type>::min();
  }
};

template <typename Scalar, typename Index, typename ComparatorType,
          typename... Properties>
KOKKOS_DEDUCTION_GUIDE MaxFirstLocCustomComparator(
    View<ValLocScalar<Scalar, Index>, Properties...> const&, ComparatorType)
    -> MaxFirstLocCustomComparator<Scalar, Index, ComparatorType,
                                   typename View<ValLocScalar<Scalar, Index>,
                                                 Properties...>::memory_space>;

//
// MinFirstLoc
//
template <class Scalar, class Index, class Space>
struct MinFirstLoc
    : Impl::BaseReducer<
          ValLocScalar<std::remove_cv_t<Scalar>, std::remove_cv_t<Index>>,
          Space> {
 private:
  using scalar_type = std::remove_cv_t<Scalar>;
  using index_type  = std::remove_cv_t<Index>;
  using parent_type =
      Impl::BaseReducer<ValLocScalar<scalar_type, index_type>, Space>;

  static_assert(!std::is_pointer_v<scalar_type> &&
                !std::is_array_v<scalar_type>);
  static_assert(std::is_integral_v<index_type>);

 public:
  using reducer    = MinFirstLoc<Scalar, Index, Space>;
  using value_type = typename parent_type::value_type;

  // Inherit constructors
  using parent_type::parent_type;

  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    if (src.val < dest.val) {
      dest = src;
    } else if (!(dest.val < src.val)) {
      dest.loc = (src.loc < dest.loc) ? src.loc : dest.loc;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val.val = reduction_identity<scalar_type>::min();
    val.loc = reduction_identity<index_type>::min();
  }
};

template <typename Scalar, typename Index, typename... Properties>
KOKKOS_DEDUCTION_GUIDE MinFirstLoc(
    View<ValLocScalar<Scalar, Index>, Properties...> const&)
    -> MinFirstLoc<Scalar, Index,
                   typename View<ValLocScalar<Scalar, Index>,
                                 Properties...>::memory_space>;

//
// MinFirstLocCustomComparator
// recall that comp(a,b) returns true is a < b
//
template <class Scalar, class Index, class ComparatorType, class Space>
struct MinFirstLocCustomComparator
    : Impl::BaseReducer<
          ValLocScalar<std::remove_cv_t<Scalar>, std::remove_cv_t<Index>>,
          Space> {
 private:
  using scalar_type = std::remove_cv_t<Scalar>;
  using index_type  = std::remove_cv_t<Index>;
  using parent_type =
      Impl::BaseReducer<ValLocScalar<scalar_type, index_type>, Space>;

  static_assert(!std::is_pointer_v<scalar_type> &&
                !std::is_array_v<scalar_type>);
  static_assert(std::is_integral_v<index_type>);

 public:
  using reducer =
      MinFirstLocCustomComparator<Scalar, Index, ComparatorType, Space>;
  using value_type       = typename parent_type::value_type;
  using result_view_type = typename parent_type::result_view_type;

 private:
  ComparatorType m_comp;

 public:
  KOKKOS_INLINE_FUNCTION
  MinFirstLocCustomComparator(value_type& value_, ComparatorType comp_)
      : parent_type(value_), m_comp(comp_) {}

  KOKKOS_INLINE_FUNCTION
  MinFirstLocCustomComparator(const result_view_type& value_,
                              ComparatorType comp_)
      : parent_type(value_), m_comp(comp_) {}

  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    if (m_comp(src.val, dest.val)) {
      dest = src;
    } else if (!m_comp(dest.val, src.val)) {
      dest.loc = (src.loc < dest.loc) ? src.loc : dest.loc;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val.val = reduction_identity<scalar_type>::min();
    val.loc = reduction_identity<index_type>::min();
  }
};

template <typename Scalar, typename Index, typename ComparatorType,
          typename... Properties>
KOKKOS_DEDUCTION_GUIDE MinFirstLocCustomComparator(
    View<ValLocScalar<Scalar, Index>, Properties...> const&, ComparatorType)
    -> MinFirstLocCustomComparator<Scalar, Index, ComparatorType,
                                   typename View<ValLocScalar<Scalar, Index>,
                                                 Properties...>::memory_space>;

//
// MinMaxFirstLastLoc
//
template <class Scalar, class Index, class Space>
struct MinMaxFirstLastLoc
    : Impl::BaseReducer<
          MinMaxLocScalar<std::remove_cv_t<Scalar>, std::remove_cv_t<Index>>,
          Space> {
 private:
  using scalar_type = std::remove_cv_t<Scalar>;
  using index_type  = std::remove_cv_t<Index>;
  using parent_type =
      Impl::BaseReducer<MinMaxLocScalar<scalar_type, index_type>, Space>;

  static_assert(!std::is_pointer_v<scalar_type> &&
                !std::is_array_v<scalar_type>);

 public:
  using reducer    = MinMaxFirstLastLoc<Scalar, Index, Space>;
  using value_type = typename parent_type::value_type;

  // Inherit constructors
  using parent_type::parent_type;

  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    if (src.min_val < dest.min_val) {
      dest.min_val = src.min_val;
      dest.min_loc = src.min_loc;
    } else if (!(dest.min_val < src.min_val)) {
      dest.min_loc = (src.min_loc < dest.min_loc) ? src.min_loc : dest.min_loc;
    }

    if (dest.max_val < src.max_val) {
      dest.max_val = src.max_val;
      dest.max_loc = src.max_loc;
    } else if (!(src.max_val < dest.max_val)) {
      dest.max_loc = (src.max_loc > dest.max_loc) ? src.max_loc : dest.max_loc;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val.max_val = ::Kokkos::reduction_identity<scalar_type>::max();
    val.min_val = ::Kokkos::reduction_identity<scalar_type>::min();
    val.max_loc = ::Kokkos::reduction_identity<index_type>::max();
    val.min_loc = ::Kokkos::reduction_identity<index_type>::min();
  }
};

template <typename Scalar, typename Index, typename... Properties>
KOKKOS_DEDUCTION_GUIDE MinMaxFirstLastLoc(
    View<MinMaxLocScalar<Scalar, Index>, Properties...> const&)
    -> MinMaxFirstLastLoc<Scalar, Index,
                          typename View<MinMaxLocScalar<Scalar, Index>,
                                        Properties...>::memory_space>;

//
// MinMaxFirstLastLocCustomComparator
// recall that comp(a,b) returns true is a < b
//
template <class Scalar, class Index, class ComparatorType, class Space>
struct MinMaxFirstLastLocCustomComparator
    : Impl::BaseReducer<
          MinMaxLocScalar<std::remove_cv_t<Scalar>, std::remove_cv_t<Index>>,
          Space> {
 private:
  using scalar_type = std::remove_cv_t<Scalar>;
  using index_type  = std::remove_cv_t<Index>;
  using parent_type =
      Impl::BaseReducer<MinMaxLocScalar<scalar_type, index_type>, Space>;

  static_assert(!std::is_pointer_v<scalar_type> &&
                !std::is_array_v<scalar_type>);

 public:
  using reducer =
      MinMaxFirstLastLocCustomComparator<Scalar, Index, ComparatorType, Space>;
  using value_type       = typename parent_type::value_type;
  using result_view_type = typename parent_type::result_view_type;

 private:
  ComparatorType m_comp;

 public:
  KOKKOS_INLINE_FUNCTION
  MinMaxFirstLastLocCustomComparator(value_type& value_, ComparatorType comp_)
      : parent_type(value_), m_comp(comp_) {}

  KOKKOS_INLINE_FUNCTION
  MinMaxFirstLastLocCustomComparator(const result_view_type& value_,
                                     ComparatorType comp_)
      : parent_type(value_), m_comp(comp_) {}

  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    if (m_comp(src.min_val, dest.min_val)) {
      dest.min_val = src.min_val;
      dest.min_loc = src.min_loc;
    } else if (!m_comp(dest.min_val, src.min_val)) {
      dest.min_loc = (src.min_loc < dest.min_loc) ? src.min_loc : dest.min_loc;
    }

    if (m_comp(dest.max_val, src.max_val)) {
      dest.max_val = src.max_val;
      dest.max_loc = src.max_loc;
    } else if (!m_comp(src.max_val, dest.max_val)) {
      dest.max_loc = (src.max_loc > dest.max_loc) ? src.max_loc : dest.max_loc;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val.max_val = ::Kokkos::reduction_identity<scalar_type>::max();
    val.min_val = ::Kokkos::reduction_identity<scalar_type>::min();
    val.max_loc = ::Kokkos::reduction_identity<index_type>::max();
    val.min_loc = ::Kokkos::reduction_identity<index_type>::min();
  }
};

template <typename Scalar, typename Index, typename ComparatorType,
          typename... Properties>
KOKKOS_DEDUCTION_GUIDE MinMaxFirstLastLocCustomComparator(
    View<MinMaxLocScalar<Scalar, Index>, Properties...> const&, ComparatorType)
    -> MinMaxFirstLastLocCustomComparator<
        Scalar, Index, ComparatorType,
        typename View<MinMaxLocScalar<Scalar, Index>,
                      Properties...>::memory_space>;

//
// FirstLoc
//
template <class Index>
struct FirstLocScalar {
  Index min_loc_true;
};

template <class Index, class Space>
struct FirstLoc
    : Impl::BaseReducer<FirstLocScalar<std::remove_cv_t<Index>>, Space> {
 private:
  using index_type = std::remove_cv_t<Index>;
  static_assert(std::is_integral_v<index_type>);
  static_assert(!std::is_pointer_v<index_type> && !std::is_array_v<index_type>);

  using parent_type =
      Impl::BaseReducer<FirstLocScalar<std::remove_cv_t<Index>>, Space>;

 public:
  using reducer    = FirstLoc<Index, Space>;
  using value_type = typename parent_type::value_type;

  // Inherit constructors
  using parent_type::parent_type;

  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    dest.min_loc_true = (src.min_loc_true < dest.min_loc_true)
                            ? src.min_loc_true
                            : dest.min_loc_true;
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val.min_loc_true = ::Kokkos::reduction_identity<index_type>::min();
  }
};

template <typename Index, typename... Properties>
KOKKOS_DEDUCTION_GUIDE
FirstLoc(View<FirstLocScalar<Index>, Properties...> const&) -> FirstLoc<
    Index, typename View<FirstLocScalar<Index>, Properties...>::memory_space>;

//
// LastLoc
//
template <class Index>
struct LastLocScalar {
  Index max_loc_true;
};

template <class Index, class Space>
struct LastLoc
    : Impl::BaseReducer<LastLocScalar<std::remove_cv_t<Index>>, Space> {
 private:
  using index_type = std::remove_cv_t<Index>;
  static_assert(std::is_integral_v<index_type>);
  static_assert(!std::is_pointer_v<index_type> && !std::is_array_v<index_type>);

  using parent_type = Impl::BaseReducer<LastLocScalar<index_type>, Space>;

 public:
  using reducer    = LastLoc<Index, Space>;
  using value_type = typename parent_type::value_type;

  // Inherit constructors
  using parent_type::parent_type;

  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    dest.max_loc_true = (src.max_loc_true > dest.max_loc_true)
                            ? src.max_loc_true
                            : dest.max_loc_true;
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val.max_loc_true = ::Kokkos::reduction_identity<index_type>::max();
  }
};

template <typename Index, typename... Properties>
KOKKOS_DEDUCTION_GUIDE LastLoc(View<LastLocScalar<Index>, Properties...> const&)
    -> LastLoc<Index, typename View<LastLocScalar<Index>,
                                    Properties...>::memory_space>;

template <class Index>
struct StdIsPartScalar {
  Index max_loc_true, min_loc_false;
};

//
// StdIsPartitioned
//
template <class Index, class Space>
struct StdIsPartitioned
    : Impl::BaseReducer<StdIsPartScalar<std::remove_cv_t<Index>>, Space> {
 private:
  using index_type = std::remove_cv_t<Index>;
  static_assert(std::is_integral_v<index_type>);
  static_assert(!std::is_pointer_v<index_type> && !std::is_array_v<index_type>);

  using parent_type = Impl::BaseReducer<StdIsPartScalar<index_type>, Space>;

 public:
  using reducer    = StdIsPartitioned<Index, Space>;
  using value_type = typename parent_type::value_type;

  // Inherit constructors
  using parent_type::parent_type;

  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    dest.max_loc_true = (dest.max_loc_true < src.max_loc_true)
                            ? src.max_loc_true
                            : dest.max_loc_true;

    dest.min_loc_false = (dest.min_loc_false < src.min_loc_false)
                             ? dest.min_loc_false
                             : src.min_loc_false;
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val.max_loc_true  = ::Kokkos::reduction_identity<index_type>::max();
    val.min_loc_false = ::Kokkos::reduction_identity<index_type>::min();
  }
};

template <typename Index, typename... Properties>
KOKKOS_DEDUCTION_GUIDE StdIsPartitioned(
    View<StdIsPartScalar<Index>, Properties...> const&)
    -> StdIsPartitioned<Index, typename View<StdIsPartScalar<Index>,
                                             Properties...>::memory_space>;

template <class Index>
struct StdPartPointScalar {
  Index min_loc_false;
};

//
// StdPartitionPoint
//
template <class Index, class Space>
struct StdPartitionPoint
    : Impl::BaseReducer<StdPartPointScalar<std::remove_cv_t<Index>>, Space> {
 private:
  using index_type = std::remove_cv_t<Index>;
  static_assert(std::is_integral_v<index_type>);
  static_assert(!std::is_pointer_v<index_type> && !std::is_array_v<index_type>);

  using parent_type = Impl::BaseReducer<StdPartPointScalar<index_type>, Space>;

 public:
  using reducer    = StdPartitionPoint<Index, Space>;
  using value_type = typename parent_type::value_type;

  // Inherit constructors
  using parent_type::parent_type;

  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    dest.min_loc_false = (dest.min_loc_false < src.min_loc_false)
                             ? dest.min_loc_false
                             : src.min_loc_false;
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val.min_loc_false = ::Kokkos::reduction_identity<index_type>::min();
  }
};

template <typename Index, typename... Properties>
KOKKOS_DEDUCTION_GUIDE StdPartitionPoint(
    View<StdPartPointScalar<Index>, Properties...> const&)
    -> StdPartitionPoint<Index, typename View<StdPartPointScalar<Index>,
                                              Properties...>::memory_space>;
}  // namespace Kokkos

namespace Kokkos {
namespace Impl {

template <typename FunctorType, typename FunctorAnalysisReducerType,
          typename Enable>
class CombinedFunctorReducer {
 public:
  using functor_type = FunctorType;
  using reducer_type = FunctorAnalysisReducerType;
  CombinedFunctorReducer(const FunctorType& functor,
                         const FunctorAnalysisReducerType& reducer)
      : m_functor(functor), m_reducer(reducer) {}
  KOKKOS_FUNCTION const FunctorType& get_functor() const { return m_functor; }
  KOKKOS_FUNCTION const FunctorAnalysisReducerType& get_reducer() const {
    return m_reducer;
  }

 private:
  FunctorType m_functor;
  FunctorAnalysisReducerType m_reducer;
};
template <typename FunctorType, typename FunctorAnalysisReducerType>
class CombinedFunctorReducer<
    FunctorType, FunctorAnalysisReducerType,
    std::enable_if_t<std::is_same_v<
        FunctorType, typename FunctorAnalysisReducerType::functor_type>>> {
 public:
  using functor_type = FunctorType;
  using reducer_type = FunctorAnalysisReducerType;
  CombinedFunctorReducer(const FunctorType& functor,
                         const FunctorAnalysisReducerType&)
      : m_reducer(functor) {}
  KOKKOS_FUNCTION const FunctorType& get_functor() const {
    return m_reducer.get_functor();
  }
  KOKKOS_FUNCTION const FunctorAnalysisReducerType& get_reducer() const {
    return m_reducer;
  }

 private:
  FunctorAnalysisReducerType m_reducer;
};

template <class T, class ReturnType, class ValueTraits>
struct ParallelReduceReturnValue;

template <class ReturnType, class FunctorType>
struct ParallelReduceReturnValue<
    std::enable_if_t<Kokkos::is_view<ReturnType>::value>, ReturnType,
    FunctorType> {
  using return_type  = ReturnType;
  using reducer_type = InvalidType;

  using value_type_scalar = typename return_type::value_type;
  using value_type_array  = typename return_type::value_type* const;

  using value_type = std::conditional_t<return_type::rank == 0,
                                        value_type_scalar, value_type_array>;

  static return_type& return_value(ReturnType& return_val, const FunctorType&) {
    return return_val;  // NOLINT(bugprone-return-const-ref-from-parameter)
  }
};

template <class ReturnType, class FunctorType>
struct ParallelReduceReturnValue<
    std::enable_if_t<!Kokkos::is_view<ReturnType>::value &&
                     (!std::is_array_v<ReturnType> &&
                      !std::is_pointer_v<
                          ReturnType>)&&!Kokkos::is_reducer<ReturnType>::value>,
    ReturnType, FunctorType> {
  using return_type =
      Kokkos::View<ReturnType, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;

  using reducer_type = InvalidType;

  using value_type = typename return_type::value_type;

  static return_type return_value(ReturnType& return_val, const FunctorType&) {
    return return_type(&return_val);
  }
};

template <class ReturnType, class FunctorType>
struct ParallelReduceReturnValue<
    std::enable_if_t<(std::is_array_v<ReturnType> ||
                      std::is_pointer_v<ReturnType>)>,
    ReturnType, FunctorType> {
  using return_type = Kokkos::View<std::remove_const_t<ReturnType>,
                                   Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;

  using reducer_type = InvalidType;

  using value_type = typename return_type::value_type[];

  static return_type return_value(ReturnType& return_val,
                                  const FunctorType& functor) {
    if (std::is_array_v<ReturnType>)
      return return_type(return_val);
    else
      return return_type(return_val, functor.value_count);
  }
};

template <class ReturnType, class FunctorType>
struct ParallelReduceReturnValue<
    std::enable_if_t<Kokkos::is_reducer<ReturnType>::value>, ReturnType,
    FunctorType> {
  using return_type  = typename ReturnType::result_view_type;
  using reducer_type = ReturnType;
  using value_type   = typename return_type::value_type;

  static auto return_value(ReturnType& return_val, const FunctorType&) {
    return return_val.view();
  }
};

template <class T, class ReturnType, class FunctorType>
struct ParallelReducePolicyType;

template <class PolicyType, class FunctorType>
struct ParallelReducePolicyType<
    std::enable_if_t<Kokkos::is_execution_policy<PolicyType>::value>,
    PolicyType, FunctorType> {
  using policy_type = PolicyType;
  static PolicyType policy(const PolicyType& policy_) { return policy_; }
};

template <class PolicyType, class FunctorType>
struct ParallelReducePolicyType<
    std::enable_if_t<std::is_integral_v<PolicyType>>, PolicyType, FunctorType> {
  using execution_space =
      typename Impl::FunctorPolicyExecutionSpace<FunctorType,
                                                 void>::execution_space;

  using policy_type = Kokkos::RangePolicy<execution_space>;

  static policy_type policy(const PolicyType& policy_) {
    return policy_type(0, policy_);
  }
};

template <class PolicyType, class FunctorType, class ReturnType>
struct ParallelReduceAdaptor {
  using return_value_adapter =
      Impl::ParallelReduceReturnValue<void, ReturnType, FunctorType>;

  // Equivalent to std::get<I>(std::tuple) but callable on the device.
  template <bool B, class T1, class T2>
  static KOKKOS_FUNCTION std::conditional_t<B, T1&&, T2&&> forwarding_switch(
      T1&& v1, T2&& v2) {
    if constexpr (B)
      return static_cast<T1&&>(v1);
    else
      return static_cast<T2&&>(v2);
  }

  static inline void execute_impl(const std::string& label,
                                  const PolicyType& policy,
                                  const FunctorType& functor,
                                  ReturnType& return_value) {
    using PassedReducerType = typename return_value_adapter::reducer_type;
    uint64_t kpID           = 0;

    constexpr bool passed_reducer_type_is_invalid =
        std::is_same_v<InvalidType, PassedReducerType>;
    using TheReducerType = std::conditional_t<passed_reducer_type_is_invalid,
                                              FunctorType, PassedReducerType>;

    using Analysis = FunctorAnalysis<FunctorPatternInterface::REDUCE,
                                     PolicyType, TheReducerType,
                                     typename return_value_adapter::value_type>;
    using CombinedFunctorReducerType =
        CombinedFunctorReducer<FunctorType, typename Analysis::Reducer>;

    CombinedFunctorReducerType functor_reducer(
        functor, typename Analysis::Reducer(
                     forwarding_switch<passed_reducer_type_is_invalid>(
                         functor, return_value)));
    const auto& response = Kokkos::Tools::Impl::begin_parallel_reduce<
        typename return_value_adapter::reducer_type>(policy, functor_reducer,
                                                     label, kpID);
    const auto& inner_policy = response.policy;

    auto closure = construct_with_shared_allocation_tracking_disabled<
        Impl::ParallelReduce<CombinedFunctorReducerType, PolicyType,
                             typename Impl::FunctorPolicyExecutionSpace<
                                 FunctorType, PolicyType>::execution_space>>(
        functor_reducer, inner_policy,
        return_value_adapter::return_value(return_value, functor));
    closure.execute();

    Kokkos::Tools::Impl::end_parallel_reduce<PassedReducerType>(
        inner_policy, functor, label, kpID);
  }

  static constexpr bool is_array_reduction =
      Impl::FunctorAnalysis<
          Impl::FunctorPatternInterface::REDUCE, PolicyType, FunctorType,
          typename return_value_adapter::value_type>::StaticValueSize == 0;

  template <typename Dummy = ReturnType>
  static inline std::enable_if_t<!(is_array_reduction &&
                                   std::is_pointer_v<Dummy>)>
  execute(const std::string& label, const PolicyType& policy,
          const FunctorType& functor, ReturnType& return_value) {
    execute_impl(label, policy, functor, return_value);
  }
};
}  // namespace Impl

//----------------------------------------------------------------------------

/*! \fn void parallel_reduce(label,policy,functor,return_argument)
    \brief Perform a parallel reduction.
    \param label An optional Label giving the call name. Must be able to
   construct a std::string from the argument. \param policy A Kokkos Execution
   Policy, such as an integer, a RangePolicy or a TeamPolicy. \param functor A
   functor with a reduction operator, and optional init, join and final
   functions. \param return_argument A return argument which can be a scalar, a
   View, or a ReducerStruct. This argument can be left out if the functor has a
   final function.
*/

// Parallel Reduce Blocking behavior

namespace Impl {
template <typename T>
struct ReducerHasTestReferenceFunction {
  template <typename E>
  static std::true_type test_func(decltype(&E::references_scalar));
  template <typename E>
  static std::false_type test_func(...);

  enum {
    value = std::is_same_v<std::true_type, decltype(test_func<T>(nullptr))>
  };
};

template <class ExecutionSpace, class Arg>
constexpr std::enable_if_t<
    // constraints only necessary because SFINAE lacks subsumption
    !ReducerHasTestReferenceFunction<Arg>::value &&
        !Kokkos::is_view<Arg>::value,
    // return type:
    bool>
parallel_reduce_needs_fence(ExecutionSpace const&, Arg const&) {
  return true;
}

template <class ExecutionSpace, class Reducer>
constexpr std::enable_if_t<
    // equivalent to:
    // (requires (Reducer const& r) {
    //   { reducer.references_scalar() } -> std::convertible_to<bool>;
    // })
    ReducerHasTestReferenceFunction<Reducer>::value,
    // return type:
    bool>
parallel_reduce_needs_fence(ExecutionSpace const&, Reducer const& reducer) {
  return reducer.references_scalar();
}

template <class ExecutionSpace, class ViewLike>
constexpr std::enable_if_t<
    // requires Kokkos::ViewLike<ViewLike>
    Kokkos::is_view<ViewLike>::value,
    // return type:
    bool>
parallel_reduce_needs_fence(ExecutionSpace const&, ViewLike const&) {
  return false;
}

template <class ExecutionSpace, class... Args>
struct ParallelReduceFence {
  template <class... ArgsDeduced>
  static void fence(const ExecutionSpace& ex, const std::string& name,
                    ArgsDeduced&&... args) {
    if (Impl::parallel_reduce_needs_fence(ex, (ArgsDeduced&&)args...)) {
      ex.fence(name);
    }
  }
};

}  // namespace Impl

/** \brief  Parallel reduction
 *
 * parallel_reduce performs parallel reductions with arbitrary functions - i.e.
 * it is not solely data based. The call expects up to 4 arguments:
 *
 *
 * Example of a parallel_reduce functor for a POD (plain old data) value type:
 * \code
 *  class FunctorType { // For POD value type
 *  public:
 *    using execution_space = ...;
 *    using value_type = <podType>;
 *    void operator()( <intType> iwork , <podType> & update ) const ;
 *    void init( <podType> & update ) const ;
 *    void join(       <podType> & update ,
 *               const <podType> & input ) const ;
 *
 *    void final( <podType> & update ) const ;
 *  };
 * \endcode
 *
 * Example of a parallel_reduce functor for an array of POD (plain old data)
 * values:
 * \code
 *  class FunctorType { // For array of POD value
 *  public:
 *    using execution_space = ...;
 *    using value_type = <podType>[];
 *    void operator()( <intType> , <podType> update[] ) const ;
 *    void init( <podType> update[] ) const ;
 *    void join(       <podType> update[] ,
 *               const <podType> input[] ) const ;
 *
 *    void final( <podType> update[] ) const ;
 *  };
 * \endcode
 */

// ReturnValue is scalar or array: take by reference

template <class PolicyType, class FunctorType, class ReturnType>
inline std::enable_if_t<Kokkos::is_execution_policy<PolicyType>::value &&
                        !(Kokkos::is_view<ReturnType>::value ||
                          Kokkos::is_reducer<ReturnType>::value ||
                          std::is_pointer_v<ReturnType>)>
parallel_reduce(const std::string& label, const PolicyType& policy,
                const FunctorType& functor, ReturnType& return_value) {
  static_assert(
      !std::is_const_v<ReturnType>,
      "A const reduction result type is only allowed for a View, pointer or "
      "reducer return type!");

  Impl::ParallelReduceAdaptor<PolicyType, FunctorType, ReturnType>::execute(
      label, policy, functor, return_value);
  Impl::ParallelReduceFence<typename PolicyType::execution_space, ReturnType>::
      fence(
          policy.space(),
          "Kokkos::parallel_reduce: fence due to result being value, not view",
          return_value);
}

template <class PolicyType, class FunctorType, class ReturnType>
inline std::enable_if_t<Kokkos::is_execution_policy<PolicyType>::value &&
                        !(Kokkos::is_view<ReturnType>::value ||
                          Kokkos::is_reducer<ReturnType>::value ||
                          std::is_pointer_v<ReturnType>)>
parallel_reduce(const PolicyType& policy, const FunctorType& functor,
                ReturnType& return_value) {
  parallel_reduce("", policy, functor, return_value);
}

template <class FunctorType, class ReturnType>
inline std::enable_if_t<!(Kokkos::is_view<ReturnType>::value ||
                          Kokkos::is_reducer<ReturnType>::value ||
                          std::is_pointer_v<ReturnType>)>
parallel_reduce(const std::string& label, const size_t& work_count,
                const FunctorType& functor, ReturnType& return_value) {
  using policy_type =
      typename Impl::ParallelReducePolicyType<void, size_t,
                                              FunctorType>::policy_type;

  parallel_reduce(label, policy_type(0, work_count), functor, return_value);
}

template <class FunctorType, class ReturnType>
inline std::enable_if_t<!(Kokkos::is_view<ReturnType>::value ||
                          Kokkos::is_reducer<ReturnType>::value ||
                          std::is_pointer_v<ReturnType>)>
parallel_reduce(const size_t& work_count, const FunctorType& functor,
                ReturnType& return_value) {
  parallel_reduce("", work_count, functor, return_value);
}

// ReturnValue as View or Reducer: take by copy to allow for inline construction

template <class PolicyType, class FunctorType, class ReturnType>
inline std::enable_if_t<Kokkos::is_execution_policy<PolicyType>::value &&
                        (Kokkos::is_view<ReturnType>::value ||
                         Kokkos::is_reducer<ReturnType>::value ||
                         std::is_pointer_v<ReturnType>)>
parallel_reduce(const std::string& label, const PolicyType& policy,
                const FunctorType& functor, const ReturnType& return_value) {
  ReturnType return_value_impl = return_value;
  Impl::ParallelReduceAdaptor<PolicyType, FunctorType, ReturnType>::execute(
      label, policy, functor, return_value_impl);
  Impl::ParallelReduceFence<typename PolicyType::execution_space, ReturnType>::
      fence(
          policy.space(),
          "Kokkos::parallel_reduce: fence due to result being value, not view",
          return_value);
}

template <class PolicyType, class FunctorType, class ReturnType>
inline std::enable_if_t<Kokkos::is_execution_policy<PolicyType>::value &&
                        (Kokkos::is_view<ReturnType>::value ||
                         Kokkos::is_reducer<ReturnType>::value ||
                         std::is_pointer_v<ReturnType>)>
parallel_reduce(const PolicyType& policy, const FunctorType& functor,
                const ReturnType& return_value) {
  parallel_reduce("", policy, functor, return_value);
}

template <class FunctorType, class ReturnType>
inline std::enable_if_t<Kokkos::is_view<ReturnType>::value ||
                        Kokkos::is_reducer<ReturnType>::value ||
                        std::is_pointer_v<ReturnType>>
parallel_reduce(const std::string& label, const size_t& work_count,
                const FunctorType& functor, const ReturnType& return_value) {
  using policy_type =
      typename Impl::ParallelReducePolicyType<void, size_t,
                                              FunctorType>::policy_type;

  parallel_reduce(label, policy_type(0, work_count), functor, return_value);
}

template <class FunctorType, class ReturnType>
inline std::enable_if_t<Kokkos::is_view<ReturnType>::value ||
                        Kokkos::is_reducer<ReturnType>::value ||
                        std::is_pointer_v<ReturnType>>
parallel_reduce(const size_t& work_count, const FunctorType& functor,
                const ReturnType& return_value) {
  parallel_reduce("", work_count, functor, return_value);
}

// No Return Argument

template <class PolicyType, class FunctorType>
inline void parallel_reduce(
    const std::string& label, const PolicyType& policy,
    const FunctorType& functor,
    std::enable_if_t<Kokkos::is_execution_policy<PolicyType>::value>* =
        nullptr) {
  using FunctorAnalysis =
      Impl::FunctorAnalysis<Impl::FunctorPatternInterface::REDUCE, PolicyType,
                            FunctorType, void>;
  using value_type = std::conditional_t<(FunctorAnalysis::StaticValueSize != 0),
                                        typename FunctorAnalysis::value_type,
                                        typename FunctorAnalysis::pointer_type>;

  static_assert(
      FunctorAnalysis::has_final_member_function,
      "Calling parallel_reduce without either return value or final function.");

  using result_view_type =
      Kokkos::View<value_type, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
  result_view_type result_view;

  Impl::ParallelReduceAdaptor<PolicyType, FunctorType,
                              result_view_type>::execute(label, policy, functor,
                                                         result_view);
}

template <class PolicyType, class FunctorType>
inline void parallel_reduce(
    const PolicyType& policy, const FunctorType& functor,
    std::enable_if_t<Kokkos::is_execution_policy<PolicyType>::value>* =
        nullptr) {
  parallel_reduce("", policy, functor);
}

template <class FunctorType>
inline void parallel_reduce(const std::string& label, const size_t& work_count,
                            const FunctorType& functor) {
  using policy_type =
      typename Impl::ParallelReducePolicyType<void, size_t,
                                              FunctorType>::policy_type;

  parallel_reduce(label, policy_type(0, work_count), functor);
}

template <class FunctorType>
inline void parallel_reduce(const size_t& work_count,
                            const FunctorType& functor) {
  parallel_reduce("", work_count, functor);
}

}  // namespace Kokkos

#endif  // KOKKOS_PARALLEL_REDUCE_HPP
