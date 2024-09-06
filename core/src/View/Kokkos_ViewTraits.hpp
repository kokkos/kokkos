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
#ifndef KOKKOS_VIEWTRAITS_HPP
#define KOKKOS_VIEWTRAITS_HPP

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_HostSpace.hpp>
#include <Kokkos_MemoryTraits.hpp>
#include <Kokkos_ExecPolicy.hpp>
#include <View/Hooks/Kokkos_ViewHooks.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template <class DataType>
struct ViewArrayAnalysis;

template <class DataType, class ArrayLayout,
          typename ValueType =
              typename ViewArrayAnalysis<DataType>::non_const_value_type>
struct ViewDataAnalysis;

template <class, class...>
class ViewMapping {
 public:
  enum : bool { is_assignable_data_type = false };
  enum : bool { is_assignable = false };
};

template <typename IntType>
constexpr KOKKOS_INLINE_FUNCTION std::size_t count_valid_integers(
    const IntType i0, const IntType i1, const IntType i2, const IntType i3,
    const IntType i4, const IntType i5, const IntType i6, const IntType i7) {
  static_assert(std::is_integral_v<IntType>,
                "count_valid_integers() must have integer arguments.");

  return (i0 != KOKKOS_INVALID_INDEX) + (i1 != KOKKOS_INVALID_INDEX) +
         (i2 != KOKKOS_INVALID_INDEX) + (i3 != KOKKOS_INVALID_INDEX) +
         (i4 != KOKKOS_INVALID_INDEX) + (i5 != KOKKOS_INVALID_INDEX) +
         (i6 != KOKKOS_INVALID_INDEX) + (i7 != KOKKOS_INVALID_INDEX);
}

// FIXME Ideally, we would not instantiate this function for every possible View
// type. We should be able to only pass "extent" when we use mdspan.
template <typename View>
KOKKOS_INLINE_FUNCTION void runtime_check_rank(
    const View&, const bool is_void_spec, const size_t i0, const size_t i1,
    const size_t i2, const size_t i3, const size_t i4, const size_t i5,
    const size_t i6, const size_t i7, const char* label) {
  (void)(label);

  if (is_void_spec) {
    const size_t num_passed_args =
        count_valid_integers(i0, i1, i2, i3, i4, i5, i6, i7);
    // We either allow to pass as many extents as the dynamic rank is, or
    // as many extents as the total rank is. In the latter case, the given
    // extents for the static dimensions must match the
    // compile-time extents.
    constexpr int rank            = View::rank();
    constexpr int dyn_rank        = View::rank_dynamic();
    const bool n_args_is_dyn_rank = num_passed_args == dyn_rank;
    const bool n_args_is_rank     = num_passed_args == rank;

    if constexpr (rank != dyn_rank) {
      if (n_args_is_rank) {
        size_t new_extents[8] = {i0, i1, i2, i3, i4, i5, i6, i7};
        for (int i = dyn_rank; i < rank; ++i)
          if (new_extents[i] != View::static_extent(i)) {
            KOKKOS_IF_ON_HOST(
                const std::string message =
                    "The specified run-time extent for Kokkos::View '" +
                    std::string(label) +
                    "' does not match the compile-time extent in dimension " +
                    std::to_string(i) + ". The given extent is " +
                    std::to_string(new_extents[i]) + " but should be " +
                    std::to_string(View::static_extent(i)) + ".\n";
                Kokkos::abort(message.c_str());)
            KOKKOS_IF_ON_DEVICE(
                Kokkos::abort(
                    "The specified run-time extents for a Kokkos::View "
                    "do not match the compile-time extents.");)
          }
      }
    }

    if (!n_args_is_dyn_rank && !n_args_is_rank) {
      KOKKOS_IF_ON_HOST(
          const std::string message =
              "Constructor for Kokkos::View '" + std::string(label) +
              "' has mismatched number of arguments. The number "
              "of arguments = " +
              std::to_string(num_passed_args) +
              " neither matches the dynamic rank = " +
              std::to_string(dyn_rank) +
              " nor the total rank = " + std::to_string(rank) + "\n";
          Kokkos::abort(message.c_str());)
      KOKKOS_IF_ON_DEVICE(Kokkos::abort("Constructor for Kokkos View has "
                                        "mismatched number of arguments.");)
    }
  }
}

} /* namespace Impl */
} /* namespace Kokkos */

// Class to provide a uniform type
namespace Kokkos {
namespace Impl {
template <class ViewType, int Traits = 0>
struct ViewUniformType;
}
}  // namespace Kokkos

namespace Kokkos {

#ifdef KOKKOS_ENABLE_IMPL_MDSPAN
namespace Impl {
struct UnsupportedKokkosArrayLayout;

template <class Traits, class Enabled = void>
struct MDSpanViewTraits {
  using mdspan_type = UnsupportedKokkosArrayLayout;
};

// "Natural" mdspan for a view if the View's ArrayLayout is supported.
template <class Traits>
struct MDSpanViewTraits<Traits, std::void_t<typename LayoutFromArrayLayout<
                                    typename Traits::array_layout>::type>> {
  using index_type = std::size_t;
  using extents_type =
      typename Impl::ExtentsFromDataType<index_type,
                                         typename Traits::data_type>::type;
  using mdspan_layout_type =
      typename LayoutFromArrayLayout<typename Traits::array_layout>::type;
  using accessor_type =
      SpaceAwareAccessor<typename Traits::memory_space,
                         Kokkos::default_accessor<typename Traits::value_type>>;
  using mdspan_type = mdspan<typename Traits::value_type, extents_type,
                             mdspan_layout_type, accessor_type>;
};
}  // namespace Impl
#endif  // KOKKOS_ENABLE_IMPL_MDSPAN

/** \class ViewTraits
 *  \brief Traits class for accessing attributes of a View.
 *
 * This is an implementation detail of View.  It is only of interest
 * to developers implementing a new specialization of View.
 *
 * Template argument options:
 *   - View< DataType >
 *   - View< DataType , Space >
 *   - View< DataType , Space , MemoryTraits >
 *   - View< DataType , ArrayLayout >
 *   - View< DataType , ArrayLayout , Space >
 *   - View< DataType , ArrayLayout , MemoryTraits >
 *   - View< DataType , ArrayLayout , Space , MemoryTraits >
 *   - View< DataType , MemoryTraits >
 */

template <class DataType, class... Properties>
struct ViewTraits;

template <>
struct ViewTraits<void> {
  using execution_space = void;
  using memory_space    = void;
  using HostMirrorSpace = void;
  using array_layout    = void;
  using memory_traits   = void;
  using specialize      = void;
  using hooks_policy    = void;
};

template <class... Prop>
struct ViewTraits<void, void, Prop...> {
  // Ignore an extraneous 'void'
  using execution_space = typename ViewTraits<void, Prop...>::execution_space;
  using memory_space    = typename ViewTraits<void, Prop...>::memory_space;
  using HostMirrorSpace = typename ViewTraits<void, Prop...>::HostMirrorSpace;
  using array_layout    = typename ViewTraits<void, Prop...>::array_layout;
  using memory_traits   = typename ViewTraits<void, Prop...>::memory_traits;
  using specialize      = typename ViewTraits<void, Prop...>::specialize;
  using hooks_policy    = typename ViewTraits<void, Prop...>::hooks_policy;
};

template <class HooksPolicy, class... Prop>
struct ViewTraits<
    std::enable_if_t<Kokkos::Experimental::is_hooks_policy<HooksPolicy>::value>,
    HooksPolicy, Prop...> {
  using execution_space = typename ViewTraits<void, Prop...>::execution_space;
  using memory_space    = typename ViewTraits<void, Prop...>::memory_space;
  using HostMirrorSpace = typename ViewTraits<void, Prop...>::HostMirrorSpace;
  using array_layout    = typename ViewTraits<void, Prop...>::array_layout;
  using memory_traits   = typename ViewTraits<void, Prop...>::memory_traits;
  using specialize      = typename ViewTraits<void, Prop...>::specialize;
  using hooks_policy    = HooksPolicy;
};

template <class ArrayLayout, class... Prop>
struct ViewTraits<std::enable_if_t<Kokkos::is_array_layout<ArrayLayout>::value>,
                  ArrayLayout, Prop...> {
  // Specify layout, keep subsequent space and memory traits arguments

  using execution_space = typename ViewTraits<void, Prop...>::execution_space;
  using memory_space    = typename ViewTraits<void, Prop...>::memory_space;
  using HostMirrorSpace = typename ViewTraits<void, Prop...>::HostMirrorSpace;
  using array_layout    = ArrayLayout;
  using memory_traits   = typename ViewTraits<void, Prop...>::memory_traits;
  using specialize      = typename ViewTraits<void, Prop...>::specialize;
  using hooks_policy    = typename ViewTraits<void, Prop...>::hooks_policy;
};

template <class Space, class... Prop>
struct ViewTraits<std::enable_if_t<Kokkos::is_space<Space>::value>, Space,
                  Prop...> {
  // Specify Space, memory traits should be the only subsequent argument.

  static_assert(
      std::is_same_v<typename ViewTraits<void, Prop...>::execution_space,
                     void> &&
          std::is_same_v<typename ViewTraits<void, Prop...>::memory_space,
                         void> &&
          std::is_same_v<typename ViewTraits<void, Prop...>::HostMirrorSpace,
                         void> &&
          std::is_same_v<typename ViewTraits<void, Prop...>::array_layout,
                         void>,
      "Only one View Execution or Memory Space template argument");

  using execution_space = typename Space::execution_space;
  using memory_space    = typename Space::memory_space;
  using HostMirrorSpace =
      typename Kokkos::Impl::HostMirror<Space>::Space::memory_space;
  using array_layout  = typename execution_space::array_layout;
  using memory_traits = typename ViewTraits<void, Prop...>::memory_traits;
  using specialize    = typename ViewTraits<void, Prop...>::specialize;
  using hooks_policy  = typename ViewTraits<void, Prop...>::hooks_policy;
};

template <class MemoryTraits, class... Prop>
struct ViewTraits<
    std::enable_if_t<Kokkos::is_memory_traits<MemoryTraits>::value>,
    MemoryTraits, Prop...> {
  // Specify memory trait, should not be any subsequent arguments

  static_assert(
      std::is_same_v<typename ViewTraits<void, Prop...>::execution_space,
                     void> &&
          std::is_same_v<typename ViewTraits<void, Prop...>::memory_space,
                         void> &&
          std::is_same_v<typename ViewTraits<void, Prop...>::array_layout,
                         void> &&
          std::is_same_v<typename ViewTraits<void, Prop...>::memory_traits,
                         void> &&
          std::is_same_v<typename ViewTraits<void, Prop...>::hooks_policy,
                         void>,
      "MemoryTrait is the final optional template argument for a View");

  using execution_space = void;
  using memory_space    = void;
  using HostMirrorSpace = void;
  using array_layout    = void;
  using memory_traits   = MemoryTraits;
  using specialize      = void;
  using hooks_policy    = void;
};

template <class DataType, class... Properties>
struct ViewTraits {
 private:
  // Unpack the properties arguments
  using prop = ViewTraits<void, Properties...>;

  using ExecutionSpace =
      std::conditional_t<!std::is_void_v<typename prop::execution_space>,
                         typename prop::execution_space,
                         Kokkos::DefaultExecutionSpace>;

  using MemorySpace =
      std::conditional_t<!std::is_void_v<typename prop::memory_space>,
                         typename prop::memory_space,
                         typename ExecutionSpace::memory_space>;

  using ArrayLayout =
      std::conditional_t<!std::is_void_v<typename prop::array_layout>,
                         typename prop::array_layout,
                         typename ExecutionSpace::array_layout>;

  using HostMirrorSpace = std::conditional_t<
      !std::is_void_v<typename prop::HostMirrorSpace>,
      typename prop::HostMirrorSpace,
      typename Kokkos::Impl::HostMirror<ExecutionSpace>::Space>;

  using MemoryTraits =
      std::conditional_t<!std::is_void_v<typename prop::memory_traits>,
                         typename prop::memory_traits,
                         typename Kokkos::MemoryManaged>;

  using HooksPolicy =
      std::conditional_t<!std::is_void_v<typename prop::hooks_policy>,
                         typename prop::hooks_policy,
                         Kokkos::Experimental::DefaultViewHooks>;

  // Analyze data type's properties,
  // May be specialized based upon the layout and value type
  using data_analysis = Kokkos::Impl::ViewDataAnalysis<DataType, ArrayLayout>;

 public:
  //------------------------------------
  // Data type traits:

  using data_type           = typename data_analysis::type;
  using const_data_type     = typename data_analysis::const_type;
  using non_const_data_type = typename data_analysis::non_const_type;

  //------------------------------------
  // Compatible array of trivial type traits:

  using scalar_array_type = typename data_analysis::scalar_array_type;
  using const_scalar_array_type =
      typename data_analysis::const_scalar_array_type;
  using non_const_scalar_array_type =
      typename data_analysis::non_const_scalar_array_type;

  //------------------------------------
  // Value type traits:

  using value_type           = typename data_analysis::value_type;
  using const_value_type     = typename data_analysis::const_value_type;
  using non_const_value_type = typename data_analysis::non_const_value_type;

  //------------------------------------
  // Mapping traits:

  using array_layout = ArrayLayout;
  using dimension    = typename data_analysis::dimension;

  using specialize = std::conditional_t<
      std::is_void_v<typename data_analysis::specialize>,
      typename prop::specialize,
      typename data_analysis::specialize>; /* mapping specialization tag */

  static constexpr unsigned rank         = dimension::rank;
  static constexpr unsigned rank_dynamic = dimension::rank_dynamic;

  //------------------------------------
  // Execution space, memory space, memory access traits, and host mirror space.

  using execution_space   = ExecutionSpace;
  using memory_space      = MemorySpace;
  using device_type       = Kokkos::Device<ExecutionSpace, MemorySpace>;
  using memory_traits     = MemoryTraits;
  using host_mirror_space = HostMirrorSpace;
  using hooks_policy      = HooksPolicy;

  using size_type = typename MemorySpace::size_type;

  enum { is_hostspace = std::is_same_v<MemorySpace, HostSpace> };
  enum { is_managed = MemoryTraits::is_unmanaged == 0 };
  enum { is_random_access = MemoryTraits::is_random_access == 1 };

  //------------------------------------
};

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Impl {
template <class ValueType, class TypeList>
struct TypeListToViewTraits;

template <class ValueType, class... Properties>
struct TypeListToViewTraits<ValueType, Kokkos::Impl::type_list<Properties...>> {
  using type = ViewTraits<ValueType, Properties...>;
};

// It is not safe to assume that subviews of views with the Aligned memory trait
// are also aligned. Hence, just remove that attribute for subviews.
template <class D, class... P>
struct RemoveAlignedMemoryTrait {
 private:
  using type_list_in  = Kokkos::Impl::type_list<P...>;
  using memory_traits = typename ViewTraits<D, P...>::memory_traits;
  using type_list_in_wo_memory_traits =
      typename Kokkos::Impl::type_list_remove_first<memory_traits,
                                                    type_list_in>::type;
  using new_memory_traits =
      Kokkos::MemoryTraits<memory_traits::impl_value & ~Kokkos::Aligned>;
  using new_type_list = typename Kokkos::Impl::concat_type_list<
      type_list_in_wo_memory_traits,
      Kokkos::Impl::type_list<new_memory_traits>>::type;

 public:
  using type = typename TypeListToViewTraits<D, new_type_list>::type;
};
}  // namespace Impl

} /* namespace Kokkos */

#endif /* KOKKOS_VIEWTRAITS_HPP */
