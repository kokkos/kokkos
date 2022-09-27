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

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#include <Kokkos_Macros.hpp>
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif
#ifndef KOKKOS_RESIZEREALLOC_HPP_
#define KOKKOS_RESIZEREALLOC_HPP_

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

namespace Impl {
template <typename ViewType>
bool size_mismatch(const ViewType& view, unsigned int max_extent,
                   const size_t new_extents[8]) {
  for (unsigned int dim = 0; dim < max_extent; ++dim)
    if (new_extents[dim] != view.extent(dim)) {
      return true;
    }
  for (unsigned int dim = max_extent; dim < 8; ++dim)
    if (new_extents[dim] != KOKKOS_IMPL_CTOR_DEFAULT_ARG) {
      return true;
    }
  return false;
}

}  // namespace Impl

/** \brief  Resize a view with copying old data to new data at the corresponding
 * indices. */
template <class T, class... P, class... ViewCtorArgs>
inline typename std::enable_if<
    std::is_same<typename Kokkos::View<T, P...>::array_layout,
                 Kokkos::LayoutLeft>::value ||
    std::is_same<typename Kokkos::View<T, P...>::array_layout,
                 Kokkos::LayoutRight>::value>::type
impl_resize(const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop,
            Kokkos::View<T, P...>& v, const size_t n0, const size_t n1,
            const size_t n2, const size_t n3, const size_t n4, const size_t n5,
            const size_t n6, const size_t n7) {
  using view_type        = Kokkos::View<T, P...>;
  using alloc_prop_input = Impl::ViewCtorProp<ViewCtorArgs...>;

  static_assert(Kokkos::ViewTraits<T, P...>::is_managed,
                "Can only resize managed views");
  static_assert(!alloc_prop_input::has_label,
                "The view constructor arguments passed to Kokkos::resize "
                "must not include a label!");
  static_assert(!alloc_prop_input::has_pointer,
                "The view constructor arguments passed to Kokkos::resize must "
                "not include a pointer!");
  static_assert(!alloc_prop_input::has_memory_space,
                "The view constructor arguments passed to Kokkos::resize must "
                "not include a memory space instance!");

  // TODO (mfh 27 Jun 2017) If the old View has enough space but just
  // different dimensions (e.g., if the product of the dimensions,
  // including extra space for alignment, will not change), then
  // consider just reusing storage.  For now, Kokkos always
  // reallocates if any of the dimensions change, even if the old View
  // has enough space.

  const size_t new_extents[8] = {n0, n1, n2, n3, n4, n5, n6, n7};
  const bool sizeMismatch = Impl::size_mismatch(v, v.rank_dynamic, new_extents);

  if (sizeMismatch) {
    auto prop_copy = Impl::with_properties_if_unset(
        arg_prop, typename view_type::execution_space{}, v.label());

    view_type v_resized(prop_copy, n0, n1, n2, n3, n4, n5, n6, n7);

    if constexpr (alloc_prop_input::has_execution_space)
      Kokkos::Impl::ViewRemap<view_type, view_type>(
          v_resized, v, Impl::get_property<Impl::ExecutionSpaceTag>(prop_copy));
    else {
      Kokkos::Impl::ViewRemap<view_type, view_type>(v_resized, v);
      Kokkos::fence("Kokkos::resize(View)");
    }

    v = v_resized;
  }
}

template <class T, class... P, class... ViewCtorArgs>
inline std::enable_if_t<
    std::is_same<typename Kokkos::View<T, P...>::array_layout,
                 Kokkos::LayoutLeft>::value ||
    std::is_same<typename Kokkos::View<T, P...>::array_layout,
                 Kokkos::LayoutRight>::value>
resize(const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop,
       Kokkos::View<T, P...>& v, const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG) {
  impl_resize(arg_prop, v, n0, n1, n2, n3, n4, n5, n6, n7);
}

template <class T, class... P>
inline std::enable_if_t<
    std::is_same<typename Kokkos::View<T, P...>::array_layout,
                 Kokkos::LayoutLeft>::value ||
    std::is_same<typename Kokkos::View<T, P...>::array_layout,
                 Kokkos::LayoutRight>::value>
resize(Kokkos::View<T, P...>& v, const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG) {
  impl_resize(Impl::ViewCtorProp<>{}, v, n0, n1, n2, n3, n4, n5, n6, n7);
}

template <class I, class T, class... P>
inline std::enable_if_t<
    (Impl::is_view_ctor_property<I>::value ||
     Kokkos::is_execution_space<I>::value) &&
    (std::is_same<typename Kokkos::View<T, P...>::array_layout,
                  Kokkos::LayoutLeft>::value ||
     std::is_same<typename Kokkos::View<T, P...>::array_layout,
                  Kokkos::LayoutRight>::value)>
resize(const I& arg_prop, Kokkos::View<T, P...>& v,
       const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG) {
  impl_resize(Kokkos::view_alloc(arg_prop), v, n0, n1, n2, n3, n4, n5, n6, n7);
}

template <class T, class... P, class... ViewCtorArgs>
inline std::enable_if_t<
    std::is_same<typename Kokkos::View<T, P...>::array_layout,
                 Kokkos::LayoutLeft>::value ||
    std::is_same<typename Kokkos::View<T, P...>::array_layout,
                 Kokkos::LayoutRight>::value ||
    std::is_same<typename Kokkos::View<T, P...>::array_layout,
                 Kokkos::LayoutStride>::value ||
    is_layouttiled<typename Kokkos::View<T, P...>::array_layout>::value>
impl_resize(const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop,
            Kokkos::View<T, P...>& v,
            const typename Kokkos::View<T, P...>::array_layout& layout) {
  using view_type        = Kokkos::View<T, P...>;
  using alloc_prop_input = Impl::ViewCtorProp<ViewCtorArgs...>;

  static_assert(Kokkos::ViewTraits<T, P...>::is_managed,
                "Can only resize managed views");
  static_assert(!alloc_prop_input::has_label,
                "The view constructor arguments passed to Kokkos::resize "
                "must not include a label!");
  static_assert(!alloc_prop_input::has_pointer,
                "The view constructor arguments passed to Kokkos::resize must "
                "not include a pointer!");
  static_assert(!alloc_prop_input::has_memory_space,
                "The view constructor arguments passed to Kokkos::resize must "
                "not include a memory space instance!");

  if (v.layout() != layout) {
    auto prop_copy = Impl::with_properties_if_unset(arg_prop, v.label());

    view_type v_resized(prop_copy, layout);

    if constexpr (alloc_prop_input::has_execution_space)
      Kokkos::Impl::ViewRemap<view_type, view_type>(
          v_resized, v, Impl::get_property<Impl::ExecutionSpaceTag>(arg_prop));
    else {
      Kokkos::Impl::ViewRemap<view_type, view_type>(v_resized, v);
      Kokkos::fence("Kokkos::resize(View)");
    }

    v = v_resized;
  }
}

// FIXME User-provided (custom) layouts are not required to have a comparison
// operator. Hence, there is no way to check if the requested layout is actually
// the same as the existing one.
template <class T, class... P, class... ViewCtorArgs>
inline std::enable_if_t<
    !(std::is_same<typename Kokkos::View<T, P...>::array_layout,
                   Kokkos::LayoutLeft>::value ||
      std::is_same<typename Kokkos::View<T, P...>::array_layout,
                   Kokkos::LayoutRight>::value ||
      std::is_same<typename Kokkos::View<T, P...>::array_layout,
                   Kokkos::LayoutStride>::value ||
      is_layouttiled<typename Kokkos::View<T, P...>::array_layout>::value)>
impl_resize(const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop,
            Kokkos::View<T, P...>& v,
            const typename Kokkos::View<T, P...>::array_layout& layout) {
  using view_type        = Kokkos::View<T, P...>;
  using alloc_prop_input = Impl::ViewCtorProp<ViewCtorArgs...>;

  static_assert(Kokkos::ViewTraits<T, P...>::is_managed,
                "Can only resize managed views");
  static_assert(!alloc_prop_input::has_label,
                "The view constructor arguments passed to Kokkos::resize "
                "must not include a label!");
  static_assert(!alloc_prop_input::has_pointer,
                "The view constructor arguments passed to Kokkos::resize must "
                "not include a pointer!");
  static_assert(!alloc_prop_input::has_memory_space,
                "The view constructor arguments passed to Kokkos::resize must "
                "not include a memory space instance!");

  auto prop_copy = Impl::with_properties_if_unset(arg_prop, v.label());

  view_type v_resized(prop_copy, layout);

  if constexpr (alloc_prop_input::has_execution_space)
    Kokkos::Impl::ViewRemap<view_type, view_type>(
        v_resized, v, Impl::get_property<Impl::ExecutionSpaceTag>(arg_prop));
  else {
    Kokkos::Impl::ViewRemap<view_type, view_type>(v_resized, v);
    Kokkos::fence("Kokkos::resize(View)");
  }

  v = v_resized;
}

template <class T, class... P, class... ViewCtorArgs>
inline void resize(const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop,
                   Kokkos::View<T, P...>& v,
                   const typename Kokkos::View<T, P...>::array_layout& layout) {
  impl_resize(arg_prop, v, layout);
}

template <class I, class T, class... P>
inline std::enable_if_t<Impl::is_view_ctor_property<I>::value ||
                        Kokkos::is_execution_space<I>::value>
resize(const I& arg_prop, Kokkos::View<T, P...>& v,
       const typename Kokkos::View<T, P...>::array_layout& layout) {
  impl_resize(arg_prop, v, layout);
}

template <class ExecutionSpace, class T, class... P>
inline void resize(const ExecutionSpace& exec_space, Kokkos::View<T, P...>& v,
                   const typename Kokkos::View<T, P...>::array_layout& layout) {
  impl_resize(Impl::ViewCtorProp<>(), exec_space, v, layout);
}

template <class T, class... P>
inline void resize(Kokkos::View<T, P...>& v,
                   const typename Kokkos::View<T, P...>::array_layout& layout) {
  impl_resize(Impl::ViewCtorProp<>{}, v, layout);
}

/** \brief  Resize a view with discarding old data. */
template <class T, class... P, class... ViewCtorArgs>
inline std::enable_if_t<
    std::is_same<typename Kokkos::View<T, P...>::array_layout,
                 Kokkos::LayoutLeft>::value ||
    std::is_same<typename Kokkos::View<T, P...>::array_layout,
                 Kokkos::LayoutRight>::value>
impl_realloc(Kokkos::View<T, P...>& v, const size_t n0, const size_t n1,
             const size_t n2, const size_t n3, const size_t n4, const size_t n5,
             const size_t n6, const size_t n7,
             const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop) {
  using view_type        = Kokkos::View<T, P...>;
  using alloc_prop_input = Impl::ViewCtorProp<ViewCtorArgs...>;

  static_assert(Kokkos::ViewTraits<T, P...>::is_managed,
                "Can only realloc managed views");
  static_assert(!alloc_prop_input::has_label,
                "The view constructor arguments passed to Kokkos::realloc must "
                "not include a label!");
  static_assert(!alloc_prop_input::has_pointer,
                "The view constructor arguments passed to Kokkos::realloc must "
                "not include a pointer!");
  static_assert(!alloc_prop_input::has_memory_space,
                "The view constructor arguments passed to Kokkos::realloc must "
                "not include a memory space instance!");

  const size_t new_extents[8] = {n0, n1, n2, n3, n4, n5, n6, n7};
  const bool sizeMismatch = Impl::size_mismatch(v, v.rank_dynamic, new_extents);

  if (sizeMismatch) {
    auto arg_prop_copy = Impl::with_properties_if_unset(arg_prop, v.label());
    v = view_type();  // Best effort to deallocate in case no other view refers
                      // to the shared allocation
    v = view_type(arg_prop_copy, n0, n1, n2, n3, n4, n5, n6, n7);
  } else if (alloc_prop_input::initialize) {
    if constexpr (alloc_prop_input::has_execution_space) {
      const auto& exec_space =
          Impl::get_property<Impl::ExecutionSpaceTag>(arg_prop);
      Kokkos::deep_copy(exec_space, v, typename view_type::value_type{});
    } else
      Kokkos::deep_copy(v, typename view_type::value_type{});
  }
}

template <class T, class... P, class... ViewCtorArgs>
inline std::enable_if_t<
    std::is_same<typename Kokkos::View<T, P...>::array_layout,
                 Kokkos::LayoutLeft>::value ||
    std::is_same<typename Kokkos::View<T, P...>::array_layout,
                 Kokkos::LayoutRight>::value>
realloc(const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop,
        Kokkos::View<T, P...>& v,
        const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG) {
  impl_realloc(v, n0, n1, n2, n3, n4, n5, n6, n7, arg_prop);
}

template <class T, class... P>
inline std::enable_if_t<
    std::is_same<typename Kokkos::View<T, P...>::array_layout,
                 Kokkos::LayoutLeft>::value ||
    std::is_same<typename Kokkos::View<T, P...>::array_layout,
                 Kokkos::LayoutRight>::value>
realloc(Kokkos::View<T, P...>& v,
        const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG) {
  impl_realloc(v, n0, n1, n2, n3, n4, n5, n6, n7, Impl::ViewCtorProp<>{});
}

template <class I, class T, class... P>
inline std::enable_if_t<
    Impl::is_view_ctor_property<I>::value &&
    (std::is_same<typename Kokkos::View<T, P...>::array_layout,
                  Kokkos::LayoutLeft>::value ||
     std::is_same<typename Kokkos::View<T, P...>::array_layout,
                  Kokkos::LayoutRight>::value)>
realloc(const I& arg_prop, Kokkos::View<T, P...>& v,
        const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG) {
  impl_realloc(v, n0, n1, n2, n3, n4, n5, n6, n7, Kokkos::view_alloc(arg_prop));
}

template <class T, class... P, class... ViewCtorArgs>
inline std::enable_if_t<
    std::is_same<typename Kokkos::View<T, P...>::array_layout,
                 Kokkos::LayoutLeft>::value ||
    std::is_same<typename Kokkos::View<T, P...>::array_layout,
                 Kokkos::LayoutRight>::value ||
    std::is_same<typename Kokkos::View<T, P...>::array_layout,
                 Kokkos::LayoutStride>::value ||
    is_layouttiled<typename Kokkos::View<T, P...>::array_layout>::value>
impl_realloc(Kokkos::View<T, P...>& v,
             const typename Kokkos::View<T, P...>::array_layout& layout,
             const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop) {
  using view_type        = Kokkos::View<T, P...>;
  using alloc_prop_input = Impl::ViewCtorProp<ViewCtorArgs...>;

  static_assert(Kokkos::ViewTraits<T, P...>::is_managed,
                "Can only realloc managed views");
  static_assert(!alloc_prop_input::has_label,
                "The view constructor arguments passed to Kokkos::realloc must "
                "not include a label!");
  static_assert(!alloc_prop_input::has_pointer,
                "The view constructor arguments passed to Kokkos::realloc must "
                "not include a pointer!");
  static_assert(!alloc_prop_input::has_memory_space,
                "The view constructor arguments passed to Kokkos::realloc must "
                "not include a memory space instance!");

  if (v.layout() != layout) {
    v = view_type();  // Deallocate first, if the only view to allocation
    v = view_type(arg_prop, layout);
  } else if (alloc_prop_input::initialize) {
    if constexpr (alloc_prop_input::has_execution_space) {
      const auto& exec_space =
          Impl::get_property<Impl::ExecutionSpaceTag>(arg_prop);
      Kokkos::deep_copy(exec_space, v, typename view_type::value_type{});
    } else
      Kokkos::deep_copy(v, typename view_type::value_type{});
  }
}

// FIXME User-provided (custom) layouts are not required to have a comparison
// operator. Hence, there is no way to check if the requested layout is actually
// the same as the existing one.
template <class T, class... P, class... ViewCtorArgs>
inline std::enable_if_t<
    !(std::is_same<typename Kokkos::View<T, P...>::array_layout,
                   Kokkos::LayoutLeft>::value ||
      std::is_same<typename Kokkos::View<T, P...>::array_layout,
                   Kokkos::LayoutRight>::value ||
      std::is_same<typename Kokkos::View<T, P...>::array_layout,
                   Kokkos::LayoutStride>::value ||
      is_layouttiled<typename Kokkos::View<T, P...>::array_layout>::value)>
impl_realloc(Kokkos::View<T, P...>& v,
             const typename Kokkos::View<T, P...>::array_layout& layout,
             const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop) {
  using view_type        = Kokkos::View<T, P...>;
  using alloc_prop_input = Impl::ViewCtorProp<ViewCtorArgs...>;

  static_assert(Kokkos::ViewTraits<T, P...>::is_managed,
                "Can only realloc managed views");
  static_assert(!alloc_prop_input::has_label,
                "The view constructor arguments passed to Kokkos::realloc must "
                "not include a label!");
  static_assert(!alloc_prop_input::has_pointer,
                "The view constructor arguments passed to Kokkos::realloc must "
                "not include a pointer!");
  static_assert(!alloc_prop_input::has_memory_space,
                "The view constructor arguments passed to Kokkos::realloc must "
                "not include a memory space instance!");

  auto arg_prop_copy = Impl::with_properties_if_unset(arg_prop, v.label());

  v = view_type();  // Deallocate first, if the only view to allocation
  v = view_type(arg_prop_copy, layout);
}

template <class T, class... P, class... ViewCtorArgs>
inline void realloc(
    const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop,
    Kokkos::View<T, P...>& v,
    const typename Kokkos::View<T, P...>::array_layout& layout) {
  impl_realloc(v, layout, arg_prop);
}

template <class I, class T, class... P>
inline std::enable_if_t<Impl::is_view_ctor_property<I>::value> realloc(
    const I& arg_prop, Kokkos::View<T, P...>& v,
    const typename Kokkos::View<T, P...>::array_layout& layout) {
  impl_realloc(v, layout, Kokkos::view_alloc(arg_prop));
}

template <class T, class... P>
inline void realloc(
    Kokkos::View<T, P...>& v,
    const typename Kokkos::View<T, P...>::array_layout& layout) {
  impl_realloc(v, layout, Impl::ViewCtorProp<>{});
}

} /* namespace Kokkos */

#endif
