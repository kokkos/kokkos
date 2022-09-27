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
#ifndef KOKKOS_CREATEMIRROR_HPP_
#define KOKKOS_CREATEMIRROR_HPP_

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

// Deduce Mirror Types
template <class Space, class T, class... P>
struct MirrorViewType {
  // The incoming view_type
  using src_view_type = typename Kokkos::View<T, P...>;
  // The memory space for the mirror view
  using memory_space = typename Space::memory_space;
  // Check whether it is the same memory space
  enum {
    is_same_memspace =
        std::is_same<memory_space, typename src_view_type::memory_space>::value
  };
  // The array_layout
  using array_layout = typename src_view_type::array_layout;
  // The data type (we probably want it non-const since otherwise we can't even
  // deep_copy to it.
  using data_type = typename src_view_type::non_const_data_type;
  // The destination view type if it is not the same memory space
  using dest_view_type = Kokkos::View<data_type, array_layout, Space>;
  // If it is the same memory_space return the existsing view_type
  // This will also keep the unmanaged trait if necessary
  using view_type =
      std::conditional_t<is_same_memspace, src_view_type, dest_view_type>;
};

template <class Space, class T, class... P>
struct MirrorType {
  // The incoming view_type
  using src_view_type = typename Kokkos::View<T, P...>;
  // The memory space for the mirror view
  using memory_space = typename Space::memory_space;
  // Check whether it is the same memory space
  enum {
    is_same_memspace =
        std::is_same<memory_space, typename src_view_type::memory_space>::value
  };
  // The array_layout
  using array_layout = typename src_view_type::array_layout;
  // The data type (we probably want it non-const since otherwise we can't even
  // deep_copy to it.
  using data_type = typename src_view_type::non_const_data_type;
  // The destination view type if it is not the same memory space
  using view_type = Kokkos::View<data_type, array_layout, Space>;
};

template <class T, class... P, class... ViewCtorArgs>
inline std::enable_if_t<
    !std::is_same<typename Kokkos::ViewTraits<T, P...>::array_layout,
                  Kokkos::LayoutStride>::value &&
        !Impl::ViewCtorProp<ViewCtorArgs...>::has_memory_space,
    typename Kokkos::View<T, P...>::HostMirror>
create_mirror(const Kokkos::View<T, P...>& src,
              const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop) {
  using src_type         = View<T, P...>;
  using dst_type         = typename src_type::HostMirror;
  using alloc_prop_input = Impl::ViewCtorProp<ViewCtorArgs...>;

  static_assert(
      !alloc_prop_input::has_label,
      "The view constructor arguments passed to Kokkos::create_mirror "
      "must not include a label!");
  static_assert(
      !alloc_prop_input::has_pointer,
      "The view constructor arguments passed to Kokkos::create_mirror must "
      "not include a pointer!");
  static_assert(
      !alloc_prop_input::allow_padding,
      "The view constructor arguments passed to Kokkos::create_mirror must "
      "not explicitly allow padding!");

  auto prop_copy = Impl::with_properties_if_unset(
      arg_prop, std::string(src.label()).append("_mirror"));

  return dst_type(
      prop_copy,
      src.rank_dynamic > 0 ? src.extent(0) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      src.rank_dynamic > 1 ? src.extent(1) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      src.rank_dynamic > 2 ? src.extent(2) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      src.rank_dynamic > 3 ? src.extent(3) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      src.rank_dynamic > 4 ? src.extent(4) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      src.rank_dynamic > 5 ? src.extent(5) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      src.rank_dynamic > 6 ? src.extent(6) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      src.rank_dynamic > 7 ? src.extent(7) : KOKKOS_IMPL_CTOR_DEFAULT_ARG);
}

template <class T, class... P, class... ViewCtorArgs>
inline std::enable_if_t<
    std::is_same<typename Kokkos::ViewTraits<T, P...>::array_layout,
                 Kokkos::LayoutStride>::value &&
        !Impl::ViewCtorProp<ViewCtorArgs...>::has_memory_space,
    typename Kokkos::View<T, P...>::HostMirror>
create_mirror(const Kokkos::View<T, P...>& src,
              const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop) {
  using src_type         = View<T, P...>;
  using dst_type         = typename src_type::HostMirror;
  using alloc_prop_input = Impl::ViewCtorProp<ViewCtorArgs...>;

  static_assert(
      !alloc_prop_input::has_label,
      "The view constructor arguments passed to Kokkos::create_mirror "
      "must not include a label!");
  static_assert(
      !alloc_prop_input::has_pointer,
      "The view constructor arguments passed to Kokkos::create_mirror must "
      "not include a pointer!");
  static_assert(
      !alloc_prop_input::allow_padding,
      "The view constructor arguments passed to Kokkos::create_mirror must "
      "not explicitly allow padding!");

  Kokkos::LayoutStride layout;

  layout.dimension[0] = src.extent(0);
  layout.dimension[1] = src.extent(1);
  layout.dimension[2] = src.extent(2);
  layout.dimension[3] = src.extent(3);
  layout.dimension[4] = src.extent(4);
  layout.dimension[5] = src.extent(5);
  layout.dimension[6] = src.extent(6);
  layout.dimension[7] = src.extent(7);

  layout.stride[0] = src.stride_0();
  layout.stride[1] = src.stride_1();
  layout.stride[2] = src.stride_2();
  layout.stride[3] = src.stride_3();
  layout.stride[4] = src.stride_4();
  layout.stride[5] = src.stride_5();
  layout.stride[6] = src.stride_6();
  layout.stride[7] = src.stride_7();

  auto prop_copy = Impl::with_properties_if_unset(
      arg_prop, std::string(src.label()).append("_mirror"));

  return dst_type(prop_copy, layout);
}

// Create a mirror in a new space (specialization for different space)
template <class T, class... P, class... ViewCtorArgs,
          class Enable = std::enable_if_t<
              Impl::ViewCtorProp<ViewCtorArgs...>::has_memory_space>>
auto create_mirror(const Kokkos::View<T, P...>& src,
                   const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop) {
  using alloc_prop_input = Impl::ViewCtorProp<ViewCtorArgs...>;

  static_assert(
      !alloc_prop_input::has_label,
      "The view constructor arguments passed to Kokkos::create_mirror "
      "must not include a label!");
  static_assert(
      !alloc_prop_input::has_pointer,
      "The view constructor arguments passed to Kokkos::create_mirror must "
      "not include a pointer!");
  static_assert(
      !alloc_prop_input::allow_padding,
      "The view constructor arguments passed to Kokkos::create_mirror must "
      "not explicitly allow padding!");

  auto prop_copy = Impl::with_properties_if_unset(
      arg_prop, std::string(src.label()).append("_mirror"));
  using alloc_prop = decltype(prop_copy);

  return typename Impl::MirrorType<typename alloc_prop::memory_space, T,
                                   P...>::view_type(prop_copy, src.layout());
}
}  // namespace Impl

template <class T, class... P>
std::enable_if_t<std::is_void<typename ViewTraits<T, P...>::specialize>::value,
                 typename Kokkos::View<T, P...>::HostMirror>
create_mirror(Kokkos::View<T, P...> const& v) {
  return Impl::create_mirror(v, Impl::ViewCtorProp<>{});
}

template <class T, class... P>
std::enable_if_t<std::is_void<typename ViewTraits<T, P...>::specialize>::value,
                 typename Kokkos::View<T, P...>::HostMirror>
create_mirror(Kokkos::Impl::WithoutInitializing_t wi,
              Kokkos::View<T, P...> const& v) {
  return Impl::create_mirror(v, view_alloc(wi));
}

template <class Space, class T, class... P,
          typename Enable = std::enable_if_t<Kokkos::is_space<Space>::value>>
std::enable_if_t<std::is_void<typename ViewTraits<T, P...>::specialize>::value,
                 typename Impl::MirrorType<Space, T, P...>::view_type>
create_mirror(Space const&, Kokkos::View<T, P...> const& v) {
  return Impl::create_mirror(v, view_alloc(typename Space::memory_space{}));
}

template <class T, class... P, class... ViewCtorArgs,
          typename Enable = std::enable_if_t<
              std::is_void<typename ViewTraits<T, P...>::specialize>::value &&
              Impl::ViewCtorProp<ViewCtorArgs...>::has_memory_space>>
auto create_mirror(Impl::ViewCtorProp<ViewCtorArgs...> const& arg_prop,
                   Kokkos::View<T, P...> const& v) {
  return Impl::create_mirror(v, arg_prop);
}

template <class T, class... P, class... ViewCtorArgs>
std::enable_if_t<
    std::is_void<typename ViewTraits<T, P...>::specialize>::value &&
        !Impl::ViewCtorProp<ViewCtorArgs...>::has_memory_space,
    typename Kokkos::View<T, P...>::HostMirror>
create_mirror(Impl::ViewCtorProp<ViewCtorArgs...> const& arg_prop,
              Kokkos::View<T, P...> const& v) {
  return Impl::create_mirror(v, arg_prop);
}

template <class Space, class T, class... P,
          typename Enable = std::enable_if_t<Kokkos::is_space<Space>::value>>
std::enable_if_t<std::is_void<typename ViewTraits<T, P...>::specialize>::value,
                 typename Impl::MirrorType<Space, T, P...>::view_type>
create_mirror(Kokkos::Impl::WithoutInitializing_t wi, Space const&,
              Kokkos::View<T, P...> const& v) {
  return Impl::create_mirror(v, view_alloc(typename Space::memory_space{}, wi));
}

namespace Impl {

template <class T, class... P, class... ViewCtorArgs>
inline std::enable_if_t<
    !Impl::ViewCtorProp<ViewCtorArgs...>::has_memory_space &&
        (std::is_same<
             typename Kokkos::View<T, P...>::memory_space,
             typename Kokkos::View<T, P...>::HostMirror::memory_space>::value &&
         std::is_same<
             typename Kokkos::View<T, P...>::data_type,
             typename Kokkos::View<T, P...>::HostMirror::data_type>::value),
    typename Kokkos::View<T, P...>::HostMirror>
create_mirror_view(const Kokkos::View<T, P...>& src,
                   const Impl::ViewCtorProp<ViewCtorArgs...>&) {
  return src;
}

template <class T, class... P, class... ViewCtorArgs>
inline std::enable_if_t<
    !Impl::ViewCtorProp<ViewCtorArgs...>::has_memory_space &&
        !(std::is_same<typename Kokkos::View<T, P...>::memory_space,
                       typename Kokkos::View<
                           T, P...>::HostMirror::memory_space>::value &&
          std::is_same<
              typename Kokkos::View<T, P...>::data_type,
              typename Kokkos::View<T, P...>::HostMirror::data_type>::value),
    typename Kokkos::View<T, P...>::HostMirror>
create_mirror_view(const Kokkos::View<T, P...>& src,
                   const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop) {
  return Kokkos::Impl::create_mirror(src, arg_prop);
}

// Create a mirror view in a new space (specialization for same space)
template <class T, class... P, class... ViewCtorArgs,
          class = std::enable_if_t<
              Impl::ViewCtorProp<ViewCtorArgs...>::has_memory_space>>
std::enable_if_t<Impl::MirrorViewType<
                     typename Impl::ViewCtorProp<ViewCtorArgs...>::memory_space,
                     T, P...>::is_same_memspace,
                 typename Impl::MirrorViewType<
                     typename Impl::ViewCtorProp<ViewCtorArgs...>::memory_space,
                     T, P...>::view_type>
create_mirror_view(const Kokkos::View<T, P...>& src,
                   const Impl::ViewCtorProp<ViewCtorArgs...>&) {
  return src;
}

// Create a mirror view in a new space (specialization for different space)
template <class T, class... P, class... ViewCtorArgs,
          class = std::enable_if_t<
              Impl::ViewCtorProp<ViewCtorArgs...>::has_memory_space>>
std::enable_if_t<!Impl::MirrorViewType<
                     typename Impl::ViewCtorProp<ViewCtorArgs...>::memory_space,
                     T, P...>::is_same_memspace,
                 typename Impl::MirrorViewType<
                     typename Impl::ViewCtorProp<ViewCtorArgs...>::memory_space,
                     T, P...>::view_type>
create_mirror_view(const Kokkos::View<T, P...>& src,
                   const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop) {
  return Kokkos::Impl::create_mirror(src, arg_prop);
}
}  // namespace Impl

template <class T, class... P>
std::enable_if_t<
    std::is_same<
        typename Kokkos::View<T, P...>::memory_space,
        typename Kokkos::View<T, P...>::HostMirror::memory_space>::value &&
        std::is_same<
            typename Kokkos::View<T, P...>::data_type,
            typename Kokkos::View<T, P...>::HostMirror::data_type>::value,
    typename Kokkos::View<T, P...>::HostMirror>
create_mirror_view(const Kokkos::View<T, P...>& src) {
  return src;
}

template <class T, class... P>
std::enable_if_t<
    !(std::is_same<
          typename Kokkos::View<T, P...>::memory_space,
          typename Kokkos::View<T, P...>::HostMirror::memory_space>::value &&
      std::is_same<
          typename Kokkos::View<T, P...>::data_type,
          typename Kokkos::View<T, P...>::HostMirror::data_type>::value),
    typename Kokkos::View<T, P...>::HostMirror>
create_mirror_view(const Kokkos::View<T, P...>& src) {
  return Kokkos::create_mirror(src);
}

template <class T, class... P>
typename Kokkos::View<T, P...>::HostMirror create_mirror_view(
    Kokkos::Impl::WithoutInitializing_t wi, Kokkos::View<T, P...> const& v) {
  return Impl::create_mirror_view(v, view_alloc(wi));
}

// FIXME_C++17 Improve SFINAE here.
template <class Space, class T, class... P,
          class Enable = std::enable_if_t<Kokkos::is_space<Space>::value>>
typename Impl::MirrorViewType<Space, T, P...>::view_type create_mirror_view(
    const Space&, const Kokkos::View<T, P...>& src,
    std::enable_if_t<Impl::MirrorViewType<Space, T, P...>::is_same_memspace>* =
        nullptr) {
  return src;
}

// FIXME_C++17 Improve SFINAE here.
template <class Space, class T, class... P,
          class Enable = std::enable_if_t<Kokkos::is_space<Space>::value>>
typename Impl::MirrorViewType<Space, T, P...>::view_type create_mirror_view(
    const Space& space, const Kokkos::View<T, P...>& src,
    std::enable_if_t<!Impl::MirrorViewType<Space, T, P...>::is_same_memspace>* =
        nullptr) {
  return Kokkos::create_mirror(space, src);
}

template <class Space, class T, class... P,
          typename Enable = std::enable_if_t<Kokkos::is_space<Space>::value>>
typename Impl::MirrorViewType<Space, T, P...>::view_type create_mirror_view(
    Kokkos::Impl::WithoutInitializing_t wi, Space const&,
    Kokkos::View<T, P...> const& v) {
  return Impl::create_mirror_view(
      v, view_alloc(typename Space::memory_space{}, wi));
}

template <class T, class... P, class... ViewCtorArgs>
auto create_mirror_view(const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop,
                        const Kokkos::View<T, P...>& v) {
  return Impl::create_mirror_view(v, arg_prop);
}

template <class... ViewCtorArgs, class T, class... P>
auto create_mirror_view_and_copy(
    const Impl::ViewCtorProp<ViewCtorArgs...>&,
    const Kokkos::View<T, P...>& src,
    std::enable_if_t<
        std::is_void<typename ViewTraits<T, P...>::specialize>::value &&
        Impl::MirrorViewType<
            typename Impl::ViewCtorProp<ViewCtorArgs...>::memory_space, T,
            P...>::is_same_memspace>* = nullptr) {
  using alloc_prop_input = Impl::ViewCtorProp<ViewCtorArgs...>;
  static_assert(
      alloc_prop_input::has_memory_space,
      "The view constructor arguments passed to "
      "Kokkos::create_mirror_view_and_copy must include a memory space!");
  static_assert(!alloc_prop_input::has_pointer,
                "The view constructor arguments passed to "
                "Kokkos::create_mirror_view_and_copy must "
                "not include a pointer!");
  static_assert(!alloc_prop_input::allow_padding,
                "The view constructor arguments passed to "
                "Kokkos::create_mirror_view_and_copy must "
                "not explicitly allow padding!");

  // same behavior as deep_copy(src, src)
  if (!alloc_prop_input::has_execution_space)
    fence(
        "Kokkos::create_mirror_view_and_copy: fence before returning src view");
  return src;
}

template <class... ViewCtorArgs, class T, class... P>
auto create_mirror_view_and_copy(
    const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop,
    const Kokkos::View<T, P...>& src,
    std::enable_if_t<
        std::is_void<typename ViewTraits<T, P...>::specialize>::value &&
        !Impl::MirrorViewType<
            typename Impl::ViewCtorProp<ViewCtorArgs...>::memory_space, T,
            P...>::is_same_memspace>* = nullptr) {
  using alloc_prop_input = Impl::ViewCtorProp<ViewCtorArgs...>;
  static_assert(
      alloc_prop_input::has_memory_space,
      "The view constructor arguments passed to "
      "Kokkos::create_mirror_view_and_copy must include a memory space!");
  static_assert(!alloc_prop_input::has_pointer,
                "The view constructor arguments passed to "
                "Kokkos::create_mirror_view_and_copy must "
                "not include a pointer!");
  static_assert(!alloc_prop_input::allow_padding,
                "The view constructor arguments passed to "
                "Kokkos::create_mirror_view_and_copy must "
                "not explicitly allow padding!");
  using Space  = typename alloc_prop_input::memory_space;
  using Mirror = typename Impl::MirrorViewType<Space, T, P...>::view_type;

  auto arg_prop_copy = Impl::with_properties_if_unset(
      arg_prop, std::string{}, WithoutInitializing,
      typename Space::execution_space{});

  std::string& label = Impl::get_property<Impl::LabelTag>(arg_prop_copy);
  if (label.empty()) label = src.label();
  auto mirror = typename Mirror::non_const_type{arg_prop_copy, src.layout()};
  if constexpr (alloc_prop_input::has_execution_space) {
    deep_copy(Impl::get_property<Impl::ExecutionSpaceTag>(arg_prop_copy),
              mirror, src);
  } else
    deep_copy(mirror, src);
  return mirror;
}

// Previously when using auto here, the intel compiler 19.3 would
// sometimes not create a symbol, guessing that it somehow is a combination
// of auto and just forwarding arguments (see issue #5196)
template <class Space, class T, class... P,
          typename Enable = std::enable_if_t<Kokkos::is_space<Space>::value>>
typename Impl::MirrorViewType<Space, T, P...>::view_type
create_mirror_view_and_copy(
    const Space&, const Kokkos::View<T, P...>& src,
    std::string const& name = "",
    std::enable_if_t<
        std::is_void<typename ViewTraits<T, P...>::specialize>::value>* =
        nullptr) {
  return create_mirror_view_and_copy(
      Kokkos::view_alloc(typename Space::memory_space{}, name), src);
}

} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif
