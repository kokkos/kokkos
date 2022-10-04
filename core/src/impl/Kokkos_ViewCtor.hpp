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

#ifndef KOKKOS_EXPERIMENTAL_IMPL_VIEW_CTOR_PROP_HPP
#define KOKKOS_EXPERIMENTAL_IMPL_VIEW_CTOR_PROP_HPP

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

struct WithoutInitializing_t {};
struct AllowPadding_t {};

template <typename>
struct is_view_ctor_property : public std::false_type {};

template <>
struct is_view_ctor_property<WithoutInitializing_t> : public std::true_type {};

template <>
struct is_view_ctor_property<AllowPadding_t> : public std::true_type {};

//----------------------------------------------------------------------------
/**\brief Whether a type can be used for a view label */

template <typename>
struct is_view_label : public std::false_type {};

template <>
struct is_view_label<std::string> : public std::true_type {};

template <unsigned N>
struct is_view_label<char[N]> : public std::true_type {};

template <unsigned N>
struct is_view_label<const char[N]> : public std::true_type {};

//----------------------------------------------------------------------------

template <typename... P>
struct ViewCtorProp;

// Forward declare
template <typename Specialize, typename T>
struct CommonViewAllocProp;

/* Dummy to allow for empty ViewCtorProp object
 */
template <>
struct ViewCtorProp<void> {};

/* Common value_type stored as ViewCtorProp
 */
template <typename Specialize, typename T>
struct ViewCtorProp<void, CommonViewAllocProp<Specialize, T>> {
  ViewCtorProp()                     = default;
  ViewCtorProp(const ViewCtorProp &) = default;
  ViewCtorProp &operator=(const ViewCtorProp &) = default;

  using type = CommonViewAllocProp<Specialize, T>;

  KOKKOS_FUNCTION
  ViewCtorProp(const type &arg) : value(arg) {}
  KOKKOS_FUNCTION
  ViewCtorProp(type &&arg) : value(arg) {}

  type value;
};

/* Property flags have constexpr value */
template <typename P>
struct ViewCtorProp<
    std::enable_if_t<std::is_same<P, AllowPadding_t>::value ||
                     std::is_same<P, WithoutInitializing_t>::value>,
    P> {
  ViewCtorProp()                     = default;
  ViewCtorProp(const ViewCtorProp &) = default;
  ViewCtorProp &operator=(const ViewCtorProp &) = default;

  using type = P;

  ViewCtorProp(const type &) {}

  type value = type();
};

/* Map input label type to std::string */
template <typename Label>
struct ViewCtorProp<std::enable_if_t<is_view_label<Label>::value>, Label> {
  ViewCtorProp()                     = default;
  ViewCtorProp(const ViewCtorProp &) = default;
  ViewCtorProp &operator=(const ViewCtorProp &) = default;

  using type = std::string;

  ViewCtorProp(const type &arg) : value(arg) {}
  ViewCtorProp(type &&arg) : value(arg) {}

  type value;
};

template <typename Space>
struct ViewCtorProp<std::enable_if_t<Kokkos::is_memory_space<Space>::value ||
                                     Kokkos::is_execution_space<Space>::value>,
                    Space> {
  ViewCtorProp()                     = default;
  ViewCtorProp(const ViewCtorProp &) = default;
  ViewCtorProp &operator=(const ViewCtorProp &) = default;

  using type = Space;

  ViewCtorProp(const type &arg) : value(arg) {}

  type value;
};

template <typename T>
struct ViewCtorProp<void, T *> {
  ViewCtorProp()                     = default;
  ViewCtorProp(const ViewCtorProp &) = default;
  ViewCtorProp &operator=(const ViewCtorProp &) = default;

  using type = T *;

  KOKKOS_FUNCTION
  ViewCtorProp(const type arg) : value(arg) {}

  type value;
};

// For some reason I don't understand I needed this specialization explicitly
// for NVCC/MSVC
template <typename T>
struct ViewCtorProp<T *> : public ViewCtorProp<void, T *> {
  static constexpr bool has_memory_space    = false;
  static constexpr bool has_execution_space = false;
  static constexpr bool has_pointer         = true;
  static constexpr bool has_label           = false;
  static constexpr bool allow_padding       = false;
  static constexpr bool initialize          = true;

  using memory_space    = void;
  using execution_space = void;
  using pointer_type    = T *;

  KOKKOS_FUNCTION ViewCtorProp(const pointer_type arg)
      : ViewCtorProp<void, pointer_type>(arg) {}
};

// If we use `ViewCtorProp<Args...>` and `ViewCtorProp<void, Args>...` directly
// in the parameter lists and base class initializers, respectively, as far as
// we can tell MSVC 16.5.5+CUDA 10.2 thinks that `ViewCtorProp` refers to the
// current instantiation, not the template itself, and gets all kinds of
// confused. To work around this, we just use a couple of alias templates that
// amount to the same thing.
template <typename... Args>
using view_ctor_prop_args = ViewCtorProp<Args...>;

template <typename Arg>
using view_ctor_prop_base = ViewCtorProp<void, Arg>;

template <typename... P>
struct ViewCtorProp : public ViewCtorProp<void, P>... {
 private:
  using var_memory_space =
      Kokkos::Impl::has_condition<void, Kokkos::is_memory_space, P...>;

  using var_execution_space =
      Kokkos::Impl::has_condition<void, Kokkos::is_execution_space, P...>;

  struct VOIDDUMMY {};

  using var_pointer =
      Kokkos::Impl::has_condition<VOIDDUMMY, std::is_pointer, P...>;

 public:
  /* Flags for the common properties */
  static constexpr bool has_memory_space    = var_memory_space::value;
  static constexpr bool has_execution_space = var_execution_space::value;
  static constexpr bool has_pointer         = var_pointer::value;
  static constexpr bool has_label =
      Kokkos::Impl::has_type<std::string, P...>::value;
  static constexpr bool allow_padding =
      Kokkos::Impl::has_type<AllowPadding_t, P...>::value;
  static constexpr bool initialize =
      !Kokkos::Impl::has_type<WithoutInitializing_t, P...>::value;

  using memory_space    = typename var_memory_space::type;
  using execution_space = typename var_execution_space::type;
  using pointer_type    = typename var_pointer::type;

  /*  Copy from a matching argument list.
   *  Requires  std::is_same< P , ViewCtorProp< void , Args >::value ...
   */
  template <typename... Args>
  inline ViewCtorProp(Args const &... args) : ViewCtorProp<void, P>(args)... {}

  template <typename... Args>
  KOKKOS_FUNCTION ViewCtorProp(pointer_type arg0, Args const &... args)
      : ViewCtorProp<void, pointer_type>(arg0),
        ViewCtorProp<void, typename ViewCtorProp<void, Args>::type>(args)... {}

  /* Copy from a matching property subset */
  KOKKOS_FUNCTION ViewCtorProp(pointer_type arg0)
      : ViewCtorProp<void, pointer_type>(arg0) {}

  // If we use `ViewCtorProp<Args...>` and `ViewCtorProp<void, Args>...` here
  // directly, MSVC 16.5.5+CUDA 10.2 appears to think that `ViewCtorProp` refers
  // to the current instantiation, not the template itself, and gets all kinds
  // of confused. To work around this, we just use a couple of alias templates
  // that amount to the same thing.
  template <typename... Args>
  ViewCtorProp(view_ctor_prop_args<Args...> const &arg)
      : view_ctor_prop_base<Args>(
            static_cast<view_ctor_prop_base<Args> const &>(arg))... {
    // Suppress an unused argument warning that (at least at one point) would
    // show up if sizeof...(Args) == 0
    (void)arg;
  }
};

template <typename... P>
auto with_properties_if_unset(const ViewCtorProp<P...> &view_ctor_prop) {
  return view_ctor_prop;
}

// Suppress faulty warning from NVCC see:
// https://github.com/kokkos/kokkos/issues/5426
#if defined(KOKKOS_COMPILER_NVCC) && (KOKKOS_COMPILER_NVCC < 1150)
// pragma pop is getting a warning from the underlying GCC
// for unknown pragma if -pedantic is used
#ifdef __CUDA_ARCH__
#pragma push
#pragma diag_suppress implicit_return_from_non_void_function
#endif
#endif
#if defined(KOKKOS_COMPILER_INTEL) && (KOKKOS_COMPILER_INTEL <= 2100)
#define KOKKOS_IMPL_INTEL_BOGUS_MISSING_RETURN_STATEMENT_AT_END_OF_NON_VOID_FUNCTION
#pragma warning(push)
#pragma warning(disable : 1011)
#endif
template <typename... P, typename Property, typename... Properties>
auto with_properties_if_unset(const ViewCtorProp<P...> &view_ctor_prop,
                              const Property &property,
                              const Properties &... properties) {
  if constexpr ((is_execution_space<Property>::value &&
                 !ViewCtorProp<P...>::has_execution_space) ||
                (is_memory_space<Property>::value &&
                 !ViewCtorProp<P...>::has_memory_space) ||
                (is_view_label<Property>::value &&
                 !ViewCtorProp<P...>::has_label) ||
                (std::is_same_v<Property, WithoutInitializing_t> &&
                 ViewCtorProp<P...>::initialize) ||
                (std::is_same_v<Property, AllowPadding_t> &&
                 !ViewCtorProp<P...>::allow_padding)) {
    using NewViewCtorProp = ViewCtorProp<P..., Property>;
    NewViewCtorProp new_view_ctor_prop(view_ctor_prop);
    static_cast<ViewCtorProp<void, Property> &>(new_view_ctor_prop).value =
        property;
    return with_properties_if_unset(new_view_ctor_prop, properties...);
  } else
    return with_properties_if_unset(view_ctor_prop, properties...);
}

struct ExecutionSpaceTag {};
struct MemorySpaceTag {};
struct LabelTag {};
struct PointerTag {};

template <typename Tag, typename... P>
KOKKOS_FUNCTION const auto &get_property(
    const ViewCtorProp<P...> &view_ctor_prop) {
  if constexpr (std::is_same_v<Tag, ExecutionSpaceTag>) {
    static_assert(ViewCtorProp<P...>::has_execution_space);
    using execution_space_type = typename ViewCtorProp<P...>::execution_space;
    return static_cast<const ViewCtorProp<void, execution_space_type> &>(
               view_ctor_prop)
        .value;
  } else if constexpr (std::is_same_v<Tag, MemorySpaceTag>) {
    static_assert(ViewCtorProp<P...>::has_memory_space);
    using memory_space_type = typename ViewCtorProp<P...>::memory_space;
    return static_cast<const ViewCtorProp<void, memory_space_type> &>(
               view_ctor_prop)
        .value;
  } else if constexpr (std::is_same_v<Tag, LabelTag>) {
    static_assert(ViewCtorProp<P...>::has_label);
    return static_cast<const ViewCtorProp<void, std::string> &>(view_ctor_prop)
        .value;
  } else if constexpr (std::is_same_v<Tag, PointerTag>) {
    static_assert(ViewCtorProp<P...>::has_pointer);
    using pointer_type = typename ViewCtorProp<P...>::pointer_type;
    return static_cast<const ViewCtorProp<void, pointer_type> &>(view_ctor_prop)
        .value;
  } else {
    static_assert(std::is_same_v<Tag, void>, "Invalid property tag!");
    return view_ctor_prop;
  }
}
#if defined(KOKKOS_COMPILER_NVCC) && (KOKKOS_COMPILER_NVCC < 1150)
// pragma pop is getting a warning from the underlying GCC
// for unknown pragma if -pedantic is used
#ifdef __CUDA_ARCH__
#pragma pop
#endif
#endif
#ifdef KOKKOS_IMPL_INTEL_BOGUS_MISSING_RETURN_STATEMENT_AT_END_OF_NON_VOID_FUNCTION
#pragma warning(pop)
#undef KOKKOS_IMPL_INTEL_BOGUS_MISSING_RETURN_STATEMENT_AT_END_OF_NON_VOID_FUNCTION
#endif

template <typename Tag, typename... P>
KOKKOS_FUNCTION auto &get_property(ViewCtorProp<P...> &view_ctor_prop) {
  // Avoid code duplication by deferring to the const-qualified overload and
  // casting the const away from the return type
  const auto &tmp = get_property<Tag>(
      static_cast<const ViewCtorProp<P...> &>(view_ctor_prop));
  return const_cast<std::decay_t<decltype(tmp)> &>(tmp);
}

} /* namespace Impl */
} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

namespace Impl {
struct ViewAllocateWithoutInitializingBackwardCompat {};

template <>
struct ViewCtorProp<void, ViewAllocateWithoutInitializingBackwardCompat> {};

// NOTE This specialization is meant to be used as the
// ViewAllocateWithoutInitializing alias below. All it does is add a
// constructor that takes the label as single argument.
template <>
struct ViewCtorProp<WithoutInitializing_t, std::string,
                    ViewAllocateWithoutInitializingBackwardCompat>
    : ViewCtorProp<WithoutInitializing_t, std::string>,
      ViewCtorProp<void, ViewAllocateWithoutInitializingBackwardCompat> {
  ViewCtorProp(std::string label)
      : ViewCtorProp<WithoutInitializing_t, std::string>(
            WithoutInitializing_t(), std::move(label)) {}
};
} /* namespace Impl */

using ViewAllocateWithoutInitializing =
    Impl::ViewCtorProp<Impl::WithoutInitializing_t, std::string,
                       Impl::ViewAllocateWithoutInitializingBackwardCompat>;

} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif
