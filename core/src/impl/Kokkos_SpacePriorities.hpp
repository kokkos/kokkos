
#include <type_traits>
#include <tuple>
#include <utility>
#include <string>
#include <iostream>
#include <typeinfo>

#include <cxxabi.h>

#ifndef __SPACE_PRIORITIES_HPP
#define __SPACE_PRIORITIES_HPP

#if defined(__clang_analyzer__)
#define KOKKOS_IMPL_DEFAULT_EXEC_SPACE_ANNOTATION \
  [[clang::annotate("DefaultExecutionSpace")]]
#define KOKKOS_IMPL_DEFAULT_HOST_EXEC_SPACE_ANNOTATION \
  [[clang::annotate("DefaultHostExecutionSpace")]]
#else
#define KOKKOS_IMPL_DEFAULT_EXEC_SPACE_ANNOTATION
#define KOKKOS_IMPL_DEFAULT_HOST_EXEC_SPACE_ANNOTATION
#endif

namespace Kokkos {
namespace Impl {

struct DummySpace {};

template <typename ConfigSet>
struct SpacePriority<DummySpace, ConfigSet> : std::integral_constant<int, 999> {
};

template <typename... T>
struct TypeList;

template <class PrioritySet, template <typename, typename> class PrioT,
          typename Curr, typename Tail>
struct GetPriority;

template <class PrioritySet, template <typename, typename> class PrioT,
          typename Curr>
struct GetPriority<PrioritySet, PrioT, Curr, TypeList<>> {
  using type = Curr;
};

template <class PrioritySet, template <typename, typename> class PrioT,
          typename Curr, typename Tp, typename... Tail>
struct GetPriority<PrioritySet, PrioT, Curr, TypeList<Tp, Tail...>> {
  using type = std::conditional_t<
      (PrioT<Curr, PrioritySet>::value < PrioT<Tp, PrioritySet>::value),
      typename GetPriority<PrioritySet, PrioT, Curr, TypeList<Tail...>>::type,
      typename GetPriority<PrioritySet, PrioT, Tp, TypeList<Tail...>>::type>;
};

template <size_t... Idx>
struct GetTypesT {
  using type = TypeList<typename SpaceProperty<Idx>::type...>;
};

template <typename Tp>
struct GetTypes;

template <size_t... Idx>
struct GetTypes<std::index_sequence<Idx...>> : GetTypesT<Idx...> {};

}  // namespace Impl

using DefaultExecutionSpace KOKKOS_IMPL_DEFAULT_EXEC_SPACE_ANNOTATION =
    typename Impl::GetPriority<
        KOKKOS_DEFAULT_EXECSPACE_PRIORITY_SET, Impl::SpacePriority,
        Impl::DummySpace,
        typename Impl::GetTypes<std::make_index_sequence<
            Impl::TotalNumberOfExecutionSpaces>>::type>::type;

using DefaultHostExecutionSpace KOKKOS_IMPL_DEFAULT_HOST_EXEC_SPACE_ANNOTATION =
    typename Impl::GetPriority<
        KOKKOS_HOSTDEFAULT_EXECSPACE_PRIORITY_SET, Impl::SpacePriority,
        Impl::DummySpace,
        typename Impl::GetTypes<std::make_index_sequence<
            Impl::TotalNumberOfExecutionSpaces>>::type>::type;

}  // namespace Kokkos

#endif  // __SPACE_PRIORITIES_HPP
