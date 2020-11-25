
#include <type_traits>
#include <tuple>
#include <utility>
#include <string>
#include <iostream>
#include <typeinfo>

#include <cxxabi.h>

#ifndef KOKKOS_SPACE_PRIORITIES_HPP
#define KOKKOS_SPACE_PRIORITIES_HPP

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

// The following aliases define the default execution space and the
// default host execution space using the above templates to determine
// which execution space is the desired default.
// The index sequence is used to iterate through the configured spaces
// and the template logic will select the space that has the lowest
// SpacePriority value.  Each execution space has a template specialization
// for SpacePriority which returns a value corresponding to it's
// proper place in the priority list. KOKKOS_DEFAULT_EXECSPACE_PRIORITY_SET
// and KOKKOS_HOSTDEFAULT_EXECSPACE_PRIORITY_SET are additional configurations
// that can be used to tweak the desired order.
// The default host executiobn space works the same as the default execution
// space, but it relies on the device execution spaces having a template
// specialization for SpacePriority that returns a value of 999, thus taking
// it out of consideration.
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
