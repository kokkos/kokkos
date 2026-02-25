#include <type_traits>
#include <utility>
#include "../Kokkos_Array.hpp"
#include "../Kokkos_Layout.hpp"

namespace Kokkos::Impl {

template <class MDRP, class Functor, class Tag>
class FlatIterate;
}  // namespace Kokkos::Impl
