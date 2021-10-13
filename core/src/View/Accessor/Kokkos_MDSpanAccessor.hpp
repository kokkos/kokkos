#include<View/Accessor/Kokkos_Accessor_Atomic.hpp>


namespace Kokkos {
namespace Impl {
template<class MemTraits, class ElementType>
struct KokkosAccessorFromMemTraits {
  using accessor_type =
    std::conditional_t<
      MemTraits::is_atomic, AtomicAccessor<ElementType>,
    std::experimental::default_accessor<ElementType>>;
};
} // namespace Impl
} // namespace Kokkos
