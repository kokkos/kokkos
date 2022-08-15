#ifndef KOKKOS_OPENACC_MDRANGEPOLICY_HPP_
#define KOKKOS_OPENACC_MDRANGEPOLICY_HPP_

#include <KokkosExp_MDRangePolicy.hpp>

namespace Kokkos {

template <>
struct default_outer_direction<Kokkos::Experimental::OpenACC> {
  using type                     = Iterate;
  static constexpr Iterate value = Iterate::Left;
};

template <>
struct default_inner_direction<Kokkos::Experimental::OpenACC> {
  using type                     = Iterate;
  static constexpr Iterate value = Iterate::Left;
};

}  // Namespace Kokkos
#endif
