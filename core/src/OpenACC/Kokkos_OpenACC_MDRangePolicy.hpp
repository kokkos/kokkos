#ifndef KOKKOS_OPENACC_MDRANGE_POLICY_HPP_
#define KOKKOS_OPENACC_MDRANGE_POLICY_HPP_

#include <KokkosExp_MDRangePolicy.hpp>

template <>
struct Kokkos::default_outer_direction<Kokkos::Experimental::OpenACC> {
  using type                     = Iterate;
  static constexpr Iterate value = Iterate::Left;
};

template <>
struct Kokkos::default_inner_direction<Kokkos::Experimental::OpenACC> {
  using type                     = Iterate;
  static constexpr Iterate value = Iterate::Left;
};

#endif
