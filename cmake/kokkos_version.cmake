set(Kokkos_VERSION_MAJOR 5)
set(Kokkos_VERSION_MINOR 2)
set(Kokkos_VERSION_PATCH 99)
set(Kokkos_VERSION "${Kokkos_VERSION_MAJOR}.${Kokkos_VERSION_MINOR}.${Kokkos_VERSION_PATCH}")
math(EXPR KOKKOS_VERSION "${Kokkos_VERSION_MAJOR} * 10000 + ${Kokkos_VERSION_MINOR} * 100 + ${Kokkos_VERSION_PATCH}")
# mathematical expressions below are not stricly necessary but they eliminate
# the rather aggravating leading 0 in the releases patch version number, and,
# in some way, are a sanity check for our arithmetic
math(EXPR KOKKOS_VERSION_MAJOR "${KOKKOS_VERSION} / 10000")
math(EXPR KOKKOS_VERSION_MINOR "${KOKKOS_VERSION} / 100 % 100")
math(EXPR KOKKOS_VERSION_PATCH "${KOKKOS_VERSION} % 100")
