#
# These variables must be set for every package or subpackage.
#
# Including Kokkos subpackage library interdependencies for this subpackage.
# Subpackage names specified as {Package}{Subpackage}
# where {Subpackage} name is defined in kokkos/cmake/Dependencies.cmake
#
SET(LIB_REQUIRED_DEP_PACKAGES KokkosCore KokkosContainers KokkosAlgorithms)
SET(LIB_OPTIONAL_DEP_PACKAGES)
SET(TEST_REQUIRED_DEP_PACKAGES)
SET(TEST_OPTIONAL_DEP_PACKAGES)
SET(LIB_REQUIRED_DEP_TPLS)
SET(LIB_OPTIONAL_DEP_TPLS)
SET(TEST_REQUIRED_DEP_TPLS )
SET(TEST_OPTIONAL_DEP_TPLS CUSPARSE MKL )
