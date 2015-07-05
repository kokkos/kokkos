SET(SUBPACKAGES_DIRS_CLASSIFICATIONS_OPTREQS
  #SubPackageName       Directory         Class    Req/Opt
  #
  # New Kokkos subpackages:
  Core                  core              PS       REQUIRED
  Containers            containers        PS       OPTIONAL
  Algorithms            algorithms        PS       OPTIONAL
  Example               example           EX       OPTIONAL
  )

SET(LIB_REQUIRED_DEP_PACKAGES )
SET(LIB_OPTIONAL_DEP_PACKAGES )
SET(TEST_REQUIRED_DEP_PACKAGES )
SET(TEST_OPTIONAL_DEP_PACKAGES )
SET(LIB_REQUIRED_DEP_TPLS )
SET(LIB_OPTIONAL_DEP_TPLS HWLOC) 
SET(TEST_REQUIRED_DEP_TPLS )
SET(TEST_OPTIONAL_DEP_TPLS )
