TRIBITS_PACKAGE_DEFINE_DEPENDENCIES(
  LIB_OPTIONAL_TPLS Pthread CUDA HWLOC DLlib ROCTHRUST
  TEST_OPTIONAL_TPLS CUSPARSE
  )

TRIBITS_TPL_TENTATIVELY_ENABLE(DLlib)
