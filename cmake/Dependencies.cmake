# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

tribits_package_define_dependencies(LIB_OPTIONAL_TPLS Pthread CUDA HWLOC DLlib)

tribits_tpl_tentatively_enable(DLlib)
