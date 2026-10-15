/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_LOAD_AND_STORE_OPENACC_HPP_
#define DESUL_ATOMICS_LOAD_AND_STORE_OPENACC_HPP_

#include <desul/atomics/Common.hpp>
#include <desul/atomics/OpenACC_Traits.hpp>
#include <desul/atomics/Compare_Exchange_OpenACC.hpp>
#include <desul/atomics/Lock_Free_Types_OpenACC.hpp>
#include <desul/atomics/Macros.hpp>

namespace desul {
namespace Impl {

//<editor-fold desc="device_atomic_{store,load}">

DESUL_IMPL_ATOMIC_LOAD_AND_STORE_WITH_CAS(DESUL_IMPL_DEVICE_FUNCTION, device)

#pragma acc routine seq
template <class T>
std::enable_if_t<is_openacc_arithmetic_type_v<T>, void> device_atomic_store(
    T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
#pragma acc atomic write
  *ptr = val;
}

#pragma acc routine seq
template <class T>
std::enable_if_t<is_openacc_arithmetic_type_v<T>, void> device_atomic_store(
    T* const ptr, const T val, MemoryOrderRelease, MemoryScopeDevice) {
  if (acc_on_device(acc_device_not_host)) {
    printf(
        "DESUL error in device_atomic_store(MemoryOrderRelease): Not supported atomic "
        "operation in the OpenACC backend\n");
  }
#pragma acc atomic write
  *ptr = val;
}

#pragma acc routine seq
template <class T>
std::enable_if_t<is_openacc_arithmetic_type_v<T>, T> device_atomic_load(
    const T* const ptr, MemoryOrderRelaxed, MemoryScopeDevice) {
  T retval;
#pragma acc atomic read
  retval = *ptr;
  return retval;
}

#pragma acc routine seq
template <class T>
std::enable_if_t<is_openacc_arithmetic_type_v<T>, T> device_atomic_load(
    const T* const ptr, MemoryOrderAcquire, MemoryScopeDevice) {
  if (acc_on_device(acc_device_not_host)) {
    printf(
        "DESUL error in device_atomic_load(MemoryOrderAcquire): Not supported atomic "
        "operation in the OpenACC backend\n");
  }
  T retval;
#pragma acc atomic read
  retval = *ptr;
  return retval;
}
//</editor-fold>

}  // namespace Impl
}  // namespace desul

#endif
