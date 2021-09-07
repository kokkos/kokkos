#ifndef KOKKOS_DESUL_ATOMICS_WRAPPER_HPP_
#define KOKKOS_DESUL_ATOMICS_WRAPPER_HPP_
#include <Kokkos_Macros.hpp>

#ifdef KOKKOS_ENABLE_IMPL_DESUL_ATOMICS
#include <desul/atomics.hpp>

#include <impl/Kokkos_Atomic_Memory_Order.hpp>
#include <impl/Kokkos_Volatile_Load.hpp>

// clang-format off
namespace Kokkos {

// FIXME: These functions don't have any use/test in unit tests ...
// ==========================================================
inline const char* atomic_query_version() { return "KOKKOS_DESUL_ATOMICS"; }

#if defined(KOKKOS_COMPILER_GNU) && !defined(__PGIC__) && \
    !defined(__CUDA_ARCH__)

#define KOKKOS_NONTEMPORAL_PREFETCH_LOAD(addr) __builtin_prefetch(addr, 0, 0)
#define KOKKOS_NONTEMPORAL_PREFETCH_STORE(addr) __builtin_prefetch(addr, 1, 0)

#else

#define KOKKOS_NONTEMPORAL_PREFETCH_LOAD(addr) ((void)0)
#define KOKKOS_NONTEMPORAL_PREFETCH_STORE(addr) ((void)0)

#endif
// ============================================================

template<class T> KOKKOS_INLINE_FUNCTION
T atomic_load(T* const dest) { return desul::atomic_load(dest, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
void atomic_store(T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> val) { return desul::atomic_store(dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
void atomic_assign(T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> val) { atomic_store(dest,val); }

KOKKOS_INLINE_FUNCTION
void memory_fence() {
  desul::atomic_thread_fence(desul::MemoryOrderSeqCst(), desul::MemoryScopeDevice());
}

KOKKOS_INLINE_FUNCTION
void load_fence() { return desul::atomic_thread_fence(desul::MemoryOrderAcquire(), desul::MemoryScopeDevice()); }

KOKKOS_INLINE_FUNCTION
void store_fence() { return desul::atomic_thread_fence(desul::MemoryOrderRelease(), desul::MemoryScopeDevice()); }

// atomic_fetch_op
template<class T> KOKKOS_INLINE_FUNCTION
T atomic_fetch_add (T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> val) { return desul::atomic_fetch_add (dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

#ifdef DESUL_IMPL_ATOMIC_CUDA_USE_DOUBLE_ATOMICADD
KOKKOS_INLINE_FUNCTION
double atomic_fetch_add(double* const dest, double val) {
  #ifdef __CUDA_ARCH__
  return atomicAdd(dest,val);
  #else
  return desul::atomic_fetch_add (dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice());
  #endif
};

KOKKOS_INLINE_FUNCTION
double atomic_fetch_sub(double* const dest, double val) {
  #ifdef __CUDA_ARCH__
  return atomicAdd(dest,-val);
  #else
  return desul::atomic_fetch_sub (dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice());
  #endif
};
#endif

template<class T> KOKKOS_INLINE_FUNCTION
T atomic_fetch_sub (T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> val) { return desul::atomic_fetch_sub (dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
T atomic_fetch_max (T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> val) { return desul::atomic_fetch_max (dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
T atomic_fetch_min (T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> val) { return desul::atomic_fetch_min (dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
T atomic_fetch_mul (T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> val) { return desul::atomic_fetch_mul (dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
T atomic_fetch_div (T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> val) { return desul::atomic_fetch_div (dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
T atomic_fetch_mod (T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> val) { return desul::atomic_fetch_mod (dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
T atomic_fetch_and (T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> val) { return desul::atomic_fetch_and (dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
T atomic_fetch_or  (T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> val) { return desul::atomic_fetch_or  (dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
T atomic_fetch_xor (T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> val) { return desul::atomic_fetch_xor (dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
T atomic_fetch_nand(T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> val) { return desul::atomic_fetch_nand(dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
T atomic_fetch_lshift(T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> val) { return desul::atomic_fetch_lshift(dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
T atomic_fetch_rshift(T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> val) { return desul::atomic_fetch_rshift(dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
T atomic_fetch_inc(T* const dest) { return desul::atomic_fetch_inc(dest, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
T atomic_fetch_dec(T* const dest) { return desul::atomic_fetch_dec(dest, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }


// atomic_op_fetch
template<class T> KOKKOS_INLINE_FUNCTION
T atomic_add_fetch (T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> val) { return desul::atomic_add_fetch (dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
T atomic_sub_fetch (T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> val) { return desul::atomic_sub_fetch (dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
T atomic_max_fetch (T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> val) { return desul::atomic_max_fetch (dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
T atomic_min_fetch (T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> val) { return desul::atomic_min_fetch (dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
T atomic_mul_fetch (T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> val) { return desul::atomic_mul_fetch (dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
T atomic_div_fetch (T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> val) { return desul::atomic_div_fetch (dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
T atomic_mod_fetch (T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> val) { return desul::atomic_mod_fetch (dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
T atomic_and_fetch (T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> val) { return desul::atomic_and_fetch (dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
T atomic_or_fetch  (T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> val) { return desul::atomic_or_fetch  (dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
T atomic_xor_fetch (T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> val) { return desul::atomic_xor_fetch (dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
T atomic_nand_fetch(T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> val) { return desul::atomic_nand_fetch(dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
T atomic_lshift_fetch(T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> val) { return desul::atomic_lshift_fetch(dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
T atomic_rshift_fetch(T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> val) { return desul::atomic_rshift_fetch(dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
T atomic_inc_fetch(T* const dest) { return desul::atomic_inc_fetch(dest, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
T atomic_dec_fetch(T* const dest) { return desul::atomic_dec_fetch(dest, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }


// atomic_op
template<class T> KOKKOS_INLINE_FUNCTION
void atomic_add(T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> val) { return desul::atomic_add (dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
void atomic_sub(T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> val) { return desul::atomic_sub (dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
void atomic_mul(T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> val) { return desul::atomic_mul (dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
void atomic_div(T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> val) { return desul::atomic_div (dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
void atomic_min(T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> val) { return desul::atomic_min (dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
void atomic_max(T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> val) { return desul::atomic_max (dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

// FIXME: Desul doesn't have atomic_and yet so call fetch_and
template<class T> KOKKOS_INLINE_FUNCTION
void atomic_and(T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> val) { (void) desul::atomic_fetch_and (dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

// FIXME: Desul doesn't have atomic_or yet so call fetch_or
template<class T> KOKKOS_INLINE_FUNCTION
void atomic_or(T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> val)  { (void) desul::atomic_fetch_or (dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
void atomic_inc(T* const dest) { return desul::atomic_inc(dest, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
void atomic_dec(T* const dest) { return desul::atomic_dec(dest, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
void atomic_increment(T* const dest) { return desul::atomic_inc(dest, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
void atomic_decrement(T* const dest) { return desul::atomic_dec(dest, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

// Exchange

template<class T> KOKKOS_INLINE_FUNCTION
T atomic_exchange(T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> val) { return desul::atomic_exchange(dest, val, desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice()); }

template<class T> KOKKOS_INLINE_FUNCTION
bool atomic_compare_exchange_strong(T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> expected, desul::Impl::dont_deduce_this_parameter_t<const T> desired) {
  T expected_ref = expected;
  return desul::atomic_compare_exchange_strong(dest, expected_ref, desired,
                  desul::MemoryOrderRelaxed(), desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice());
}

template<class T> KOKKOS_INLINE_FUNCTION
T atomic_compare_exchange(T* const dest, desul::Impl::dont_deduce_this_parameter_t<const T> compare, desul::Impl::dont_deduce_this_parameter_t<const T> desired) {
  return desul::atomic_compare_exchange(dest, compare, desired,
                  desul::MemoryOrderRelaxed(), desul::MemoryScopeDevice());
}

namespace Impl {

  template<class MemoryOrder>
  struct KokkosToDesulMemoryOrder;

  template<>
  struct KokkosToDesulMemoryOrder<memory_order_seq_cst_t> {
    using type = desul::MemoryOrderSeqCst;
  };
  template<>
  struct KokkosToDesulMemoryOrder<memory_order_acquire_t> {
    using type = desul::MemoryOrderAcquire;
  };
  template<>
  struct KokkosToDesulMemoryOrder<memory_order_release_t> {
    using type = desul::MemoryOrderRelease;
  };
  template<>
  struct KokkosToDesulMemoryOrder<memory_order_acq_rel_t> {
    using type = desul::MemoryOrderAcqRel;
  };
  template<>
  struct KokkosToDesulMemoryOrder<memory_order_relaxed_t> {
    using type = desul::MemoryOrderRelaxed;
  };
  template<class T, class MemOrderSuccess, class MemOrderFailure> KOKKOS_INLINE_FUNCTION
  bool atomic_compare_exchange_strong(T* const dest, T& expected, const T desired, MemOrderSuccess, MemOrderFailure) {
    return desul::atomic_compare_exchange_strong(dest, expected, desired,
                  typename KokkosToDesulMemoryOrder<MemOrderSuccess>::type(),
                  typename KokkosToDesulMemoryOrder<MemOrderFailure>::type(),
                  desul::MemoryScopeDevice());

  }
  template<class T, class MemoryOrder>
  KOKKOS_INLINE_FUNCTION
  T atomic_load(const T* const src, MemoryOrder) {
    return desul::atomic_load(src, typename KokkosToDesulMemoryOrder<MemoryOrder>::type(), desul::MemoryScopeDevice());
  }
  template<class T, class MemoryOrder>
  KOKKOS_INLINE_FUNCTION
  void atomic_store(T* const src, const T val, MemoryOrder) {
    return desul::atomic_store(src, val, typename KokkosToDesulMemoryOrder<MemoryOrder>::type(), desul::MemoryScopeDevice());
  }
}

}
// clang-format on
#endif  // KOKKOS_ENABLE_IMPL_DESUL_ATOMICS
#endif
