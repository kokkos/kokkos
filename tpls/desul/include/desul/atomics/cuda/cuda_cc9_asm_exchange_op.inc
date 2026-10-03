
#define __DESUL_IMPL_CUDA_ASM_ATOMIC_EXCHANGE() \
template<class ctype> \
inline __device__ typename ::std::enable_if<sizeof(ctype)==16, ctype>::type device_atomic_exchange(ctype* dest, ctype value, __DESUL_IMPL_CUDA_ASM_MEMORY_ORDER, __DESUL_IMPL_CUDA_ASM_MEMORY_SCOPE) { \
  ulong2 asm_value = *reinterpret_cast<ulong2*>(&value); \
  ulong2 asm_result; \
  asm volatile("atom.exch" __DESUL_IMPL_CUDA_ASM_MEMORY_ORDER_ASM __DESUL_IMPL_CUDA_ASM_MEMORY_SCOPE_ASM ".b128" " {%0,%1},[%2],{%3,%4};" : "=l"(asm_result.x),"=l"(asm_result.y) : "l"(dest), "l"(asm_value.x),"l"(asm_value.y) : "memory"); \
  return reinterpret_cast<ctype&>(asm_result); \
}

#define __DESUL_IMPL_CUDA_ASM_ATOMIC_COMPARE_EXCHANGE() \
template<class ctype> \
inline __device__ typename ::std::enable_if<sizeof(ctype)==16, ctype>::type device_atomic_compare_exchange(ctype* dest, ctype compare, ctype value, __DESUL_IMPL_CUDA_ASM_MEMORY_ORDER, __DESUL_IMPL_CUDA_ASM_MEMORY_SCOPE) { \
  ulong2 asm_value = *reinterpret_cast<ulong2*>(&value); \
  ulong2 asm_compare = *reinterpret_cast<ulong2*>(&compare); \
  ulong2 asm_result; \
  asm volatile("atom.cas" __DESUL_IMPL_CUDA_ASM_MEMORY_ORDER_ASM __DESUL_IMPL_CUDA_ASM_MEMORY_SCOPE_ASM ".b128" " {%0,%1},[%2],{%3,%4},{%5,%6};" : "=l"(asm_result.x),"=l"(asm_result.y) : "l"(dest),"l"(asm_compare.x),"l"(asm_compare.y),"l"(asm_value.x),"l"(asm_value.y) : "memory"); \
  return reinterpret_cast<ctype&>(asm_result); \
}

__DESUL_IMPL_CUDA_ASM_ATOMIC_EXCHANGE()
__DESUL_IMPL_CUDA_ASM_ATOMIC_COMPARE_EXCHANGE()

#undef __DESUL_IMPL_CUDA_ASM_ATOMIC_EXCHANGE
#undef __DESUL_IMPL_CUDA_ASM_ATOMIC_COMPARE_EXCHANGE
