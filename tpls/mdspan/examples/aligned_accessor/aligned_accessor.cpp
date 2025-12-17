//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER
#include <mdspan/mdspan.hpp>

// Just checking __cpp_lib_int_pow2 isn't enough for some GCC versions.
// The <bit> header exists, but std::has_single_bit does not.
#if defined(__cpp_lib_int_pow2) && __cplusplus >= 202002L
#  include <bit>
#endif
#include <cassert>
#include <chrono>
#include <cstdlib> // aligned_alloc, posix_memalign (if applicable)
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <type_traits>
#include <string> // stoi

// mfh 2022/08/08: This is based on my comment on RAPIDS RAFT issue 725:
// https://github.com/rapidsai/raft/pull/725#discussion_r937991701

namespace {

using test_value_type = float;
constexpr std::size_t min_overalignment_factor = 8;
constexpr std::size_t min_byte_alignment = min_overalignment_factor * sizeof(float);

// Use int, not size_t, as the index_type.
// Some compilers have trouble optimizing loops with unsigned or 64-bit index types.
using index_type = int;


// Prefer std::assume_aligned if available, as it is in the C++ Standard.
// Otherwise, use a compiler-specific equivalent if available.

// NOTE (mfh 2022/08/08) BYTE_ALIGNMENT must be unsigned and a power of 2.
#if defined(__cpp_lib_assume_aligned)
#  define MDSPAN_IMPL_ASSUME_ALIGNED( ELEMENT_TYPE, POINTER, BYTE_ALIGNMENT ) (std::assume_aligned< BYTE_ALIGNMENT >( POINTER ))
  constexpr char assume_aligned_method[] = "std::assume_aligned";
#elif defined(__ICL)
#  define MDSPAN_IMPL_ASSUME_ALIGNED( ELEMENT_TYPE, POINTER, BYTE_ALIGNMENT ) POINTER
  constexpr char assume_aligned_method[] = "(none)";
#elif defined(__ICC)
#  define MDSPAN_IMPL_ASSUME_ALIGNED( ELEMENT_TYPE, POINTER, BYTE_ALIGNMENT ) POINTER
  constexpr char assume_aligned_method[] = "(none)";
#elif defined(__clang__)
#  define MDSPAN_IMPL_ASSUME_ALIGNED( ELEMENT_TYPE, POINTER, BYTE_ALIGNMENT ) POINTER
  constexpr char assume_aligned_method[] = "(none)";
#elif defined(__GNUC__)
  // __builtin_assume_aligned returns void*
#  define MDSPAN_IMPL_ASSUME_ALIGNED( ELEMENT_TYPE, POINTER, BYTE_ALIGNMENT ) reinterpret_cast< ELEMENT_TYPE* >(__builtin_assume_aligned( POINTER, BYTE_ALIGNMENT ))
  constexpr char assume_aligned_method[] = "__builtin_assume_aligned";
#else
#  define MDSPAN_IMPL_ASSUME_ALIGNED( ELEMENT_TYPE, POINTER, BYTE_ALIGNMENT ) POINTER
  constexpr char assume_aligned_method[] = "(none)";
#endif

// Some compilers other than Clang or GCC like to define __clang__ or __GNUC__.
// Thus, we order the tests from most to least specific.
#if defined(__ICL)
#  define MDSPAN_IMPL_ALIGN_VALUE_ATTRIBUTE( BYTE_ALIGNMENT ) __declspec(align_value( BYTE_ALIGNMENT ))
  constexpr char align_attribute_method[] = "__declspec(align_value(BYTE_ALIGNMENT))";
#elif defined(__ICC)
#  define MDSPAN_IMPL_ALIGN_VALUE_ATTRIBUTE( BYTE_ALIGNMENT ) __attribute__((align_value( BYTE_ALIGNMENT )))
  constexpr char align_attribute_method[] = "__attribute__((align_value(BYTE_ALIGNMENT)))";
#elif defined(__clang__)
#  define MDSPAN_IMPL_ALIGN_VALUE_ATTRIBUTE( BYTE_ALIGNMENT ) __attribute__((align_value( BYTE_ALIGNMENT )))
  constexpr char align_attribute_method[] = "__attribute__((align_value(BYTE_ALIGNMENT)))";
#else
#  define MDSPAN_IMPL_ALIGN_VALUE_ATTRIBUTE( BYTE_ALIGNMENT )
  constexpr char align_attribute_method[] = "(none)";
#endif

constexpr bool
is_nonzero_power_of_two(const std::size_t x)
{
// Just checking __cpp_lib_int_pow2 isn't enough for some GCC versions.
// The <bit> header exists, but std::has_single_bit does not.
#if defined(__cpp_lib_int_pow2) && __cplusplus >= 202002L
  return std::has_single_bit(x);
#else
  return x != 0 && (x & (x - 1)) == 0;
#endif
}

template<class ElementType>
constexpr bool
valid_byte_alignment(const std::size_t byte_alignment)
{
  return is_nonzero_power_of_two(byte_alignment) && byte_alignment >= alignof(ElementType);
}

// We define aligned_pointer_t through a struct
// so we can check whether the byte alignment is valid.
// This makes it impossible to use the alias
// with an invalid byte alignment.
template<class ElementType, std::size_t byte_alignment>
struct aligned_pointer {
  static_assert(valid_byte_alignment<ElementType>(byte_alignment),
		"byte_alignment must be a power of two no less than "
		"the minimum required alignment of ElementType.");

#if defined(__ICC)
  // x86-64 ICC 2021.5.0 emits warning #3186 ("expected typedef declaration") here.
  // No other compiler (including Clang, which has a similar type attribute) has this issue.
#  pragma warning push
#  pragma warning disable 3186
#endif

  using type = ElementType* MDSPAN_IMPL_ALIGN_VALUE_ATTRIBUTE( byte_alignment );

#if defined(__ICC)
#  pragma warning pop
#endif
};

template<class ElementType, std::size_t byte_alignment>
using aligned_pointer_t = typename aligned_pointer<ElementType, byte_alignment>::type;

template<class ElementType, std::size_t byte_alignment>
aligned_pointer_t<ElementType, byte_alignment>
bless(ElementType* ptr, std::integral_constant<std::size_t, byte_alignment> /* ba */ )
{
  return MDSPAN_IMPL_ASSUME_ALIGNED( ElementType, ptr, byte_alignment );
}

template<class ElementType, std::size_t byte_alignment>
struct aligned_accessor {
  using offset_policy = Kokkos::default_accessor<ElementType>;
  using element_type = ElementType;
  using reference = ElementType&;
  using data_handle_type = aligned_pointer_t<ElementType, byte_alignment>;

  constexpr aligned_accessor() noexcept = default;

  MDSPAN_TEMPLATE_REQUIRES(
    class OtherElementType,
    std::size_t other_byte_alignment,
    /* requires */ (std::is_convertible<OtherElementType(*)[], element_type(*)[]>::value && other_byte_alignment == byte_alignment)
    )
  constexpr aligned_accessor(aligned_accessor<OtherElementType, other_byte_alignment>) noexcept {}

  constexpr reference access(data_handle_type p, size_t i) const noexcept {
    // This may declare alignment twice, depending on
    // if we have an attribute for marking pointer types.
    return MDSPAN_IMPL_ASSUME_ALIGNED( ElementType, p, byte_alignment )[i];
  }

  constexpr typename offset_policy::data_handle_type
  offset(data_handle_type p, size_t i) const noexcept {
    return p + i;
  }
};

template<class ElementType>
struct delete_raw {
  void operator()(ElementType* p) const {
    if (p != nullptr) {
      // All the aligned allocation methods below go with std::free.
      // If we implement a new method that uses a different
      // deallocation function, that function would go here.
      std::free(p);
    }
  }
};

template<class ElementType>
using allocation_t = std::unique_ptr<
    ElementType[],
    delete_raw<ElementType>
  >;

template<class ElementType, std::size_t byte_alignment>
allocation_t<ElementType>
allocate_raw(const std::size_t num_elements)
{
  static_assert(byte_alignment >= sizeof(ElementType),
		"byte_alignment must be at least sizeof(ElementType).");
  static constexpr std::size_t overalignment = byte_alignment / sizeof(ElementType);
  static_assert(overalignment * sizeof(ElementType) == byte_alignment,
		"overalignment * sizeof(ElementType) must equal byte_alignment.");

  const std::size_t num_bytes = num_elements * sizeof(ElementType);
  auto deleter = delete_raw<ElementType>{};

  void* ptr = nullptr;
#ifdef _MSC_VER
  // MSVC 19.32 (or "latest" on godbolt.org) does NOT have aligned_alloc,
  // even with /std:c++latest.  Instead, we use MSVC-specific _aligned_malloc.
  ptr = _aligned_malloc(num_bytes, byte_alignment);
#elif __cplusplus < 201703L || (defined(__APPLE__) && defined(__apple_build_version__))
  // aligned_alloc is a C11 function.  Apple Clang and GCC do not provide it with
  // -std=c++14.  We have coverage for the Windows case (at least with MSVC) above,
  // so we can resort to the POSIX function posix_memalign.
  const int err = posix_memalign(&ptr, byte_alignment, num_bytes);
  if(err != 0) {
    if(err == EINVAL) {
      throw std::runtime_error("posix_memalign failed: alignment not a power of two");
    } else if(err == ENOMEM) {
      throw std::runtime_error("posix_memalign failed: insufficient memory");
    } else {
      throw std::runtime_error("posix_memalign failed: unknown error");
    }
  }
#else
  ptr = std::aligned_alloc(byte_alignment, num_bytes);
#endif

  return {reinterpret_cast<ElementType*>(ptr), deleter};
}

// A dynamically allocated array of ElementType,
// whose zeroth element has byte alignment byte_alignment.
//
// ElementType: A trivially copyable type whose size is a power of two bytes.
// byte_alignment: A power of two that is a multiple of sizeof(ElementType).
//
// This needs to be a class in order to preserve the invariant that
// "pointer" points to the allocation.
template<class ElementType, std::size_t byte_alignment>
class aligned_array_allocation {
  static_assert(byte_alignment >= sizeof(ElementType),
		"byte_alignment must be at least sizeof(ElementType).");
  static constexpr std::size_t overalignment = byte_alignment / sizeof(ElementType);
  static_assert(overalignment * sizeof(ElementType) == byte_alignment,
		"overalignment * sizeof(ElementType) must equal byte_alignment.");
public:
  aligned_array_allocation(std::size_t number_of_elements) :
    allocation(allocate_raw<ElementType, byte_alignment>(number_of_elements)),
    pointer(allocation.get()),
    num_elements(number_of_elements)
  {}

  aligned_pointer_t<ElementType, byte_alignment> data() const
  {
    return MDSPAN_IMPL_ASSUME_ALIGNED( ElementType, pointer, byte_alignment );
  }

private:
  allocation_t<ElementType> allocation{nullptr, delete_raw<ElementType>{}};
  aligned_pointer_t<ElementType, byte_alignment> pointer{nullptr};
  std::size_t num_elements{0};
};

// This represents an array allocation that is deliberately
// aligned at most to sizeof(ElementType) bytes.
//
// We want this "unaligned" memory to make sure that the compiler or
// C++ Standard Library implementation isn't adding overalignment.
// For example, if the compiler knows that float arrays are always
// allocated to 2*sizeof(float) alignment, it could make loops use
// 2-wide SIMD code without needing a loop prelude or postlude to
// handle the "unaligned remainder."  Forcing "unalignment" will thus
// give us a better performance comparison with the aligned case.
template<class ElementType>
class deliberately_unaligned_array_allocation {
public:
  deliberately_unaligned_array_allocation(std::size_t number_of_elements) :
    allocation(allocate_raw<ElementType, sizeof(ElementType)>(number_of_elements + 1)),
    pointer(unaligned_pointer(allocation.get())),
    num_elements(number_of_elements)
  {}

  ElementType* data() const {
    return pointer;
  }

private:
  // Just asking for the minimum alignment of sizeof(ElementType)
  // bytes isn't enough, because the allocator might overalign that
  // allocation.  Instead, we ask for an extra element, check whether
  // the allocation is "odd" or "even," and add one if needed.
  static ElementType*
  unaligned_pointer(ElementType* allocation_pointer)
  {
    if(allocation_pointer == nullptr) {
      return nullptr;
    }
    const auto ptr_as_uint = reinterpret_cast<std::uintptr_t>(allocation_pointer);
    const auto bias = ptr_as_uint % std::uintptr_t(2 * sizeof(ElementType));
    if(bias == 0) {
      // It's aligned to sizeof(ElementType) times 2^k for integer k > 0.
      // Add one (that is, sizeof(ElementType) bytes) to make it "odd" again.
      return allocation_pointer + 1;
    } else {
      return allocation_pointer;
    }
  }

  allocation_t<ElementType> allocation{nullptr, delete_raw<ElementType>{}};
  ElementType* pointer{nullptr};
  std::size_t num_elements{0};
};

template<class ElementType, std::size_t byte_alignment>
using aligned_mdspan_1d =
  Kokkos::mdspan<ElementType,
		Kokkos::dextents<index_type, 1>,
		Kokkos::layout_right,
		aligned_accessor<ElementType, byte_alignment>>;

template<class ElementType>
using mdspan_1d =
  Kokkos::mdspan<ElementType,
		Kokkos::dextents<index_type, 1>,
		Kokkos::layout_right,
		Kokkos::default_accessor<ElementType>>;

#define TICK() const auto tick = std::chrono::steady_clock::now()

using double_seconds = std::chrono::duration<double, std::ratio<1>>;

#define TOCK() std::chrono::duration_cast<double_seconds>(std::chrono::steady_clock::now() - tick).count()

template<class ElementType, std::size_t byte_alignment>
void add_aligned_mdspan_1d(aligned_mdspan_1d<const ElementType, byte_alignment> x,
			   aligned_mdspan_1d<const ElementType, byte_alignment> y,
			   aligned_mdspan_1d<ElementType, byte_alignment> z)
{
  const index_type n = z.extent(0);
  for (index_type i = 0; i < n; ++i) {
    z[i] = x[i] + y[i];
  }
}

template<class ElementType, std::size_t byte_alignment>
auto benchmark_add_aligned_mdspan_1d(const std::size_t num_trials,
				     const index_type n,
				     const ElementType x[],
				     const ElementType y[],
				     ElementType z[],
				     std::integral_constant<std::size_t, byte_alignment> /* ba */ )
{
  TICK();
  aligned_mdspan_1d<const ElementType, byte_alignment> x2{x, n};
  aligned_mdspan_1d<const ElementType, byte_alignment> y2{y, n};
  aligned_mdspan_1d<ElementType, byte_alignment> z2{z, n};
  for (std::size_t trial = 0; trial < num_trials; ++trial) {
    add_aligned_mdspan_1d(x2, y2, z2);
  }
  return TOCK();
}

template<class ElementType>
void add_mdspan_1d(mdspan_1d<const ElementType> x,
		   mdspan_1d<const ElementType> y,
		   mdspan_1d<ElementType> z)
{
  const index_type n = z.extent(0);
  for (index_type i = 0; i < n; ++i) {
    z[i] = x[i] + y[i];
  }
}

template<class ElementType>
auto benchmark_add_mdspan_1d(const std::size_t num_trials,
			     const index_type n,
			     const ElementType x[],
			     const ElementType y[],
			     ElementType z[])
{
  TICK();
  mdspan_1d<const ElementType> x2{x, n};
  mdspan_1d<const ElementType> y2{y, n};
  mdspan_1d<ElementType> z2{z, n};
  for (std::size_t trial = 0; trial < num_trials; ++trial) {
    add_mdspan_1d(x2, y2, z2);
  }
  return TOCK();
}

template<class ElementType>
void add_raw_1d(const index_type n,
		const ElementType x[],
		const ElementType y[],
		ElementType z[])
{
  for (index_type i = 0; i < n; ++i) {
    z[i] = x[i] + y[i];
  }
}

template<class ElementType>
auto benchmark_add_raw_1d(const std::size_t num_trials,
			  const index_type n,
			  const ElementType x[],
			  const ElementType y[],
			  ElementType z[])
{
  TICK();
  for (std::size_t trial = 0; trial < num_trials; ++trial) {
    add_raw_1d(n, x, y, z);
  }
  return TOCK();
}

// Assume that x, y, and z all have the same alignment.
template<class ElementType, std::size_t byte_alignment>
void add_aligned_raw_1d(const index_type n,
			aligned_pointer_t<const ElementType, byte_alignment> x,
			aligned_pointer_t<const ElementType, byte_alignment> y,
			aligned_pointer_t<ElementType, byte_alignment> z)
{
  for (index_type i = 0; i < n; ++i) {
    z[i] = x[i] + y[i];
  }
}

// Assume that x, y, and z all have the same alignment.
template<class ElementType, std::size_t byte_alignment>
auto benchmark_add_aligned_raw_1d(const std::size_t num_trials,
				  const index_type n,
				  const ElementType x[],
				  const ElementType y[],
				  ElementType z[],
				  std::integral_constant<std::size_t, byte_alignment> ba)
{
  TICK();
  auto x_blessed = bless(x, ba);
  auto y_blessed = bless(y, ba);
  auto z_blessed = bless(z, ba);
  for (std::size_t trial = 0; trial < num_trials; ++trial) {
    add_aligned_raw_1d<ElementType, byte_alignment>(n, x_blessed, y_blessed, z_blessed);
  }
  return TOCK();
}

#ifdef _OPENMP
template<class ElementType, std::size_t byte_alignment>
void add_omp_simd_aligned_mdspan_1d(aligned_mdspan_1d<const ElementType, byte_alignment> x,
				    aligned_mdspan_1d<const ElementType, byte_alignment> y,
				    aligned_mdspan_1d<ElementType, byte_alignment> z)
{
  const index_type n = z.extent(0);
  // Test whether OpenMP can figure out that the pointers are aligned.
#pragma omp simd
  for (index_type i = 0; i < n; ++i) {
    z[i] = x[i] + y[i];
  }
}

template<class ElementType, std::size_t byte_alignment>
auto benchmark_add_omp_simd_aligned_mdspan_1d(const std::size_t num_trials,
					      const index_type n,
					      const ElementType x[],
					      const ElementType y[],
					      ElementType z[],
					      std::integral_constant<std::size_t, byte_alignment> /* ba */ )
{
  TICK();
  aligned_mdspan_1d<const ElementType, byte_alignment> x2{x, n};
  aligned_mdspan_1d<const ElementType, byte_alignment> y2{y, n};
  aligned_mdspan_1d<ElementType, byte_alignment> z2{z, n};
  for (std::size_t trial = 0; trial < num_trials; ++trial) {
    add_omp_simd_aligned_mdspan_1d(x2, y2, z2);
  }
  return TOCK();
}

template<class ElementType>
void add_omp_simd_mdspan_1d(mdspan_1d<const ElementType> x,
			    mdspan_1d<const ElementType> y,
			    mdspan_1d<ElementType> z)
{
  const index_type n = z.extent(0);
#pragma omp simd
  for (index_type i = 0; i < n; ++i) {
    z[i] = x[i] + y[i];
  }
}

template<class ElementType>
auto benchmark_add_omp_simd_mdspan_1d(const std::size_t num_trials,
				      const index_type n,
				      const ElementType x[],
				      const ElementType y[],
				      ElementType z[])
{
  TICK();
  mdspan_1d<const ElementType> x2{x, n};
  mdspan_1d<const ElementType> y2{y, n};
  mdspan_1d<ElementType> z2{z, n};
  for (std::size_t trial = 0; trial < num_trials; ++trial) {
    add_omp_simd_mdspan_1d(x2, y2, z2);
  }
  return TOCK();
}

template<class ElementType>
void add_omp_simd_raw_1d(const index_type n,
			 const ElementType x[],
			 const ElementType y[],
			 ElementType z[])
{
#pragma omp simd
  for (index_type i = 0; i < n; ++i) {
    z[i] = x[i] + y[i];
  }
}

template<class ElementType>
auto benchmark_add_omp_simd_raw_1d(const std::size_t num_trials,
				   const index_type n,
				   const ElementType x[],
				   const ElementType y[],
				   ElementType z[])
{
  TICK();
  for (std::size_t trial = 0; trial < num_trials; ++trial) {
    add_omp_simd_raw_1d(n, x, y, z);
  }
  return TOCK();
}

template<class ElementType, std::size_t byte_alignment>
void add_omp_aligned_simd_raw_1d(const index_type n,
				 const ElementType x[],
				 const ElementType y[],
				 ElementType z[],
				 std::integral_constant<std::size_t, byte_alignment> /* ba */ )
{
  // The "aligned" clause might only work
  // for pointers or (raw) arrays.
  // C++23 adopting mdspan might change that,
  // at least for the default layout and accessor.
#pragma omp simd aligned(z,x,y:byte_alignment)
  for (index_type i = 0; i < n; ++i) {
    z[i] = x[i] + y[i];
  }
}

template<class ElementType, std::size_t byte_alignment>
auto benchmark_add_omp_aligned_simd_raw_1d(const std::size_t num_trials,
					   const index_type n,
					   const ElementType x[],
					   const ElementType y[],
					   ElementType z[],
					   std::integral_constant<std::size_t, byte_alignment> ba)
{
  TICK();
  for (std::size_t trial = 0; trial < num_trials; ++trial) {
    add_omp_aligned_simd_raw_1d(n, x, y, z, ba);
  }
  return TOCK();
}

template<class ElementType, std::size_t byte_alignment>
void add_omp_simd_aligned_raw_1d(const index_type n,
				 aligned_pointer_t<const ElementType, byte_alignment> x,
				 aligned_pointer_t<const ElementType, byte_alignment> y,
				 aligned_pointer_t<ElementType, byte_alignment> z)
{
#pragma omp simd
  for (index_type i = 0; i < n; ++i) {
    z[i] = x[i] + y[i];
  }
}

// Assume that x, y, and z all have the same alignment.
template<class ElementType, std::size_t byte_alignment>
auto benchmark_add_omp_simd_aligned_raw_1d(const std::size_t num_trials,
					   const index_type n,
					   const ElementType x[],
					   const ElementType y[],
					   ElementType z[],
					   std::integral_constant<std::size_t, byte_alignment> ba)
{
  TICK();
  auto x_blessed = bless(x, ba);
  auto y_blessed = bless(y, ba);
  auto z_blessed = bless(z, ba);
  for (std::size_t trial = 0; trial < num_trials; ++trial) {
    add_omp_simd_aligned_raw_1d<ElementType, byte_alignment>(n, x_blessed, y_blessed, z_blessed);
  }
  return TOCK();
}

template<class ElementType, std::size_t byte_alignment>
void add_omp_aligned_simd_aligned_raw_1d(const index_type n,
					 aligned_pointer_t<const ElementType, byte_alignment> x,
					 aligned_pointer_t<const ElementType, byte_alignment> y,
					 aligned_pointer_t<ElementType, byte_alignment> z)
{
#pragma omp simd aligned(z,x,y:byte_alignment)
  for (index_type i = 0; i < n; ++i) {
    z[i] = x[i] + y[i];
  }
}

// Assume that x, y, and z all have the same alignment.
template<class ElementType, std::size_t byte_alignment>
auto benchmark_add_omp_aligned_simd_aligned_raw_1d(
  const std::size_t num_trials,
  const index_type n,
  const ElementType x[],
  const ElementType y[],
  ElementType z[],
  std::integral_constant<std::size_t, byte_alignment> ba)
{
  TICK();
  auto x_blessed = bless(x, ba);
  auto y_blessed = bless(y, ba);
  auto z_blessed = bless(z, ba);
  for (std::size_t trial = 0; trial < num_trials; ++trial) {
    // Passing in ba doesn't help the compiler
    // deduce the template parameters,
    // so we just specify them explicitly.
    add_omp_aligned_simd_aligned_raw_1d<ElementType, byte_alignment>(n, x_blessed, y_blessed, z_blessed);
  }
  return TOCK();
}
#endif // _OPENMP

template<class ElementType>
void set_elements_of_arrays(const index_type n,
			    ElementType x[],
			    ElementType y[],
			    ElementType z[])
{
  for (index_type i = 0; i < n; ++i) {
    x[i] = 1.0;
    y[i] = 2.0;
    z[i] = 0.0;
  }
}

} // namespace (anonymous)

int main(int argc, char* argv[])
{
  using std::cout;
  using std::cerr;
  using std::endl;
  constexpr std::integral_constant<std::size_t, min_byte_alignment> byte_alignment;

  if(argc != 3) {
    cerr << "Usage: main <n> <num_trials>" << endl;
    return -1;
  }

  const int n = std::stoi(argv[1]);
  const int num_trials = std::stoi(argv[2]);

  aligned_array_allocation<float, min_byte_alignment> x_aligned(n);
  aligned_array_allocation<float, min_byte_alignment> y_aligned(n);
  aligned_array_allocation<float, min_byte_alignment> z_aligned(n);
  set_elements_of_arrays(n, x_aligned.data(), y_aligned.data(), z_aligned.data());

  auto aligned_mdspan_result =
    benchmark_add_aligned_mdspan_1d(num_trials, n, x_aligned.data(),
				    y_aligned.data(), z_aligned.data(),
				    byte_alignment);
  auto aligned_raw_result =
    benchmark_add_aligned_raw_1d(num_trials, n, x_aligned.data(),
				 y_aligned.data(), z_aligned.data(),
				 byte_alignment);
#ifdef _OPENMP
  auto omp_simd_aligned_mdspan_result =
    benchmark_add_omp_simd_aligned_mdspan_1d(num_trials, n, x_aligned.data(),
					     y_aligned.data(), z_aligned.data(),
					     byte_alignment);
  auto omp_aligned_simd_raw_result =
    benchmark_add_omp_aligned_simd_raw_1d(num_trials, n,
					  x_aligned.data(), y_aligned.data(),
					  z_aligned.data(), byte_alignment);
  auto omp_simd_aligned_raw_result =
    benchmark_add_omp_simd_aligned_raw_1d(num_trials, n,
					  x_aligned.data(), y_aligned.data(),
					  z_aligned.data(), byte_alignment);
  auto omp_aligned_simd_aligned_raw_result =
    benchmark_add_omp_aligned_simd_aligned_raw_1d(num_trials, n,
						  x_aligned.data(), y_aligned.data(),
						  z_aligned.data(), byte_alignment);
#endif // _OPENMP

  deliberately_unaligned_array_allocation<float> x_unaligned(n);
  deliberately_unaligned_array_allocation<float> y_unaligned(n);
  deliberately_unaligned_array_allocation<float> z_unaligned(n);
  set_elements_of_arrays(n, x_unaligned.data(), y_unaligned.data(), z_unaligned.data());

  auto mdspan_result =
    benchmark_add_mdspan_1d(num_trials, n, x_unaligned.data(),
			    y_unaligned.data(), z_unaligned.data());
  auto raw_result =
    benchmark_add_raw_1d(num_trials, n, x_unaligned.data(),
			 y_unaligned.data(), z_unaligned.data());

#ifdef _OPENMP
  auto omp_simd_mdspan_result =
    benchmark_add_omp_simd_mdspan_1d(num_trials, n, x_unaligned.data(),
				     y_unaligned.data(), z_unaligned.data());
  auto omp_simd_raw_result =
    benchmark_add_omp_simd_raw_1d(num_trials, n, x_unaligned.data(),
				  y_unaligned.data(), z_unaligned.data());
#endif // _OPENMP

  cout << "Number of trials: " << num_trials << endl
       << "Number of loop iterations per trial: " << n << endl
       << "Way to declare a pointer value aligned, if any: "
       << assume_aligned_method << endl
       << "Way to declare a pointer type aligned, if any: "
       << align_attribute_method << endl
       << "Total time in seconds for non-OpenMP loops:" << endl
       << "  aligned mdspan: " << aligned_mdspan_result << endl
       << "  unaligned mdspan: " << mdspan_result << endl
       << "  aligned raw: " << aligned_raw_result << endl
       << "  unaligned raw: " << raw_result << endl;

#ifdef _OPENMP
  cout << "Total time in seconds for OpenMP (omp simd) loops:" << endl
       << "  omp_simd_aligned_mdspan: " << omp_simd_aligned_mdspan_result << endl
       << "  omp_simd_mdspan: " << omp_simd_mdspan_result << endl
       << "  omp_simd_raw: " << omp_simd_raw_result << endl
       << "  omp_aligned_simd_raw (aligned clause): "
       << omp_aligned_simd_raw_result << endl
       << "  omp_simd_aligned_raw (pointer declarations): "
       << omp_simd_aligned_raw_result << endl
       << "  omp_aligned_simd_aligned_raw (both): "
       << omp_aligned_simd_aligned_raw_result << endl;
#endif // _OPENMP
  return 0;
}
