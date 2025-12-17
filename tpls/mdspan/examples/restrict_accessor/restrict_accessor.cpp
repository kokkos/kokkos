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

#include <cassert>
#include <chrono>
#include <iostream>
#include <type_traits>
#include <vector>
#include <string> // stoi

// mfh 2022/08/04: This is based on my comments on reference mdspan
// implementation issue https://github.com/kokkos/mdspan/issues/169.

namespace {


#if defined(MDSPAN_IMPL_COMPILER_MSVC) || defined(__INTEL_COMPILER)
#  define MDSPAN_IMPL_RESTRICT_KEYWORD __restrict
#elif defined(__GNUC__) || defined(__clang__)
#  define MDSPAN_IMPL_RESTRICT_KEYWORD __restrict__
#else
#  define MDSPAN_IMPL_RESTRICT_KEYWORD
#endif

#define MDSPAN_IMPL_RESTRICT_POINTER( ELEMENT_TYPE ) ELEMENT_TYPE * MDSPAN_IMPL_RESTRICT_KEYWORD

// https://en.cppreference.com/w/c/language/restrict gives examples
// of the kinds of optimizations that may apply to restrict.  For instance,
// "[r]estricted pointers can be assigned to unrestricted pointers freely,
// the optimization opportunities remain in place
// as long as the compiler is able to analyze the code:"
//
// void f(int n, float * restrict r, float * restrict s) {
//   float * p = r, * q = s; // OK
//   while(n-- > 0) *p++ = *q++; // almost certainly optimized just like *r++ = *s++
// }
//
// This is relevant because restrict_accessor<ElementType>::reference is _not_ restrict.
// (It's not formally correct to apply C restrict wording to C++ references.
// However, GCC defines this extension:
//
// https://gcc.gnu.org/onlinedocs/gcc/Restricted-Pointers.html
//
// In what follows, I'll assume that this has a reasonable definition.)
// The idea is that even though p[i] has type ElementType& and not ElementType& restrict,
// the compiler can figure out that the reference comes from a pointer based on p,
// which is marked restrict.
//
// Note that any performance improvements can only be determined by experiment.
// Compilers are not required to do anything with restrict.
// Any use of this keyword is not Standard C++,
// so you'll have to refer to the compiler's documentation,
// look at the assembler output, and do performance experiments.
template<class ElementType>
struct restrict_accessor {
  using offset_policy = Kokkos::default_accessor<ElementType>;
  using element_type = ElementType;
  using reference = ElementType&;
  using data_handle_type = MDSPAN_IMPL_RESTRICT_POINTER( ElementType );

  constexpr restrict_accessor() noexcept = default;

  MDSPAN_TEMPLATE_REQUIRES(
    class OtherElementType,
    /* requires */ (std::is_convertible<OtherElementType(*)[], element_type(*)[]>::value)
    )
  constexpr restrict_accessor(restrict_accessor<OtherElementType>) noexcept {}

  constexpr reference access(data_handle_type p, size_t i) const noexcept {
    return p[i];
  }
  constexpr typename offset_policy::data_handle_type
  offset(data_handle_type p, size_t i) const noexcept {
    return p + i;
  }
};

// Use int, not size_t, as the index_type.
// Some compilers have trouble optimizing loops with unsigned or 64-bit index types.
using index_type = int;

template<class ElementType>
using restrict_mdspan_1d =
  Kokkos::mdspan<ElementType, Kokkos::dextents<index_type, 1>, Kokkos::layout_right, restrict_accessor<ElementType>>;

template<class ElementType>
using mdspan_1d =
  Kokkos::mdspan<ElementType, Kokkos::dextents<index_type, 1>, Kokkos::layout_right, Kokkos::default_accessor<ElementType>>;

#define TICK() const auto tick = std::chrono::steady_clock::now()

using double_seconds = std::chrono::duration<double, std::ratio<1>>;

#define TOCK() std::chrono::duration_cast<double_seconds>(std::chrono::steady_clock::now() - tick).count()

template<class ElementType>
void add_restrict_mdspan_1d(restrict_mdspan_1d<const ElementType> x, restrict_mdspan_1d<const ElementType> y, restrict_mdspan_1d<ElementType> z)
{
  const index_type n = z.extent(0);
  for (index_type i = 0; i < n; ++i) {
    z[i] = x[i] + y[i];
  }
}

template<class ElementType>
auto benchmark_add_restrict_mdspan_1d(const std::size_t num_trials, const index_type n, const ElementType* x, const ElementType* y, ElementType* z)
{
  TICK();
  restrict_mdspan_1d<const ElementType> x2{x, n};
  restrict_mdspan_1d<const ElementType> y2{y, n};
  restrict_mdspan_1d<ElementType> z2{z, n};
  for (std::size_t trial = 0; trial < num_trials; ++trial) {
    add_restrict_mdspan_1d(x2, y2, z2);
  }
  return TOCK();
}

template<class ElementType>
void add_mdspan_1d(mdspan_1d<const ElementType> x, mdspan_1d<const ElementType> y, mdspan_1d<ElementType> z)
{
  const index_type n = z.extent(0);
  for (index_type i = 0; i < n; ++i) {
    z[i] = x[i] + y[i];
  }
}

template<class ElementType>
auto benchmark_add_mdspan_1d(const std::size_t num_trials, const index_type n, const ElementType* x, const ElementType* y, ElementType* z)
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
void add_raw_1d(const index_type n, const ElementType x[], const ElementType y[], ElementType z[])
{
  for (index_type i = 0; i < n; ++i) {
    z[i] = x[i] + y[i];
  }
}

template<class ElementType>
auto benchmark_add_raw_1d(const std::size_t num_trials, const index_type n, const ElementType* x, const ElementType* y, ElementType* z)
{
  TICK();
  for (std::size_t trial = 0; trial < num_trials; ++trial) {
    add_raw_1d(n, x, y, z);
  }
  return TOCK();
}

template<class ElementType>
void add_restrict_raw_1d(const index_type n,
  MDSPAN_IMPL_RESTRICT_POINTER(const ElementType) x,
  MDSPAN_IMPL_RESTRICT_POINTER(const ElementType) y,
  MDSPAN_IMPL_RESTRICT_POINTER(ElementType) z)
{
  for (index_type i = 0; i < n; ++i) {
    z[i] = x[i] + y[i];
  }
}

template<class ElementType>
auto benchmark_add_restrict_raw_1d(const std::size_t num_trials, const index_type n, const ElementType* x, const ElementType* y, ElementType* z)
{
  TICK();
  MDSPAN_IMPL_RESTRICT_POINTER(const ElementType) x2 = x;
  MDSPAN_IMPL_RESTRICT_POINTER(const ElementType) y2 = y;
  MDSPAN_IMPL_RESTRICT_POINTER(ElementType) z2 = z;
  for (std::size_t trial = 0; trial < num_trials; ++trial) {
    add_restrict_raw_1d(n, x2, y2, z2);
  }
  return TOCK();
}

// Start with restrict pointers, then assign to nonrestrict pointers.
// This checks whether the compiler's alias analysis suffices.
template<class ElementType>
void add_unrestrict_raw_1d(const index_type n,
  MDSPAN_IMPL_RESTRICT_POINTER(const ElementType) x,
  MDSPAN_IMPL_RESTRICT_POINTER(const ElementType) y,
  MDSPAN_IMPL_RESTRICT_POINTER(ElementType) z)
{
  const ElementType* x2 = x;
  const ElementType* y2 = y;
  ElementType* z2 = z;
  add_raw_1d(n, x2, y2, z2);
}

template<class ElementType>
auto benchmark_add_unrestrict_raw_1d(const std::size_t num_trials, const index_type n, const ElementType* x, const ElementType* y, ElementType* z)
{
  TICK();
  MDSPAN_IMPL_RESTRICT_POINTER(const ElementType) x2 = x;
  MDSPAN_IMPL_RESTRICT_POINTER(const ElementType) y2 = y;
  MDSPAN_IMPL_RESTRICT_POINTER(ElementType) z2 = z;
  for (std::size_t trial = 0; trial < num_trials; ++trial) {
    add_unrestrict_raw_1d(n, x2, y2, z2);
  }
  return TOCK();
}

template<class ElementType>
auto warmup(const std::size_t num_trials, const index_type n, const ElementType* x, const ElementType* y, ElementType* z)
{
  return benchmark_add_raw_1d(num_trials, n, x, y, z);
}

} // namespace (anonymous)

int main(int argc, char* argv[])
{
  using std::cout;
  using std::cerr;
  using std::endl;

  if(argc != 3) {
    cerr << "Usage: main <n> <num_trials>" << endl;
    return -1;
  }

  int n = 10000;
  int num_trials = 100;
  n = std::stoi(argv[1]);
  num_trials = std::stoi(argv[2]);

  std::vector<double> x(n);
  std::vector<double> y(n);
  std::vector<double> z(n);

  auto warmup_result = warmup(num_trials, n, x.data(), y.data(), z.data());
  auto restrict_mdspan_result = benchmark_add_restrict_mdspan_1d(num_trials, n, x.data(), y.data(), z.data());
  auto mdspan_result = benchmark_add_mdspan_1d(num_trials, n, x.data(), y.data(), z.data());
  auto raw_result = benchmark_add_raw_1d(num_trials, n, x.data(), y.data(), z.data());
  auto restrict_raw_result = benchmark_add_restrict_raw_1d(num_trials, n, x.data(), y.data(), z.data());
  auto unrestrict_raw_result = benchmark_add_unrestrict_raw_1d(num_trials, n, x.data(), y.data(), z.data());

  cout << "num_trials: " << num_trials << ", n: " << n << endl
       << "Total time (unit: second):" << endl
       << "  warmup: " << warmup_result << endl
       << "  restrict_mdspan: " << restrict_mdspan_result << endl
       << "  mdspan: " << mdspan_result << endl
       << "  raw_mdspan: " << raw_result << endl
       << "  restrict_raw: " << restrict_raw_result << endl
       << "  unrestrict_raw: " << unrestrict_raw_result << endl;
  return 0;
}
