/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <gtest/gtest.h>

#include <Kokkos_Simd.hpp>

#include <impl/Kokkos_Error.hpp>

#include <Kokkos_MinMaxClamp.hpp>

class gtest_checker {
 public:
  void truth(bool x) const
  {
    EXPECT_TRUE(x);
  }
  template <class T>
  void equality(T const& a, T const& b) const
  {
    if (a != b) std::abort();
    EXPECT_EQ(a, b);
  }
};

class kokkos_checker {
 public:
  KOKKOS_INLINE_FUNCTION void truth(bool x) const
  {
    KOKKOS_ASSERT(x);
  }
  template <class T>
  KOKKOS_INLINE_FUNCTION void equality(T const& a, T const& b) const
  {
    KOKKOS_ASSERT(a == b);
  }
};

template <class Checker, class T, class Abi>
KOKKOS_INLINE_FUNCTION void check_equality(
    Kokkos::Experimental::simd<T, Abi> const& expected_result,
    Kokkos::Experimental::simd<T, Abi> const& computed_result)
{
  Checker checker;
  checker.truth(all_of(expected_result == computed_result));
  checker.truth(none_of(expected_result != computed_result));
  // double-check that with manual comparison of each entry
  for (std::size_t i = 0; i < expected_result.size(); ++i) {
    checker.equality(expected_result[i], computed_result[i]);
  }
}

class load_element_aligned {
 public:
  template <class T, class Abi>
  KOKKOS_INLINE_FUNCTION bool operator()(
      T const* mem,
      std::size_t n,
      Kokkos::Experimental::simd<T, Abi>& result) const
  {
    if (n < result.size()) return false;
    result.copy_from(mem, Kokkos::Experimental::element_aligned_tag());
    return true;
  }
};

class load_masked {
 public:
  template <class T, class Abi>
  KOKKOS_INLINE_FUNCTION bool operator()(
      T const* mem,
      std::size_t n,
      Kokkos::Experimental::simd<T, Abi>& result) const
  {
    using mask_type = typename Kokkos::Experimental::simd<T, Abi>::mask_type;
    mask_type mask(false);
    for (std::size_t i = 0; i < n; ++i) {
      mask[i] = true;
    }
    where(mask, result).copy_from(mem, Kokkos::Experimental::element_aligned_tag());
    where(!mask, result) = T(0);
    return true;
  }
};

class load_as_scalars {
 public:
  template <class T, class Abi>
  KOKKOS_INLINE_FUNCTION bool operator()(
      T const* mem,
      std::size_t n,
      Kokkos::Experimental::simd<T, Abi>& result) const
  {
    for (std::size_t i = 0; i < n; ++i) {
      result[i] = mem[i];
    }
    for (std::size_t i = n; i < result.size(); ++i) {
      result[i] = T(0);
    }
    return true;
  }
};

template <class Abi, class Checker, class Loader, class BinaryOp, class T>
__attribute__((noinline)) void check_binary_op_one_loader(
    BinaryOp binary_op,
    std::size_t n,
    T const* first_args,
    T const* second_args)
{
  Loader loader;
  using simd_type = Kokkos::Experimental::simd<T, Abi>;
  std::size_t constexpr width = simd_type::size();
  for (std::size_t i = 0; i < n; i += width) {
    std::size_t const nremaining = n - i;
    std::size_t const nlanes = Kokkos::Experimental::min(nremaining, width);
    simd_type first_arg;
    bool const loaded_first_arg = loader(first_args + i, nlanes, first_arg);
    simd_type second_arg;
    bool const loaded_second_arg = loader(second_args + i, nlanes, second_arg);
    if (!(loaded_first_arg && loaded_second_arg)) continue;
    simd_type expected_result;
    for (std::size_t i = 0; i < width; ++i) {
      expected_result[i] = binary_op(first_arg[i], second_arg[i]);
    }
    simd_type const computed_result = binary_op(first_arg, second_arg);
    check_equality<Checker>(expected_result, computed_result);
  }
}

template <class Abi, class Checker, class BinaryOp, class T>
KOKKOS_INLINE_FUNCTION void check_binary_op_all_loaders(
    BinaryOp binary_op,
    std::size_t n,
    T const* first_args,
    T const* second_args)
{
  check_binary_op_one_loader<Abi, Checker, load_element_aligned>(
      binary_op, n, first_args, second_args);
  check_binary_op_one_loader<Abi, Checker, load_masked>(
      binary_op, n, first_args, second_args);
  check_binary_op_one_loader<Abi, Checker, load_as_scalars>(
      binary_op, n, first_args, second_args);
}

class plus {
 public:
  template <class T>
  KOKKOS_INLINE_FUNCTION auto operator()(T const& a, T const& b) const
  {
    return a + b;
  }
};

template <class Abi, class Checker>
KOKKOS_INLINE_FUNCTION void check_addition()
{
  std::size_t constexpr n = 7;
  double const first_args[n] =  {1, 2, -1, 10, 0,  1, -2};
  double const second_args[n] = {1, 2,  1,  1, 0, -3, -2};
  check_binary_op_all_loaders<Abi, Checker>(plus(), n, first_args, second_args);
}

template <class Abi, class Checker>
KOKKOS_INLINE_FUNCTION void check_abi()
{
  check_addition<Abi, Checker>();
}

template <class Checker>
KOKKOS_INLINE_FUNCTION void check_abis(Kokkos::Experimental::simd_abi::abi_set<> set)
{
}

template <class Checker, class FirstAbi, class ... RestAbis>
KOKKOS_INLINE_FUNCTION void check_abis(Kokkos::Experimental::simd_abi::abi_set<FirstAbi, RestAbis...> set)
{
  check_abi<FirstAbi, Checker>();
  check_abis<Checker>(Kokkos::Experimental::simd_abi::abi_set<RestAbis...>());
}

TEST(simd, host)
{
  check_abis<gtest_checker>(Kokkos::Experimental::simd_abi::host_abi_set());
}
