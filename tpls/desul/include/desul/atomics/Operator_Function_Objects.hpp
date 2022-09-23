/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_OPERATOR_FUNCTION_OBJECTS_HPP_
#define DESUL_ATOMICS_OPERATOR_FUNCTION_OBJECTS_HPP_

#include <desul/atomics/Macros.hpp>
#include <type_traits>

// Function objects that represent common arithmetic and logical
// Combination operands to be used in a compare-and-exchange based atomic operation
namespace desul {
namespace Impl {

template <class Scalar1, class Scalar2>
struct MaxOper {
  DESUL_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) {
    return (val1 > val2 ? val1 : val2);
  }
  DESUL_FORCEINLINE_FUNCTION
  static constexpr bool check_early_exit(Scalar1 const& val1, Scalar2 const& val2) {
    return val1 > val2;
  }
};

template <class Scalar1, class Scalar2>
struct MinOper {
  DESUL_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) {
    return (val1 < val2 ? val1 : val2);
  }
  DESUL_FORCEINLINE_FUNCTION
  static constexpr bool check_early_exit(Scalar1 const& val1, Scalar2 const& val2) {
    return val1 < val2;
  }
};

template <class Op, class Scalar1, class Scalar2, class = bool>
struct may_exit_early : std::false_type {};

// This exit early optimization causes weird compiler errors with MSVC 2019
#ifndef DESUL_HAVE_MSVC_ATOMICS
template <class Op, class Scalar1, class Scalar2>
struct may_exit_early<Op,
                      Scalar1,
                      Scalar2,
                      decltype(Op::check_early_exit(std::declval<Scalar1 const&>(),
                                                    std::declval<Scalar2 const&>()))>
    : std::true_type {};
#endif

template <class Op, class Scalar1, class Scalar2>
constexpr DESUL_FUNCTION
    std::enable_if_t<may_exit_early<Op, Scalar1, Scalar2>::value, bool>
    check_early_exit(Op const&, Scalar1 const& val1, Scalar2 const& val2) {
  return Op::check_early_exit(val1, val2);
}

template <class Op, class Scalar1, class Scalar2>
constexpr DESUL_FUNCTION
    std::enable_if_t<!may_exit_early<Op, Scalar1, Scalar2>::value, bool>
    check_early_exit(Op const&, Scalar1 const&, Scalar2 const&) {
  return false;
}

template <class Scalar1, class Scalar2>
struct AddOper {
  DESUL_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) { return val1 + val2; }
};

template <class Scalar1, class Scalar2>
struct SubOper {
  DESUL_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) { return val1 - val2; }
};

template <class Scalar1, class Scalar2>
struct MulOper {
  DESUL_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) { return val1 * val2; }
};

template <class Scalar1, class Scalar2>
struct DivOper {
  DESUL_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) { return val1 / val2; }
};

template <class Scalar1, class Scalar2>
struct ModOper {
  DESUL_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) { return val1 % val2; }
};

template <class Scalar1, class Scalar2>
struct AndOper {
  DESUL_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) { return val1 & val2; }
};

template <class Scalar1, class Scalar2>
struct OrOper {
  DESUL_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) { return val1 | val2; }
};

template <class Scalar1, class Scalar2>
struct XorOper {
  DESUL_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) { return val1 ^ val2; }
};

template <class Scalar1, class Scalar2>
struct NandOper {
  DESUL_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) {
    return ~(val1 & val2);
  }
};

template <class Scalar1, class Scalar2>
struct LShiftOper {
  DESUL_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) {
    return val1 << val2;
  }
};

template <class Scalar1, class Scalar2>
struct RShiftOper {
  DESUL_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) {
    return val1 >> val2;
  }
};

template <class Scalar1, class Scalar2>
struct IncModOper {
  DESUL_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) {
    return ((val1 >= val2) ? Scalar1(0) : val1 + Scalar1(1));
  }
};

template <class Scalar1, class Scalar2>
struct DecModOper {
  DESUL_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2& val2) {
    return (((val1 == Scalar1(0)) | (val1 > val2)) ? val2 : (val1 - Scalar1(1)));
  }
};

template <class Scalar1, class Scalar2>
struct StoreOper {
  DESUL_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1&, const Scalar2& val2) { return val2; }
};

template <class Scalar1, class Scalar2>
struct LoadOper {
  DESUL_FORCEINLINE_FUNCTION
  static Scalar1 apply(const Scalar1& val1, const Scalar2&) { return val1; }
};

}  // namespace Impl
}  // namespace desul

#endif
