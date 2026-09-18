// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_STRING_MANIPULATION_HPP
#define KOKKOS_STRING_MANIPULATION_HPP

#include <Kokkos_Macros.hpp>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <Kokkos_BitManipulation.hpp>
#include <Kokkos_Printf.hpp>

namespace Kokkos {
namespace Impl {

// This header provides a subset of the functionality from <cstring>.  In
// contrast to the standard library header, functions are usable on the device
// and in constant expressions.  It also includes functionality from <charconv>
// to convert an integer value to a character sequence.

//<editor-fold desc="String examination">
// returns the length of a given string
KOKKOS_INLINE_FUNCTION constexpr std::size_t strlen(const char *str) {
  std::size_t i = 0;
  while (str[i] != '\0') {
    ++i;
  }
  return i;
}

// compares two strings
KOKKOS_INLINE_FUNCTION constexpr int strcmp(const char *lhs, const char *rhs) {
  while (*lhs == *rhs++) {
    if (*lhs++ == '\0') {
      return 0;
    }
  }
  return static_cast<unsigned int>(*lhs) -
         static_cast<unsigned int>(*(rhs - 1));
}

// compares a certain number of characters from two strings
KOKKOS_INLINE_FUNCTION constexpr int strncmp(const char *lhs, const char *rhs,
                                             std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    if (lhs[i] != rhs[i]) {
      return lhs[i] < rhs[i] ? -1 : 1;
    } else if (lhs[i] == '\0') {
      return 0;
    }
  }
  return 0;
}
//</editor-fold>

//<editor-fold desc="String manipulation">
// copies one string to another
KOKKOS_INLINE_FUNCTION constexpr char *strcpy(char *dest, const char *src) {
  char *d = dest;
  for (; (*d = *src) != '\0'; ++d, ++src) {
  }
  return dest;
}

// copies a certain amount of characters from one string to another
KOKKOS_INLINE_FUNCTION constexpr char *strncpy(char *dest, const char *src,
                                               std::size_t count) {
  if (count != 0) {
    char *d = dest;
    do {
      if (char const c = (*d++ = *src++); c == '\0') {
        while (--count != 0) {
          *d++ = '\0';
        }
        break;
      }
    } while (--count != 0);
  }
  return dest;
}

// concatenates two strings
KOKKOS_INLINE_FUNCTION constexpr char *strcat(char *dest, const char *src) {
  char *d = dest;
  for (; *d != '\0'; ++d) {
  }
  while ((*d++ = *src++) != '\0') {
  }
  return dest;
}

// concatenates a certain amount of characters of two strings
KOKKOS_INLINE_FUNCTION constexpr char *strncat(char *dest, const char *src,
                                               std::size_t count) {
  if (count != 0) {
    char *d = dest;
    for (; *d != '\0'; ++d) {
    }
    do {
      if (char const c = (*d = *src++); c == '\0') {
        break;
      }
      d++;
    } while (--count != 0);
    *d = '\0';
  }
  return dest;
}
//</editor-fold>

//<editor-fold desc="Character conversions">
template <std::unsigned_integral Unsigned>
KOKKOS_FUNCTION constexpr unsigned int to_chars_len(Unsigned val) {
  unsigned int const base = 10;
  unsigned int n          = 1;
  while (val >= base) {
    val /= base;
    ++n;
  }
  return n;
}

template <typename FloatType>
KOKKOS_FUNCTION unsigned int to_chars_len(FloatType f) {
  static_assert(std::is_same_v<FloatType, double> ||
                std::is_same_v<FloatType, float>);

  using uint_t                         = Kokkos::equivalent_int_t<FloatType>;
  constexpr unsigned int mantissa_bits = Kokkos::mantissa_bits_v<FloatType>;
  constexpr unsigned int exponent_bits = Kokkos::exponent_bits_v<FloatType>;

  constexpr uint_t exp_mask = (uint_t(1) << exponent_bits) - uint_t(1);

  // Extract sign and exponent
  uint_t u    = Kokkos::bit_cast<uint_t>(f);
  uint_t sign = u >> (mantissa_bits + exponent_bits);
  uint_t exp  = (u >> mantissa_bits) & exp_mask;

  unsigned int ret = 0;

  ret += sign;
  if (exp == exp_mask) {
    // 'inf' and 'nan' are both 3 chars
    return ret + 3;
  }

  // Only a single format is supported (which is the default output of
  // printf("%e", d);):
  //  - 1 significant number
  //  - 1 decimal point ('.')
  //  - 6 fractional numbers
  //  - 1 exponent marker ('e')
  //  - 1 exponent sign ('+' or '-')
  //  - 2 digits, padded with 0, for the exponent if it is between -99 and 99, 3
  //  digits if it is smaller than -99 or bigger than 99.

  if constexpr (std::is_same_v<FloatType, double>) {
    // Check whether the exponent has two or three decimal digits by comparing
    // with the double that is the tipping point:
    //  - 0x2b617f7d402b1834 is the first number that rounds to 1.0000000e-99,
    //  previous number 0x2b617f7d402b1833 rounds to 9.9999999e-100
    //  - 0x54b249ad163d7d25 is the last number that rounds to 9.9999999e+99,
    //  0x54b249ad163d7d26 rounds to 1.0000000e+100
    u &= (uint_t(1) << (mantissa_bits + exponent_bits)) - uint_t(1);
#ifdef KOKKOS_ENABLE_SYCL
    // FIXME: `u != 0` doesn't have the correct value in SYCL when f == -0.
    if (f != 0 && (u < 0x2b617f7d402b1834 || u > 0x54b249ad163d7d25)) {
#else
    if ((u != 0 && u < 0x2b617f7d402b1834) || u > 0x54b249ad163d7d25) {
#endif
      return ret + 13;
    }
  }
  return ret + 12;
}

template <class Unsigned>
KOKKOS_FUNCTION constexpr void to_chars_impl(char *first, unsigned int len,
                                             Unsigned val) {
  unsigned int const base = 10;
  static_assert(std::is_integral_v<Unsigned>, "implementation bug");
  static_assert(std::is_unsigned_v<Unsigned>, "implementation bug");
  unsigned int pos = len - 1;
  while (val > 0) {
    auto const num = val % base;
    val /= base;
    first[pos] = '0' + num;
    --pos;
  }
}

// define values of portable error conditions that correspond to the POSIX error
// codes
enum class errc {
  ok              = 0,
  value_too_large = 75  // equivalent POSIX error is EOVERFLOW
};
struct to_chars_result {
  char *ptr;
  errc ec;
};

// converts an integer value to a character sequence
template <class Integral>
KOKKOS_FUNCTION constexpr to_chars_result to_chars_i(char *first, char *last,
                                                     Integral value) {
  using Unsigned = std::conditional_t<sizeof(Integral) <= sizeof(unsigned int),
                                      unsigned int, unsigned long long>;
  Unsigned unsigned_val = value;  // NOLINT(bugprone-signed-char-misuse)
  if (value == 0) {
    *first = '0';
    return {first + 1, {}};
  } else if constexpr (std::is_signed_v<Integral>) {
    if (value < 0) {
      *first++     = '-';
      unsigned_val = Unsigned(~value) + Unsigned(1);
    }
  }
  std::ptrdiff_t const len = to_chars_len(unsigned_val);
  if (last - first < len) {
    return {last, errc::value_too_large};
  }
  to_chars_impl(first, len, unsigned_val);
  return {first + len, {}};
}
//</editor-fold>

/*
 * Decimal representation of a floating point number
 */
template <typename FloatType, std::size_t precision>
class DecimalRepresentation {
 public:
  using uint_t = Kokkos::equivalent_int_t<FloatType>;

  // Buffer that contains the decimal representation of a number in scientific
  // notation (without decimal separator)
  uint8_t buffer[precision];
  // Exponent of the decimal representation stored
  int exp10;
  // For instance, if the number stored is 2^10, `buffer` contains
  // "102400000..." and `exp10` is 3

 protected:
  // frac is the fractional part of the floating point number that is beyond
  // the decimal representation. It is always included between 0 and 1
  // (excluded).
  // For instance, if precision is 3 and the number stored is 2^-6 = 0.015625:
  //  - exp10 = -2
  //  - buffer = 156
  //  - frac = 0.25
  double frac;

 public:
  // Initialize with decimal representation of 0.
  KOKKOS_FUNCTION constexpr DecimalRepresentation()
      : buffer{0}, exp10{0}, frac{0} {}

  // Print buffer for debugging purpose
  KOKKOS_FUNCTION void print() const {
    char tmp[precision + 2];
    tmp[0] = buffer[0] + '0';
    tmp[1] = '.';
    for (std::size_t i = 2; i < precision + 1; ++i) {
      tmp[i] = buffer[i - 1] + '0';
    }
    tmp[precision + 1] = '\0';
    Kokkos::printf("%se%+03i (%e)", tmp, exp10, frac);
  }

 protected:
  // This can't be higher than 60 since we need to be able to store up to
  // (2^max_div - 1) * 9 in the remainder/carry: bigger exponent would overflow
  // an uint64_t
  static constexpr uint_t max_div = 60;
  static constexpr uint_t max_mul = 60;

  // Shift the decimal representation in buffer `shift` time to the right,
  // inserting `insert` as the leading numbers.
  KOKKOS_FUNCTION void shift_right(int shift, uint8_t insert) {
    for (int i = precision - 1; i >= 0; --i) {
      // Add discarded digit to the remainder if the curent cell is shifted out
      if (std::size_t(i) + std::size_t(shift) >= precision) {
        frac = (buffer[i] + frac) / 10.;
      }

      if (i >= shift) {
        buffer[i] = buffer[i - shift];
      } else {
        buffer[i] = insert;
      }
    }

    if (std::size_t(shift) > precision) {
      for (int i = precision; i < shift; ++i) {
        frac = (insert + frac) / 10.;
      }
    }

    exp10 += shift;
  }

  // Divide decimal number by 2^exp
  // Returns garbage if exp > max_div
  KOKKOS_FUNCTION void divide_by_power_of_two(uint_t exp) {
    uint64_t dividend   = 0x0llu;
    uint64_t divisor    = 0x1llu << exp;
    std::size_t src_idx = 0;
    std::size_t res_idx = 0;

    do {
      // Update dividend with the next digit (coming either from the decimal
      // buffer or from `frac`)
      dividend *= 10;
      if (src_idx < precision) {
        dividend += buffer[src_idx++];
      } else {
        frac *= 10.;
        dividend += (uint64_t)frac;
        frac -= (uint64_t)frac;
      }

      if (dividend < divisor && res_idx == 0) {
        // As long as we haven't found a non-zero digit, we don't output 0: we
        // only track the exponent of the future most significant digit
        --exp10;
      } else {
        // Otherwise, buffer receives the result of the division and dividend
        // the remainder
        buffer[res_idx++] = dividend >> exp;
        // divisor is a power of two: use optimisation x % 2^n == x & (2^n - 1)
        dividend &= divisor - 1;
      }
    } while (res_idx < precision);

    frac = ((double)dividend + frac) / (double)divisor;
  }

  // Multiply decimal number by 2^exp
  // Returns garbage if exp > max_mul
  KOKKOS_FUNCTION void multiply_by_power_of_two(uint_t exp) {
    const uint64_t multiplier = 0x1llu << exp;

    frac *= multiplier;
    uint64_t carry = frac;
    frac -= carry;

    // Multiply each digit of the input by 2^n
    for (int i = precision - 1; i >= 0; --i) {
      uint64_t tmp = (uint64_t)(buffer[i]) * multiplier + carry;
      buffer[i]    = tmp % 10;
      carry        = tmp / 10;
    }

    // Continue adding the carry to the computed number
    while (carry > 0) {
      shift_right(1, carry % 10);  // shift_right updates the exponent
      carry /= 10;
    }
  }

 public:
  // Rounds the number up or down according to the fractional part contained in
  // `frac`, respecting the "round to nearest, ties to even" convention
  KOKKOS_FUNCTION void round() {
    int round_up = false;
    if (frac >= .5) {
      if (frac == .5) {
        // In case `frac` is exactly 0.5, we round up if `buffer[6]` is an odd
        // number, down otherwise.
        // This is the `tie to even` part of the rounding.
        round_up = (buffer[precision - 1] % 2) == 1;
      } else {
        round_up = true;
      }
    }
    frac = 0.;

    if (!round_up) {
      return;
    }

    // Add 1 to the decimal representation (the carry can propagate)
    for (int i = precision - 1; i >= 0; --i) {
      if (++buffer[i] > 9) {
        buffer[i] = 0;
      } else {
        return;
      }
    }

    // If we reach this point, initial number had the form "999....999": rounds
    // to "100....000" with `exp10`+1
    shift_right(1, 1);
  }

  KOKKOS_FUNCTION DecimalRepresentation<FloatType, precision> &operator+=(
      DecimalRepresentation<FloatType, precision> rhs) {
    // Shift operands to ensure both have the same exponent
    if (rhs.exp10 < exp10) {
      rhs.shift_right(exp10 - rhs.exp10, 0);
    } else if (exp10 < rhs.exp10) {
      shift_right(rhs.exp10 - exp10, 0);
    }

    // frac < 1, thus carry can be 0 or 1 only
    frac += rhs.frac;
    int carry = frac;
    frac -= carry;

    for (int i = precision - 1; i >= 0; --i) {
      // buffer[i] <= 9, rhs.buffer[i] <= 9, carry <= 1: the sum can never be
      // bigger than 19, the new carry can't be bigger than 1.
      buffer[i] += rhs.buffer[i] + carry;
      if (buffer[i] > 9) {
        carry = 1;
        buffer[i] -= 10;
      } else {
        carry = 0;
      }
    }

    if (carry) {
      shift_right(1, 1);
    }

    return *this;
  }
};

/**
 * Decimal representation of a power of two
 */
template <class FloatType, std::size_t precision>
class BaseTwoExponent : public DecimalRepresentation<FloatType, precision> {
 protected:
  using uint_t  = Kokkos::equivalent_int_t<FloatType>;
  using decimal = DecimalRepresentation<FloatType, precision>;
  using decimal::buffer;
  using decimal::exp10;
  using decimal::max_div;
  using decimal::max_mul;

  // Exponent bias (different from the IEEE754 one, to take subnormals into
  // account)
  static constexpr uint_t bias =
      (uint_t(1) << (Kokkos::exponent_bits_v<FloatType> - 1)) +
      Kokkos::mantissa_bits_v<FloatType> - 2;

  // Exponent of the power stored in `buffer`
  // Stored with the total bias added
  uint_t exp2;

 public:
  // Initialize with the decimal representation of 2^0 = 1
  KOKKOS_FUNCTION constexpr BaseTwoExponent() {
    decimal::buffer[0] = 1;
    exp2               = bias;
  }

  KOKKOS_FUNCTION void print() const {
    decimal::print();
    Kokkos::printf(" = 2^%i", (int)exp2 - (int)bias);
  }

  // Generate the decimal representation of 2 ^ exp2_target, using the current
  // representation as the basis for the computation
  KOKKOS_FUNCTION void generate_nth_exp(uint_t exp2_target) {
    if (exp2_target < exp2) {
      // Divide max_div by max_div until we can reach the target with one
      // smaller step
      while (exp2 > exp2_target + max_div) {
        decimal::divide_by_power_of_two(max_div);
        exp2 -= max_div;
      }

      // Last division needed to reach the target
      if (exp2 != exp2_target) {
        decimal::divide_by_power_of_two(exp2 - exp2_target);
        exp2 = exp2_target;
      }
    } else if (exp2_target > exp2) {
      // Multiply max_mul by max_mul until we can reach the target with one
      // smaller step
      while (exp2 + max_mul < exp2_target) {
        decimal::multiply_by_power_of_two(max_mul);
        exp2 += max_mul;
      }

      // Last multiplication needed to reach the target
      if (exp2 != exp2_target) {
        decimal::multiply_by_power_of_two(exp2_target - exp2);
        exp2 = exp2_target;
      }
    }
  }
};

template <typename FloatType>
KOKKOS_FUNCTION to_chars_result to_chars_f(char *first, char *last,
                                           FloatType f) {
  static_assert(std::is_same_v<FloatType, double> ||
                std::is_same_v<FloatType, float>);

  using uint_t                         = Kokkos::equivalent_int_t<FloatType>;
  constexpr unsigned int mantissa_bits = Kokkos::mantissa_bits_v<FloatType>;
  constexpr unsigned int exponent_bits = Kokkos::exponent_bits_v<FloatType>;

  constexpr uint_t exp_mask      = (uint_t(1) << exponent_bits) - 1;
  constexpr uint_t mantissa_mask = (uint_t(1) << mantissa_bits) - 1;

  // Number of decimal digits outputed
  constexpr std::size_t precision = 7;

  // Check that we have enough space to output the result
  std::ptrdiff_t const len = to_chars_len(f);
  if (last - first < len) {
    return {last, errc::value_too_large};
  }

  // Extract sign, exponent and mantissa
  uint_t u        = Kokkos::bit_cast<uint_t>(f);
  uint_t sign     = u >> (mantissa_bits + exponent_bits);
  uint_t exp      = (u >> mantissa_bits) & exp_mask;
  uint_t mantissa = u & mantissa_mask;

  char *out = first;

  // Add sign to output if needed
  if (sign) {
    *out++ = '-';
  }

  // Inf and NaN
  if (exp == exp_mask) {
    if (mantissa) {
      Kokkos::Impl::strcpy(out, "nan");
    } else {
      Kokkos::Impl::strcpy(out, "inf");
    }
    return {first + len, {}};
  }

  // Normalize number if needed (subnormals don't need normalization)
  if (exp != 0) {
    // Apply implicit leading 1
    mantissa |= (uint_t(1) << mantissa_bits);
    // Take implicit 1 into account for the exponent
    --exp;
  }

  // Decimal representation of the powers of two that will be added together to
  // compute the result
  BaseTwoExponent<FloatType, precision> base;

  // Decimal representation of the result
  DecimalRepresentation<FloatType, precision> decimal;

  bool found_set_bit = false;
  // Loop over each bit of the mantissa, add the corresponding power of 2 if
  // the bit is set
  while (mantissa) {
    if (mantissa & uint_t(1)) {
      base.generate_nth_exp(exp);

      if (!found_set_bit) {
        decimal       = base;
        found_set_bit = true;
      } else {
        // Add current power of two with number
        decimal += base;
      }
    }

    mantissa >>= 1u;
    ++exp;
  }

  decimal.round();

  // Copy the digits to the output, with decimal separator
  *out++ = decimal.buffer[0] + '0';
  *out++ = '.';
  for (std::size_t i = 1; i < precision; ++i) {
    *out++ = decimal.buffer[i] + '0';
  }

  // Write exponent
  *out++ = 'e';
  int abs_exp;
  if (decimal.exp10 >= 0) {
    *out++  = '+';
    abs_exp = decimal.exp10;
  } else {
    *out++  = '-';
    abs_exp = -decimal.exp10;
  }
  if (abs_exp <= 9) {
    // Pad with 0 if exp is only 1 digit long
    *out++ = '0';
  }
  return {to_chars_i(out, last, abs_exp).ptr, {}};
}

}  // namespace Impl
}  // namespace Kokkos

#endif
