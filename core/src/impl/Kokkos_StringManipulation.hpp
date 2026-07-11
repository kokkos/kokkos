// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_STRING_MANIPULATION_HPP
#define KOKKOS_STRING_MANIPULATION_HPP

#include <Kokkos_Macros.hpp>
#include <impl/Kokkos_Half_FloatingPointWrapper.hpp>
#include <concepts>
#include <cstddef>
#include <ostream>
#include <type_traits>

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

template <std::signed_integral Signed>
KOKKOS_FUNCTION constexpr unsigned int to_chars_len(Signed val) {
  using Unsigned = std::conditional_t<sizeof(Signed) <= sizeof(unsigned int),
                                      unsigned int, unsigned long long>;
  Unsigned unsigned_val;
  int n = 0;

  if (val < 0) {
    unsigned_val = Unsigned(~val) + Unsigned(1);
    ++n;
  } else {
    unsigned_val = val;  // NOLINT(bugprone-signed-char-misuse)
  }

  return n + to_chars_len(unsigned_val);
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
enum class errc : int {
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
  // NOLINTBEGIN(bugprone-invalid-enum-default-initialization)
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
  // NOLINTEND(bugprone-invalid-enum-default-initialization)
}

//</editor-fold>

template <std::size_t capacity>
class StaticString {
  // Need room to guarantee we can at least print the trunc pattern
  static_assert(capacity >= 4, "capacity too low for StaticString");

  char m_data[capacity];
  // String can be full even with size < capacity if we tried to add something
  // that was too big
  bool m_is_full;
  std::size_t m_size;

  /*
   * This function is called when one's try to add more characters than there
   * is room for in a string.
   * It adds '...\0' at the end of the string.
   */
  KOKKOS_FUNCTION constexpr void trunc() {
    m_is_full = true;

    std::size_t start_pattern_idx;

    if (m_size < capacity - 4) {
      start_pattern_idx = m_size;
    } else {
      start_pattern_idx = capacity - 4;
    }

    m_data[start_pattern_idx]     = '.';
    m_data[start_pattern_idx + 1] = '.';
    m_data[start_pattern_idx + 2] = '.';
    m_data[start_pattern_idx + 3] = '\0';
  }

 public:
  // Constructors
  KOKKOS_FUNCTION constexpr StaticString() : m_is_full(false), m_size(0) {
    m_data[0] = '\0';
  }

  KOKKOS_FUNCTION constexpr StaticString(const char *init)
      : m_is_full(false), m_size(0) {
    m_data[0] = '\0';
    *this << (init);
  }

  // Accessors
  KOKKOS_FUNCTION constexpr const char *c_str() const { return m_data; }

  KOKKOS_FUNCTION constexpr std::size_t size() const { return m_size; }

  KOKKOS_FUNCTION constexpr bool is_full() const { return m_is_full; }

  // Append cstring
  KOKKOS_FUNCTION constexpr StaticString<capacity> &operator<<(
      const char *string) {
    if (m_is_full) {
      return *this;
    }

    while (*string != '\0' && m_size < capacity) {
      m_data[m_size++] = *string++;
    }

    if (m_size >= capacity) {
      trunc();
    } else {
      m_data[m_size] = '\0';
    }

    return *this;
  }

  // Append int value
  template <std::integral Integer>
  KOKKOS_FUNCTION constexpr StaticString<capacity> &operator<<(Integer val) {
    if (m_is_full) {
      return *this;
    }

    // +1 because we need room for the '\0'
    std::size_t needed_size = to_chars_len(val) + 1;
    // Comparison written this way to avoid risk of overflowing size_t
    if ((needed_size > capacity) || (capacity - needed_size < m_size)) {
      trunc();
    } else {
      to_chars_i(m_data + m_size, m_data + capacity, val);
      m_size += needed_size - 1;
      m_data[m_size] = '\0';
    }

    return *this;
  }

  // Append float value
  template <std::floating_point FloatType>
  KOKKOS_FUNCTION constexpr StaticString<capacity> &operator<<(FloatType val) {
    /*
    if (m_is_full) {
      return *this;
    }

    // +1 because we need room for the '\0'
    int needed_size = to_chars_len(val) + 1;
    // Comparison written this way to avoid risk of overflowing size_t
    if ((needed_size > capacity) || (capacity - needed_size < m_size)) {
      trunc();
    } else {
      to_chars_f(m_data + m_size, m_data + capacity, val);
      m_size += needed_size - 1;
      m_data[m_size] = '\0';
    }

    return *this;
    */

    // TODO: merge #9244
    *this << (long long int)val;
    return *this;
  }

#if defined(KOKKOS_HALF_T_IS_FLOAT) && !KOKKOS_HALF_T_IS_FLOAT
  // Append half value
  KOKKOS_FUNCTION constexpr StaticString<capacity> &operator<<(
      Kokkos::Experimental::half_t val) {
    *this << cast_from_half<float>(val);
    return *this;
  }
#endif

#if defined(KOKKOS_BHALF_T_IS_FLOAT) && !KOKKOS_BHALF_T_IS_FLOAT
  // Append bhalf value
  KOKKOS_FUNCTION constexpr StaticString<capacity> &operator<<(
      Kokkos::Experimental::bhalf_t val) {
    *this << cast_from_bhalf<float>(val);
    return *this;
  }
#endif

  // Append boolean value
  KOKKOS_FUNCTION constexpr StaticString<capacity> &operator<<(bool val) {
    if (m_is_full) {
      return *this;
    }

    if (val) {
      *this << "True";
    } else {
      *this << "False";
    }

    return *this;
  }

  friend std::ostream &operator<<(std::ostream &os,
                                  const StaticString<capacity> &ss) {
    os << ss.c_str();
    return os;
  }
};

}  // namespace Impl
}  // namespace Kokkos

#endif
