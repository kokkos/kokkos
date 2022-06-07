#ifndef KOKKOS_SIMD_SCALAR_HPP
#define KOKKOS_SIMD_SCALAR_HPP

#include <type_traits>

#include <Kokkos_Simd_Common.hpp>

namespace Kokkos {
namespace Experimental {

namespace simd_abi {

class scalar {};

}

template <class T>
class simd_mask<T, simd_abi::scalar> {
  bool m_value;
 public:
  using value_type = bool;
  using simd_type = simd<T, simd_abi::scalar>;
  using abi_type = simd_abi::scalar;
  using reference = value_type&;
  KOKKOS_DEFAULTED_FUNCTION simd_mask() = default;
  KOKKOS_FORCEINLINE_FUNCTION static constexpr std::size_t size() { return 1; }
  KOKKOS_FORCEINLINE_FUNCTION explicit simd_mask(value_type value)
    :m_value(value)
  {}
  template <class U>
  KOKKOS_FORCEINLINE_FUNCTION
  simd_mask(simd_mask<U, simd_abi::scalar> const& other)
    :m_value(static_cast<U>(other))
  {
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit operator bool () const { return m_value; }
  KOKKOS_FORCEINLINE_FUNCTION reference operator[](std::size_t)
  {
    return m_value;
  }
  KOKKOS_FORCEINLINE_FUNCTION value_type operator[](std::size_t) const
  {
    return m_value;
  }
  KOKKOS_FORCEINLINE_FUNCTION simd_mask operator||(simd_mask const& other) const {
    return simd_mask(m_value || other.m_value);
  }
  KOKKOS_FORCEINLINE_FUNCTION simd_mask operator&&(simd_mask const& other) const {
    return simd_mask(m_value && other.m_value);
  }
  KOKKOS_FORCEINLINE_FUNCTION simd_mask operator!() const {
    return simd_mask(!m_value);
  }
  KOKKOS_FORCEINLINE_FUNCTION bool operator==(simd_mask const& other) const {
    return m_value == other.m_value;
  }
};

template <class T>
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION
bool none_of(simd_mask<T, simd_abi::scalar> const& mask)
{
  return !static_cast<bool>(mask);
}

template <class T>
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION
bool all_of(simd_mask<T, simd_abi::scalar> const& mask)
{
  return static_cast<bool>(mask);
}

template <class T>
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION
bool any_of(simd_mask<T, simd_abi::scalar> const& mask)
{
  return static_cast<bool>(mask);
}

template <class T>
class simd<T, simd_abi::scalar> {
  T m_value;
 public:
  using value_type = T;
  using abi_type = simd_abi::scalar;
  using mask_type = simd_mask<T, abi_type>;
  using reference = value_type&;
  KOKKOS_DEFAULTED_FUNCTION simd() = default;
  KOKKOS_DEFAULTED_FUNCTION simd(simd const&) = default;
  KOKKOS_DEFAULTED_FUNCTION simd(simd&&) = default;
  KOKKOS_DEFAULTED_FUNCTION simd& operator=(simd const&) = default;
  KOKKOS_DEFAULTED_FUNCTION simd& operator=(simd&&) = default;
  KOKKOS_FORCEINLINE_FUNCTION static constexpr std::size_t size() { return 1; }
  KOKKOS_FORCEINLINE_FUNCTION simd(value_type value)
    :m_value(value)
  {}
  template <class U>
  KOKKOS_FORCEINLINE_FUNCTION explicit simd(simd<U, abi_type> const& other)
    :m_value(static_cast<U>(other))
  {}
  template <class G,
    typename std::enable_if<
      // basically, can you do { value_type r = gen(std::integral_constant<std::size_t, i>()); }
      std::is_invocable_r_v<value_type, G, std::integral_constant<std::size_t, 0>>,
      bool>::type = false>
  KOKKOS_FORCEINLINE_FUNCTION simd(G&& gen)
    :m_value(gen(std::integral_constant<std::size_t, 0>()))
  {
  }
  KOKKOS_FORCEINLINE_FUNCTION simd operator*(simd const& other) const {
    return simd(m_value * other.m_value);
  }
  KOKKOS_FORCEINLINE_FUNCTION simd operator/(simd const& other) const {
    return simd(m_value / other.m_value);
  }
  KOKKOS_FORCEINLINE_FUNCTION simd operator+(simd const& other) const {
    return simd(m_value + other.m_value);
  }
  KOKKOS_FORCEINLINE_FUNCTION simd operator-(simd const& other) const {
    return simd(m_value - other.m_value);
  }
  KOKKOS_FORCEINLINE_FUNCTION simd operator-() const {
    return simd(-m_value);
  }
  KOKKOS_FORCEINLINE_FUNCTION simd operator>>(int rhs) const {
    return simd(m_value >> rhs);
  }
  KOKKOS_FORCEINLINE_FUNCTION simd operator>>(simd<int, abi_type> const& rhs) const {
    return simd(m_value >> static_cast<int>(rhs));
  }
  KOKKOS_FORCEINLINE_FUNCTION simd operator<<(int rhs) const {
    return simd(m_value << rhs);
  }
  KOKKOS_FORCEINLINE_FUNCTION simd operator<<(simd<int, abi_type> const& rhs) const {
    return simd(m_value << static_cast<int>(rhs));
  }
  KOKKOS_FORCEINLINE_FUNCTION simd operator&(simd const& other) const {
    return m_value & other.m_value;
  }
  KOKKOS_FORCEINLINE_FUNCTION simd operator|(simd const& other) const {
    return m_value | other.m_value;
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit operator T () const { return m_value; }
  KOKKOS_FORCEINLINE_FUNCTION mask_type operator<(simd const& other) const {
    return mask_type(m_value < other.m_value);
  }
  KOKKOS_FORCEINLINE_FUNCTION mask_type operator>(simd const& other) const {
    return mask_type(m_value > other.m_value);
  }
  KOKKOS_FORCEINLINE_FUNCTION mask_type operator<=(simd const& other) const {
    return mask_type(m_value <= other.m_value);
  }
  KOKKOS_FORCEINLINE_FUNCTION mask_type operator>=(simd const& other) const {
    return mask_type(m_value >= other.m_value);
  }
  KOKKOS_FORCEINLINE_FUNCTION mask_type operator==(simd const& other) const {
    return mask_type(m_value == other.m_value);
  }
  KOKKOS_FORCEINLINE_FUNCTION mask_type operator!=(simd const& other) const {
    return mask_type(m_value != other.m_value);
  }
  KOKKOS_FORCEINLINE_FUNCTION void copy_from(T const* ptr, element_aligned_tag) {
    m_value = *ptr;
  }
  KOKKOS_FORCEINLINE_FUNCTION void copy_to(T* ptr, element_aligned_tag) const {
    *ptr = m_value;
  }
  KOKKOS_FORCEINLINE_FUNCTION reference operator[](std::size_t)
  {
    return m_value;
  }
  KOKKOS_FORCEINLINE_FUNCTION value_type operator[](std::size_t) const
  {
    return m_value;
  }
};

template <class T>
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION
simd<T, simd_abi::scalar> abs(simd<T, simd_abi::scalar> const& a)
{
  return simd<T, simd_abi::scalar>(std::abs(static_cast<T>(a)));
}

template <class T>
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION
simd<T, simd_abi::scalar> sqrt(simd<T, simd_abi::scalar> const& a)
{
  return simd<T, simd_abi::scalar>(std::sqrt(static_cast<T>(a)));
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION
simd<T, simd_abi::scalar> fma(
    simd<T, simd_abi::scalar> const& x,
    simd<T, simd_abi::scalar> const& y,
    simd<T, simd_abi::scalar> const& z)
{
  return simd<T, simd_abi::scalar>((static_cast<T>(x) * static_cast<T>(y)) + static_cast<T>(z));
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION
simd<T, simd_abi::scalar> condition(
    desul::Impl::dont_deduce_this_parameter_t<simd_mask<T, simd_abi::scalar>> const& a,
    simd<T, simd_abi::scalar> const& b,
    simd<T, simd_abi::scalar> const& c)
{
  return simd<T, simd_abi::scalar>(static_cast<bool>(a) ? static_cast<T>(b) : static_cast<T>(c));
}

template <class T, class Abi>
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION
simd<T, Abi> copysign(simd<T, Abi> const& a, simd<T, Abi> const& b) {
  return std::copysign(static_cast<T>(a), static_cast<T>(b));
}

template <class T>
class const_where_expression<simd_mask<T, simd_abi::scalar>, simd<T, simd_abi::scalar>> {
 public:
  using abi_type = simd_abi::scalar;
  using value_type = simd<T, abi_type>;
  using mask_type = simd_mask<T, abi_type>;
 protected:
  value_type& m_value;
  mask_type const& m_mask;
 public:
  KOKKOS_FORCEINLINE_FUNCTION
  const_where_expression(mask_type const& mask_arg, value_type const& value_arg)
    :m_value(const_cast<value_type&>(value_arg))
    ,m_mask(mask_arg)
  {}
  KOKKOS_FORCEINLINE_FUNCTION
  mask_type const& mask() const { return m_mask; }
  KOKKOS_FORCEINLINE_FUNCTION
  value_type const& value() const { return m_value; }
  KOKKOS_FORCEINLINE_FUNCTION
  void copy_to(T* mem, element_aligned_tag) const {
    if (static_cast<bool>(m_mask)) *mem = static_cast<T>(m_value);
  }
  template <class Integral>
  KOKKOS_FORCEINLINE_FUNCTION
  std::enable_if_t<std::is_integral_v<Integral>>
  scatter_to(T* mem, simd<Integral, simd_abi::scalar> const& index) const {
    if (static_cast<bool>(m_mask)) mem[static_cast<Integral>(index)] = static_cast<T>(m_value);
  }
};

template <class T>
class where_expression<simd_mask<T, simd_abi::scalar>, simd<T, simd_abi::scalar>>
 : public const_where_expression<simd_mask<T, simd_abi::scalar>, simd<T, simd_abi::scalar>> {
  using base_type = const_where_expression<simd_mask<T, simd_abi::scalar>, simd<T, simd_abi::scalar>>;
 public:
  using typename base_type::value_type;
  KOKKOS_FORCEINLINE_FUNCTION
  where_expression(simd_mask<T, simd_abi::scalar> const& mask_arg, simd<T, simd_abi::scalar>& value_arg)
    :base_type(mask_arg, value_arg)
  {}
  KOKKOS_FORCEINLINE_FUNCTION
  void copy_from(T const* mem, element_aligned_tag)
  {
    if (static_cast<bool>(this->m_mask)) this->m_value = *mem;
  }
  template <class Integral>
  KOKKOS_FORCEINLINE_FUNCTION
  std::enable_if_t<std::is_integral_v<Integral>>
  gather_from(T const* mem, simd<Integral, simd_abi::scalar> const& index)
  {
    if (static_cast<bool>(this->m_mask)) this->m_value = mem[static_cast<Integral>(index)];
  }
  template <class U>
  KOKKOS_FORCEINLINE_FUNCTION
  void operator=(U const& other)
  {
    if (static_cast<bool>(this->m_mask)) this->m_value = static_cast<simd<T, simd_abi::scalar>>(other);
  }
};

template <class T, class BinaryOp>
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION
T reduce(
    const_where_expression<simd_mask<T, simd_abi::scalar>, simd<T, simd_abi::scalar>> const& x,
    T identity_element,
    BinaryOp)
{
  return static_cast<bool>(x.mask()) ? static_cast<T>(x.value()) : identity_element;
}

}
}

#endif
