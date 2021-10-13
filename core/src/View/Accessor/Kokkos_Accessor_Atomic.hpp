#include<Kokkos_Macros.hpp>

#include <Kokkos_Atomic.hpp>

namespace Kokkos {
namespace Impl {

// The following tag is used to prevent an implicit call of the constructor when
// trying to assign a literal 0 int ( = 0 );
struct AtomicViewConstTag {};

// TODO replace this with atomic_ref at some point
template <class ElementType>
class AtomicReference {
 public:
  using value_type           = ElementType;
  using const_value_type     = std::add_const_t<value_type>;
  using non_const_value_type = std::remove_const_t<value_type>;
  volatile value_type* const ptr;

  KOKKOS_INLINE_FUNCTION
  AtomicReference(value_type* ptr_, AtomicViewConstTag) : ptr(ptr_) {}

  KOKKOS_INLINE_FUNCTION
  const_value_type operator=(const_value_type& val) const {
    *ptr = val;
    return val;
  }
  KOKKOS_INLINE_FUNCTION
  const_value_type operator=(volatile const_value_type& val) const {
    *ptr = val;
    return val;
  }

  KOKKOS_INLINE_FUNCTION
  void inc() const { Kokkos::atomic_increment(ptr); }

  KOKKOS_INLINE_FUNCTION
  void dec() const { Kokkos::atomic_decrement(ptr); }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator++() const {
    const_value_type tmp =
        Kokkos::atomic_fetch_add(ptr, non_const_value_type(1));
    return tmp + 1;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator--() const {
    const_value_type tmp =
        Kokkos::atomic_fetch_sub(ptr, non_const_value_type(1));
    return tmp - 1;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator++(int) const {
    return Kokkos::atomic_fetch_add(ptr, non_const_value_type(1));
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator--(int) const {
    return Kokkos::atomic_fetch_sub(ptr, non_const_value_type(1));
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator+=(const_value_type& val) const {
    const_value_type tmp = Kokkos::atomic_fetch_add(ptr, val);
    return tmp + val;
  }
  KOKKOS_INLINE_FUNCTION
  const_value_type operator+=(volatile const_value_type& val) const {
    const_value_type tmp = Kokkos::atomic_fetch_add(ptr, val);
    return tmp + val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator-=(const_value_type& val) const {
    const_value_type tmp = Kokkos::atomic_fetch_sub(ptr, val);
    return tmp - val;
  }
  KOKKOS_INLINE_FUNCTION
  const_value_type operator-=(volatile const_value_type& val) const {
    const_value_type tmp = Kokkos::atomic_fetch_sub(ptr, val);
    return tmp - val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator*=(const_value_type& val) const {
    return Kokkos::atomic_mul_fetch(ptr, val);
  }
  KOKKOS_INLINE_FUNCTION
  const_value_type operator*=(volatile const_value_type& val) const {
    return Kokkos::atomic_mul_fetch(ptr, val);
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator/=(const_value_type& val) const {
    return Kokkos::atomic_div_fetch(ptr, val);
  }
  KOKKOS_INLINE_FUNCTION
  const_value_type operator/=(volatile const_value_type& val) const {
    return Kokkos::atomic_div_fetch(ptr, val);
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator%=(const_value_type& val) const {
    return Kokkos::atomic_mod_fetch(ptr, val);
  }
  KOKKOS_INLINE_FUNCTION
  const_value_type operator%=(volatile const_value_type& val) const {
    return Kokkos::atomic_mod_fetch(ptr, val);
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator&=(const_value_type& val) const {
    return Kokkos::atomic_and_fetch(ptr, val);
  }
  KOKKOS_INLINE_FUNCTION
  const_value_type operator&=(volatile const_value_type& val) const {
    return Kokkos::atomic_and_fetch(ptr, val);
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator^=(const_value_type& val) const {
    return Kokkos::atomic_xor_fetch(ptr, val);
  }
  KOKKOS_INLINE_FUNCTION
  const_value_type operator^=(volatile const_value_type& val) const {
    return Kokkos::atomic_xor_fetch(ptr, val);
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator|=(const_value_type& val) const {
    return Kokkos::atomic_or_fetch(ptr, val);
  }
  KOKKOS_INLINE_FUNCTION
  const_value_type operator|=(volatile const_value_type& val) const {
    return Kokkos::atomic_or_fetch(ptr, val);
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator<<=(const_value_type& val) const {
    return Kokkos::atomic_lshift_fetch(ptr, val);
  }
  KOKKOS_INLINE_FUNCTION
  const_value_type operator<<=(volatile const_value_type& val) const {
    return Kokkos::atomic_lshift_fetch(ptr, val);
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator>>=(const_value_type& val) const {
    return Kokkos::atomic_rshift_fetch(ptr, val);
  }
  KOKKOS_INLINE_FUNCTION
  const_value_type operator>>=(volatile const_value_type& val) const {
    return Kokkos::atomic_rshift_fetch(ptr, val);
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator+(const_value_type& val) const { return *ptr + val; }
  KOKKOS_INLINE_FUNCTION
  const_value_type operator+(volatile const_value_type& val) const {
    return *ptr + val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator-(const_value_type& val) const { return *ptr - val; }
  KOKKOS_INLINE_FUNCTION
  const_value_type operator-(volatile const_value_type& val) const {
    return *ptr - val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator*(const_value_type& val) const { return *ptr * val; }
  KOKKOS_INLINE_FUNCTION
  const_value_type operator*(volatile const_value_type& val) const {
    return *ptr * val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator/(const_value_type& val) const { return *ptr / val; }
  KOKKOS_INLINE_FUNCTION
  const_value_type operator/(volatile const_value_type& val) const {
    return *ptr / val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator%(const_value_type& val) const { return *ptr ^ val; }
  KOKKOS_INLINE_FUNCTION
  const_value_type operator%(volatile const_value_type& val) const {
    return *ptr ^ val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator!() const { return !*ptr; }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator&&(const_value_type& val) const {
    return *ptr && val;
  }
  KOKKOS_INLINE_FUNCTION
  const_value_type operator&&(volatile const_value_type& val) const {
    return *ptr && val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator||(const_value_type& val) const {
    return *ptr | val;
  }
  KOKKOS_INLINE_FUNCTION
  const_value_type operator||(volatile const_value_type& val) const {
    return *ptr | val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator&(const_value_type& val) const { return *ptr & val; }
  KOKKOS_INLINE_FUNCTION
  const_value_type operator&(volatile const_value_type& val) const {
    return *ptr & val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator|(const_value_type& val) const { return *ptr | val; }
  KOKKOS_INLINE_FUNCTION
  const_value_type operator|(volatile const_value_type& val) const {
    return *ptr | val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator^(const_value_type& val) const { return *ptr ^ val; }
  KOKKOS_INLINE_FUNCTION
  const_value_type operator^(volatile const_value_type& val) const {
    return *ptr ^ val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator~() const { return ~*ptr; }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator<<(const unsigned int& val) const {
    return *ptr << val;
  }
  KOKKOS_INLINE_FUNCTION
  const_value_type operator<<(volatile const unsigned int& val) const {
    return *ptr << val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator>>(const unsigned int& val) const {
    return *ptr >> val;
  }
  KOKKOS_INLINE_FUNCTION
  const_value_type operator>>(volatile const unsigned int& val) const {
    return *ptr >> val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator==(const AtomicReference& val) const { return *ptr == val; }
  KOKKOS_INLINE_FUNCTION
  bool operator==(volatile const AtomicReference& val) const {
    return *ptr == val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator!=(const AtomicReference& val) const { return *ptr != val; }
  KOKKOS_INLINE_FUNCTION
  bool operator!=(volatile const AtomicReference& val) const {
    return *ptr != val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator>=(const_value_type& val) const { return *ptr >= val; }
  KOKKOS_INLINE_FUNCTION
  bool operator>=(volatile const_value_type& val) const { return *ptr >= val; }

  KOKKOS_INLINE_FUNCTION
  bool operator<=(const_value_type& val) const { return *ptr <= val; }
  KOKKOS_INLINE_FUNCTION
  bool operator<=(volatile const_value_type& val) const { return *ptr <= val; }

  KOKKOS_INLINE_FUNCTION
  bool operator<(const_value_type& val) const { return *ptr < val; }
  KOKKOS_INLINE_FUNCTION
  bool operator<(volatile const_value_type& val) const { return *ptr < val; }

  KOKKOS_INLINE_FUNCTION
  bool operator>(const_value_type& val) const { return *ptr > val; }
  KOKKOS_INLINE_FUNCTION
  bool operator>(volatile const_value_type& val) const { return *ptr > val; }

  KOKKOS_INLINE_FUNCTION
  operator const_value_type() const {
    // return Kokkos::atomic_load(ptr);
    return *ptr;
  }

  KOKKOS_INLINE_FUNCTION
  operator volatile non_const_value_type() volatile const {
    // return Kokkos::atomic_load(ptr);
    return *ptr;
  }
};

} // namespace Impl

template<class ElementType>
struct AtomicAccessor {
  using offset_policy = AtomicAccessor;
  using element_type = ElementType;
  using reference = Impl::AtomicReference<ElementType>;
  using pointer = ElementType*;

  constexpr AtomicAccessor() = default;

  template<class OtherElementType, class Enable =
      std::enable_if_t<std::is_convertible<typename AtomicAccessor<OtherElementType>::pointer,pointer>::value>>
  constexpr AtomicAccessor(const AtomicAccessor<OtherElementType>&) {}

  constexpr AtomicAccessor(const std::experimental::default_accessor<std::remove_const_t<ElementType>>&) {}
  constexpr AtomicAccessor(const std::experimental::default_accessor<std::add_const_t<ElementType>>&) {}

  constexpr operator std::experimental::default_accessor<std::remove_const_t<ElementType>>() const { return std::experimental::default_accessor<ElementType>{}; }
  constexpr operator std::experimental::default_accessor<std::add_const_t<ElementType>>() const { return std::experimental::default_accessor<ElementType>{}; }

  constexpr typename offset_policy::pointer
    offset(pointer p, size_t i) const {
    return p+i;
  }

  constexpr reference access(pointer p, size_t i) const {
    return reference(p+i, Impl::AtomicViewConstTag());
  }
};

} // namespace Kokkos
