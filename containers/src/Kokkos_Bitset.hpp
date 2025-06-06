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

#ifndef KOKKOS_BITSET_HPP
#define KOKKOS_BITSET_HPP
#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_BITSET
#endif

#include <Kokkos_Core.hpp>
#include <Kokkos_BitManipulation.hpp>
#include <Kokkos_Functional.hpp>

#include <impl/Kokkos_Bitset_impl.hpp>

namespace Kokkos {

template <typename Device = Kokkos::DefaultExecutionSpace>
class Bitset;

template <typename Device = Kokkos::DefaultExecutionSpace>
class ConstBitset;

template <typename DstDevice, typename SrcDevice>
void deep_copy(Bitset<DstDevice>& dst, Bitset<SrcDevice> const& src);

template <typename DstDevice, typename SrcDevice>
void deep_copy(Bitset<DstDevice>& dst, ConstBitset<SrcDevice> const& src);

template <typename DstDevice, typename SrcDevice>
void deep_copy(ConstBitset<DstDevice>& dst, ConstBitset<SrcDevice> const& src);

/// A thread safe view to a bitset
template <typename Device>
class Bitset {
 public:
  using execution_space = typename Device::execution_space;
  using size_type       = unsigned int;

  static constexpr unsigned BIT_SCAN_REVERSE   = 1u;
  static constexpr unsigned MOVE_HINT_BACKWARD = 2u;

  static constexpr unsigned BIT_SCAN_FORWARD_MOVE_HINT_FORWARD = 0u;
  static constexpr unsigned BIT_SCAN_REVERSE_MOVE_HINT_FORWARD =
      BIT_SCAN_REVERSE;
  static constexpr unsigned BIT_SCAN_FORWARD_MOVE_HINT_BACKWARD =
      MOVE_HINT_BACKWARD;
  static constexpr unsigned BIT_SCAN_REVERSE_MOVE_HINT_BACKWARD =
      BIT_SCAN_REVERSE | MOVE_HINT_BACKWARD;

 private:
  static constexpr unsigned block_size = sizeof(unsigned) * CHAR_BIT;
  static constexpr unsigned block_mask = block_size - 1u;
  static constexpr unsigned block_shift =
      Kokkos::has_single_bit(block_size) ? Kokkos::bit_width(block_size) - 1
                                         : ~0u;

  //! Type of @ref m_blocks.
  using block_view_type = View<unsigned*, Device, MemoryTraits<RandomAccess>>;

 public:
  Bitset() = default;

  /// arg_size := number of bit in set
  Bitset(unsigned arg_size) : Bitset(Kokkos::view_alloc(), arg_size) {}

  template <class... P>
  Bitset(const Impl::ViewCtorProp<P...>& arg_prop, unsigned arg_size)
      : m_size(arg_size), m_last_block_mask(0u) {
    //! Ensure that allocation properties are consistent.
    using alloc_prop_t = std::decay_t<decltype(arg_prop)>;
    static_assert(alloc_prop_t::initialize,
                  "Allocation property 'initialize' should be true.");
    static_assert(
        !alloc_prop_t::has_pointer,
        "Allocation properties should not contain the 'pointer' property.");

    //! Update 'label' property and allocate.
    const auto prop_copy =
        Impl::with_properties_if_unset(arg_prop, std::string("Bitset"));
    m_blocks =
        block_view_type(prop_copy, ((m_size + block_mask) >> block_shift));

    for (int i = 0, end = static_cast<int>(m_size & block_mask); i < end; ++i) {
      m_last_block_mask |= 1u << i;
    }
  }

  KOKKOS_DEFAULTED_FUNCTION
  Bitset(const Bitset<Device>&) = default;

  KOKKOS_DEFAULTED_FUNCTION
  Bitset& operator=(const Bitset<Device>&) = default;

  KOKKOS_DEFAULTED_FUNCTION
  Bitset(Bitset<Device>&&) = default;

  KOKKOS_DEFAULTED_FUNCTION
  Bitset& operator=(Bitset<Device>&&) = default;

  KOKKOS_DEFAULTED_FUNCTION
  ~Bitset() = default;

  /// number of bits in the set
  /// can be call from the host or the device
  KOKKOS_FORCEINLINE_FUNCTION
  unsigned size() const { return m_size; }

  /// number of bits which are set to 1
  /// can only be called from the host
  unsigned count() const {
    Impl::BitsetCount<Bitset<Device>> f(*this);
    return f.apply();
  }

  /// set all bits to 1
  /// can only be called from the host
  void set() {
    Kokkos::deep_copy(m_blocks, ~0u);

    if (m_last_block_mask) {
      // clear the unused bits in the last block
      auto last_block = Kokkos::subview(m_blocks, m_blocks.extent(0) - 1u);
      Kokkos::deep_copy(typename Device::execution_space{}, last_block,
                        m_last_block_mask);
      Kokkos::fence(
          "Bitset::set: fence after clearing unused bits copying from "
          "HostSpace");
    }
  }

  /// set all bits to 0
  /// can only be called from the host
  void reset() { Kokkos::deep_copy(m_blocks, 0u); }

  /// set all bits to 0
  /// can only be called from the host
  void clear() { Kokkos::deep_copy(m_blocks, 0u); }

  /// set i'th bit to 1
  /// can only be called from the device
  KOKKOS_FORCEINLINE_FUNCTION
  bool set(unsigned i) const {
    if (i < m_size) {
      unsigned* block_ptr = &m_blocks[i >> block_shift];
      const unsigned mask = 1u << static_cast<int>(i & block_mask);

      return !(atomic_fetch_or(block_ptr, mask) & mask);
    }
    return false;
  }

  /// set i'th bit to 0
  /// can only be called from the device
  KOKKOS_FORCEINLINE_FUNCTION
  bool reset(unsigned i) const {
    if (i < m_size) {
      unsigned* block_ptr = &m_blocks[i >> block_shift];
      const unsigned mask = 1u << static_cast<int>(i & block_mask);

      return atomic_fetch_and(block_ptr, ~mask) & mask;
    }
    return false;
  }

  /// return true if the i'th bit set to 1
  /// can only be called from the device
  KOKKOS_FORCEINLINE_FUNCTION
  bool test(unsigned i) const {
    if (i < m_size) {
#ifdef KOKKOS_ENABLE_SYCL
      const unsigned block = Kokkos::atomic_load(&m_blocks[i >> block_shift]);
#else
      const unsigned block = volatile_load(&m_blocks[i >> block_shift]);
#endif
      const unsigned mask = 1u << static_cast<int>(i & block_mask);
      return block & mask;
    }
    return false;
  }

  /// used with find_any_set_near or find_any_unset_near functions
  /// returns the max number of times those functions should be call
  /// when searching for an available bit
  KOKKOS_FORCEINLINE_FUNCTION
  unsigned max_hint() const { return m_blocks.extent(0); }

  /// find a bit set to 1 near the hint
  /// returns a pair< bool, unsigned> where if result.first is true then
  /// result.second is the bit found and if result.first is false the
  /// result.second is a new hint
  KOKKOS_INLINE_FUNCTION
  Kokkos::pair<bool, unsigned> find_any_set_near(
      unsigned hint,
      unsigned scan_direction = BIT_SCAN_FORWARD_MOVE_HINT_FORWARD) const {
    const unsigned block_idx =
        (hint >> block_shift) < m_blocks.extent(0) ? (hint >> block_shift) : 0;
    const unsigned offset = hint & block_mask;
#ifdef KOKKOS_ENABLE_SYCL
    unsigned block = Kokkos::atomic_load(&m_blocks[block_idx]);
#else
    unsigned block = volatile_load(&m_blocks[block_idx]);
#endif
    block = !m_last_block_mask || (block_idx < (m_blocks.extent(0) - 1))
                ? block
                : block & m_last_block_mask;

    return find_any_helper(block_idx, offset, block, scan_direction);
  }

  /// find a bit set to 0 near the hint
  /// returns a pair< bool, unsigned> where if result.first is true then
  /// result.second is the bit found and if result.first is false the
  /// result.second is a new hint
  KOKKOS_INLINE_FUNCTION
  Kokkos::pair<bool, unsigned> find_any_unset_near(
      unsigned hint,
      unsigned scan_direction = BIT_SCAN_FORWARD_MOVE_HINT_FORWARD) const {
    const unsigned block_idx = hint >> block_shift;
    const unsigned offset    = hint & block_mask;
#ifdef KOKKOS_ENABLE_SYCL
    unsigned block = Kokkos::atomic_load(&m_blocks[block_idx]);
#else
    unsigned block = volatile_load(&m_blocks[block_idx]);
#endif
    block = !m_last_block_mask || (block_idx < (m_blocks.extent(0) - 1))
                ? ~block
                : ~block & m_last_block_mask;

    return find_any_helper(block_idx, offset, block, scan_direction);
  }

  KOKKOS_INLINE_FUNCTION constexpr bool is_allocated() const {
    return m_blocks.is_allocated();
  }

 private:
  KOKKOS_FORCEINLINE_FUNCTION
  Kokkos::pair<bool, unsigned> find_any_helper(unsigned block_idx,
                                               unsigned offset, unsigned block,
                                               unsigned scan_direction) const {
    Kokkos::pair<bool, unsigned> result(block > 0u, 0);

    if (!result.first) {
      result.second = update_hint(block_idx, offset, scan_direction);
    } else {
      result.second =
          scan_block((block_idx << block_shift), offset, block, scan_direction);
    }
    return result;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  unsigned scan_block(unsigned block_start, int offset, unsigned block,
                      unsigned scan_direction) const {
    offset = !(scan_direction & BIT_SCAN_REVERSE)
                 ? offset
                 : (offset + block_mask) & block_mask;
    block  = Experimental::rotr_builtin(block, offset);
    return (((!(scan_direction & BIT_SCAN_REVERSE)
                  ? Experimental::countr_zero_builtin(block)
                  : Experimental::bit_width_builtin(block) - 1) +
             offset) &
            block_mask) +
           block_start;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  unsigned update_hint(long long block_idx, unsigned offset,
                       unsigned scan_direction) const {
    block_idx += scan_direction & MOVE_HINT_BACKWARD ? -1 : 1;
    block_idx = block_idx >= 0 ? block_idx : m_blocks.extent(0) - 1;
    block_idx =
        block_idx < static_cast<long long>(m_blocks.extent(0)) ? block_idx : 0;

    return static_cast<unsigned>(block_idx) * block_size + offset;
  }

 private:
  unsigned m_size            = 0;
  unsigned m_last_block_mask = 0;
  block_view_type m_blocks;

 private:
  template <typename DDevice>
  friend class Bitset;

  template <typename DDevice>
  friend class ConstBitset;

  template <typename Bitset>
  friend struct Impl::BitsetCount;

  template <typename DstDevice, typename SrcDevice>
  friend void deep_copy(Bitset<DstDevice>& dst, Bitset<SrcDevice> const& src);

  template <typename DstDevice, typename SrcDevice>
  friend void deep_copy(Bitset<DstDevice>& dst,
                        ConstBitset<SrcDevice> const& src);
};

/// a thread-safe view to a const bitset
/// i.e. can only test bits
template <typename Device>
class ConstBitset {
 public:
  using execution_space = typename Device::execution_space;
  using size_type       = unsigned int;
  using block_view_type = typename Bitset<Device>::block_view_type::const_type;

 private:
  static constexpr unsigned block_size = sizeof(unsigned) * CHAR_BIT;
  static constexpr unsigned block_mask = block_size - 1u;
  static constexpr unsigned block_shift =
      Kokkos::has_single_bit(block_size) ? Kokkos::bit_width(block_size) - 1
                                         : ~0u;

 public:
  KOKKOS_FUNCTION
  ConstBitset() : m_size(0) {}

  KOKKOS_FUNCTION
  ConstBitset(Bitset<Device> const& rhs)
      : m_size(rhs.m_size), m_blocks(rhs.m_blocks) {}

  KOKKOS_FUNCTION
  ConstBitset(ConstBitset<Device> const& rhs)
      : m_size(rhs.m_size), m_blocks(rhs.m_blocks) {}

  KOKKOS_FUNCTION
  ConstBitset<Device>& operator=(Bitset<Device> const& rhs) {
    this->m_size   = rhs.m_size;
    this->m_blocks = rhs.m_blocks;

    return *this;
  }

  KOKKOS_FUNCTION
  ConstBitset<Device>& operator=(ConstBitset<Device> const& rhs) {
    this->m_size   = rhs.m_size;
    this->m_blocks = rhs.m_blocks;

    return *this;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  unsigned size() const { return m_size; }

  unsigned count() const {
    Impl::BitsetCount<ConstBitset<Device>> f(*this);
    return f.apply();
  }

  KOKKOS_FORCEINLINE_FUNCTION
  bool test(unsigned i) const {
    if (i < m_size) {
      const unsigned block = m_blocks[i >> block_shift];
      const unsigned mask  = 1u << static_cast<int>(i & block_mask);
      return block & mask;
    }
    return false;
  }

 private:
  unsigned m_size;
  block_view_type m_blocks;

 private:
  template <typename DDevice>
  friend class ConstBitset;

  template <typename Bitset>
  friend struct Impl::BitsetCount;

  template <typename DstDevice, typename SrcDevice>
  friend void deep_copy(Bitset<DstDevice>& dst,
                        ConstBitset<SrcDevice> const& src);

  template <typename DstDevice, typename SrcDevice>
  friend void deep_copy(ConstBitset<DstDevice>& dst,
                        ConstBitset<SrcDevice> const& src);
};

template <typename DstDevice, typename SrcDevice>
void deep_copy(Bitset<DstDevice>& dst, Bitset<SrcDevice> const& src) {
  if (dst.size() != src.size()) {
    Kokkos::Impl::throw_runtime_exception(
        "Error: Cannot deep_copy bitsets of different sizes!");
  }
  Kokkos::deep_copy(dst.m_blocks, src.m_blocks);
}

template <typename DstDevice, typename SrcDevice>
void deep_copy(Bitset<DstDevice>& dst, ConstBitset<SrcDevice> const& src) {
  if (dst.size() != src.size()) {
    Kokkos::Impl::throw_runtime_exception(
        "Error: Cannot deep_copy bitsets of different sizes!");
  }
  Kokkos::deep_copy(dst.m_blocks, src.m_blocks);
}

template <typename DstDevice, typename SrcDevice>
void deep_copy(ConstBitset<DstDevice>& dst, ConstBitset<SrcDevice> const& src) {
  if (dst.size() != src.size()) {
    Kokkos::Impl::throw_runtime_exception(
        "Error: Cannot deep_copy bitsets of different sizes!");
  }
  Kokkos::deep_copy(dst.m_blocks, src.m_blocks);
}

}  // namespace Kokkos

#ifdef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_BITSET
#undef KOKKOS_IMPL_PUBLIC_INCLUDE
#undef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_BITSET
#endif
#endif  // KOKKOS_BITSET_HPP
