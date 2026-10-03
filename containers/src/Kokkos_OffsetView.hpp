// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_OFFSETVIEW_HPP_
#define KOKKOS_OFFSETVIEW_HPP_
#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_OFFSETVIEW
#endif

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
import kokkos.core_impl;
#else
#include <Kokkos_Core.hpp>
#endif

#include <Kokkos_View.hpp>

#include <array>
#include <span>
#include <type_traits>

namespace Kokkos {

namespace Experimental {
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

template <class DataType, class... Properties>
class OffsetView;

template <class>
struct is_offset_view : public std::false_type {};

template <class D, class... P>
struct is_offset_view<OffsetView<D, P...>> : public std::true_type {};

template <class D, class... P>
struct is_offset_view<const OffsetView<D, P...>> : public std::true_type {};

template <class T>
inline constexpr bool is_offset_view_v = is_offset_view<T>::value;

#define KOKKOS_INVALID_OFFSET int64_t(0x7FFFFFFFFFFFFFFFLL)
#define KOKKOS_INVALID_INDEX_RANGE \
  { KOKKOS_INVALID_OFFSET, KOKKOS_INVALID_OFFSET }

template <typename iType,
          std::enable_if_t<std::is_integral_v<iType> && std::is_signed_v<iType>,
                           iType> = 0>
using IndexRange = Kokkos::Array<iType, 2>;

using index_list_type = std::initializer_list<int64_t>;

//  template <typename iType,
//    std::enable_if_t< std::is_integral<iType>::value &&
//      std::is_signed<iType>::value, iType > = 0> using min_index_type =
//      std::initializer_list<iType>;

namespace Impl {

// Fixed-size integral index containers usable as OffsetView begins/ends.
// Only std::array, Kokkos::Array, and static-extent std::span are accepted;
// their length is known at compile time (no runtime-sized ranges, no
// dynamic-extent std::span). The primary template rejects everything else.
template <typename>
struct FixedSizeIndexRange : std::false_type {};

template <typename T, std::size_t N>
struct FixedSizeIndexRange<Kokkos::Array<T, N>> {
  using value_type                  = Kokkos::Array<T, N>::value_type;
  static constexpr std::size_t size = N;
};

template <typename T, std::size_t N>
struct FixedSizeIndexRange<std::array<T, N>> {
  using value_type                  = std::array<T, N>::value_type;
  static constexpr std::size_t size = N;
};

template <typename T, std::size_t N>
struct FixedSizeIndexRange<std::span<T, N>>
    : std::bool_constant<N != std::dynamic_extent> {
  using value_type                  = std::span<T, N>::value_type;
  static constexpr std::size_t size = N;
};

// A fixed-size integral index range whose compile-time length equals Rank.
template <typename Range, std::size_t Rank>
concept IsFixedIntegralIndexRange = requires {
  requires std::is_integral_v<typename FixedSizeIndexRange<Range>::value_type>;
  requires FixedSizeIndexRange<Range>::size == Rank;
};

// FIXME Verification of bounds of OffsetView is not applied
template <unsigned, class MapType, class BeginsType>
KOKKOS_INLINE_FUNCTION bool offsetview_verify_operator_bounds(
    const MapType&, const BeginsType&) {
  return true;
}

template <unsigned R, class MapType, class BeginsType, class iType,
          class... Args>
KOKKOS_INLINE_FUNCTION bool offsetview_verify_operator_bounds(
    const MapType& map, const BeginsType& begins, const iType& i,
    Args... args) {
  const bool legalIndex =
      (int64_t(i) >= begins[R]) &&
      (int64_t(i) <= int64_t(begins[R] + map.extent(R) - 1));
  return legalIndex &&
         offsetview_verify_operator_bounds<R + 1>(map, begins, args...);
}
template <unsigned, class MapType, class BeginsType>
inline void offsetview_error_operator_bounds(char*, int, const MapType&,
                                             const BeginsType&) {}

template <unsigned R, class MapType, class BeginsType, class iType,
          class... Args>
inline void offsetview_error_operator_bounds(char* buf, int len,
                                             const MapType& map,
                                             const BeginsType begins,
                                             const iType& i, Args... args) {
  const int64_t b = begins[R];
  const int64_t e = b + map.extent(R) - 1;
  const int n =
      snprintf(buf, len, " %ld <= %ld <= %ld %c", static_cast<unsigned long>(b),
               static_cast<unsigned long>(i), static_cast<unsigned long>(e),
               (sizeof...(Args) ? ',' : ')'));
  offsetview_error_operator_bounds<R + 1>(buf + n, len - n, map, begins,
                                          args...);
}

template <class MemorySpace, class MapType, class BeginsType, class... Args>
KOKKOS_INLINE_FUNCTION void offsetview_verify_operator_bounds(
    Kokkos::Impl::SharedAllocationTracker const& tracker, const MapType& map,
    const BeginsType& begins, Args... args) {
  if (!offsetview_verify_operator_bounds<0>(map, begins, args...)) {
    KOKKOS_IF_ON_HOST(
        (enum {LEN = 1024}; char buffer[LEN];
         const std::string label = tracker.template get_label<MemorySpace>();
         int n                   = snprintf(buffer, LEN,
                                            "OffsetView bounds error of view labeled %s (",
                                            label.c_str());
         offsetview_error_operator_bounds<0>(buffer + n, LEN - n, map, begins,
                                             args...);
         Kokkos::abort(buffer);))

    KOKKOS_IF_ON_DEVICE(
        (Kokkos::abort("OffsetView bounds error"); (void)tracker;))
  }
}
// Fixed-capacity error message usable on host and device. Appended text is
// truncated once the buffer is full.
class OffsetViewErrorMessage {
  char m_buf[1024] = "Kokkos::Experimental::OffsetView ERROR: ";

 public:
  KOKKOS_FUNCTION OffsetViewErrorMessage& operator<<(const char* s) {
    Kokkos::Impl::strncat(m_buf, s,
                          sizeof(m_buf) - 1 - Kokkos::Impl::strlen(m_buf));
    return *this;
  }

  template <class Integral>
    requires(std::is_integral_v<Integral>)
  KOKKOS_FUNCTION OffsetViewErrorMessage& operator<<(Integral value) {
    char digits[24] = {};  // sign + 20 digits + '\0'
    Kokkos::Impl::to_chars_i(digits, digits + sizeof(digits) - 1, value);
    return *this << digits;
  }

  KOKKOS_FUNCTION const char* c_str() const { return m_buf; }
};
}  // namespace Impl

template <class DataType, class... Properties>
class OffsetView : public View<DataType, Properties...> {
 private:
  template <class, class...>
  friend class OffsetView;

  using base_t = View<DataType, Properties...>;

 public:
  // typedefs to reduce typing base_t:: further down
  using traits = typename base_t::traits;
  // FIXME: should be base_t::index_type after refactor
  using index_type   = typename base_t::memory_space::size_type;
  using pointer_type = typename base_t::pointer_type;

  using begins_type = Kokkos::Array<int64_t, base_t::rank()>;

  template <typename iType,
            std::enable_if_t<std::is_integral_v<iType>, iType> = 0>
  KOKKOS_FUNCTION int64_t begin(const iType local_dimension) const {
    return static_cast<size_t>(local_dimension) < base_t::rank()
               ? m_begins[local_dimension]
               : KOKKOS_INVALID_OFFSET;
  }

  KOKKOS_FUNCTION
  begins_type begins() const { return m_begins; }

  template <typename iType,
            std::enable_if_t<std::is_integral_v<iType>, iType> = 0>
  KOKKOS_FUNCTION int64_t end(const iType local_dimension) const {
    return begin(local_dimension) + base_t::extent(local_dimension);
  }

 private:
  begins_type m_begins;

 public:
  //----------------------------------------
  /** \brief  Compatible view of data type */
  using type =
      OffsetView<typename traits::data_type, typename traits::array_layout,
                 typename traits::device_type, typename traits::memory_traits>;

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_5
  /** \brief  Compatible view of array of scalar types */
  using array_type KOKKOS_DEPRECATED_WITH_COMMENT("Use type instead.") = type;
#endif

  /** \brief  Compatible view of const data type */
  using const_type =
      OffsetView<typename traits::const_data_type,
                 typename traits::array_layout, typename traits::device_type,
                 typename traits::memory_traits>;

  /** \brief  Compatible view of non-const data type */
  using non_const_type =
      OffsetView<typename traits::non_const_data_type,
                 typename traits::array_layout, typename traits::device_type,
                 typename traits::memory_traits>;

  /** \brief  Compatible host mirror view */
  using host_mirror_type = OffsetView<typename traits::non_const_data_type,
                                      typename traits::array_layout,
                                      typename traits::host_mirror_space>;

  template <size_t... I, class... OtherIndexTypes>
  KOKKOS_FUNCTION typename base_t::reference_type offset_operator(
      std::integer_sequence<size_t, I...>, OtherIndexTypes... indices) const {
    return base_t::operator()((indices - m_begins[I])...);
  }

  template <class OtherIndexType>
    requires(std::is_convertible_v<OtherIndexType, index_type> &&
             std::is_nothrow_constructible_v<index_type, OtherIndexType> &&
             (base_t::rank() == 1))
  KOKKOS_FUNCTION constexpr typename base_t::reference_type operator[](
      const OtherIndexType& idx) const {
    return base_t::operator[](idx - m_begins[0]);
  }

  template <class... OtherIndexTypes>
    requires((std::is_convertible_v<OtherIndexTypes, index_type> && ...) &&
             (std::is_nothrow_constructible_v<index_type, OtherIndexTypes> &&
              ...) &&
             (sizeof...(OtherIndexTypes) == base_t::rank()))
  KOKKOS_FUNCTION constexpr typename base_t::reference_type operator()(
      OtherIndexTypes... indices) const {
    return offset_operator(std::make_index_sequence<base_t::rank()>(),
                           indices...);
  }

  template <class... OtherIndexTypes>
  KOKKOS_FUNCTION constexpr typename base_t::reference_type access(
      OtherIndexTypes... args) const = delete;

  //----------------------------------------

  //----------------------------------------
  // Standard destructor, constructors, and assignment operators

  KOKKOS_FUNCTION
  OffsetView() : base_t() {
    for (size_t i = 0; i < base_t::rank(); ++i)
      m_begins[i] = KOKKOS_INVALID_OFFSET;
  }

  // interoperability with View
 private:
  using view_type =
      View<typename traits::data_type, typename traits::array_layout,
           typename traits::device_type, typename traits::memory_traits>;

 public:
  KOKKOS_FUNCTION
  view_type view() const { return *this; }

  template <class RT, class... RP>
  KOKKOS_FUNCTION OffsetView(const View<RT, RP...>& aview) : base_t(aview) {
    for (size_t i = 0; i < View<RT, RP...>::rank(); ++i) {
      m_begins[i] = 0;
    }
  }

  template <class RT, class... RP>
  KOKKOS_FUNCTION OffsetView(const View<RT, RP...>& aview,
                             const index_list_type& begins)
      : base_t(aview) {
    // No view constructor properties are given, so there is no label, just as
    // for a View constructed from properties without one.
    runtime_check_begins(begins, "");
    for (size_t i = 0; i < begins.size(); ++i) {
      m_begins[i] = at(begins, i);
    }
  }
  template <class RT, class... RP>
  KOKKOS_FUNCTION OffsetView(const View<RT, RP...>& aview,
                             const begins_type& beg)
      : base_t(aview), m_begins(beg) {}

  // may assign unmanaged from managed.

  template <class RT, class... RP>
  KOKKOS_FUNCTION OffsetView(const OffsetView<RT, RP...>& rhs)
      : base_t(rhs.view()), m_begins(rhs.m_begins) {}

 private:
  // Label used in error messages for views that do not own their allocation.
  // A function rather than a static data member, which device code can't use.
  KOKKOS_FUNCTION static constexpr const char* unmanaged_label() {
    return "UNMANAGED";
  }

  enum class subtraction_failure {
    none,
    negative,
    overflow,
  };

  // Subtraction should return a non-negative number and not overflow
  KOKKOS_FUNCTION static subtraction_failure check_subtraction(int64_t lhs,
                                                               int64_t rhs) {
    if (lhs < rhs) return subtraction_failure::negative;

    if (static_cast<uint64_t>(-1) / static_cast<uint64_t>(2) <
        static_cast<uint64_t>(lhs) - static_cast<uint64_t>(rhs))
      return subtraction_failure::overflow;

    return subtraction_failure::none;
  }

  // Get the element at position pos from any integral index range (e.g.
  // Kokkos::Array or std::initializer_list). Returns by value.
  template <typename Range>
  KOKKOS_FUNCTION static int64_t at(const Range& a, size_t pos) {
    return *(a.begin() + pos);
  }

  // Whether an index range holds exactly one entry per rank. The length of
  // fixed-size ranges is already enforced by the constructor constraints, so
  // only runtime-sized ranges (index_list_type) are checked.
  template <typename Range>
  KOKKOS_FUNCTION static constexpr bool rank_is_equal_size(const Range& r) {
    if constexpr (Impl::IsFixedIntegralIndexRange<Range, base_t::rank()>) {
      return true;
    } else {
      return r.size() == base_t::rank();
    }
  }

  template <typename Range>
  KOKKOS_FUNCTION static void append_rank_error(
      Impl::OffsetViewErrorMessage& msg, const Range& r, const char* name) {
    if (rank_is_equal_size(r)) return;
    msg << name << ".size() (" << r.size() << ") != Rank (" << base_t::rank()
        << ")\n";
  }

  // Number of entries of an index range that are not KOKKOS_INVALID_OFFSET
  template <typename Range>
  KOKKOS_FUNCTION static size_t count_valid_offsets(const Range& r) {
    size_t num_offsets = 0;
    for (size_t i = 0; i < r.size(); ++i)
      if (at(r, i) != KOKKOS_INVALID_OFFSET) ++num_offsets;
    return num_offsets;
  }

  // Whether an index range provides one valid offset per rank. Offsets are only
  // supported for views whose extents are all dynamic.
  template <typename Range>
  KOKKOS_FUNCTION static bool has_valid_range(const Range& r) {
    return traits::rank_dynamic == base_t::rank() && rank_is_equal_size(r) &&
           count_valid_offsets(r) == traits::rank_dynamic;
  }

  // Appends the reasons why has_valid_range(r) fails.
  template <typename Range>
  KOKKOS_FUNCTION static void append_range_error(
      Impl::OffsetViewErrorMessage& msg, const Range& r, const char* name) {
    constexpr size_t rank_dynamic = traits::rank_dynamic;
    const size_t valid_offsets    = count_valid_offsets(r);

    if (rank_dynamic != base_t::rank())
      msg << "The full rank must be the same as the dynamic rank. full rank = "
          << base_t::rank() << " dynamic rank = " << rank_dynamic << "\n";
    if (!rank_is_equal_size(r)) {
      append_rank_error(msg, r, name);
    } else if (valid_offsets != rank_dynamic) {
      msg << "The number of offsets provided in " << name << " ( "
          << valid_offsets << " ) must equal the dynamic rank ( "
          << rank_dynamic << " ).\n";
    }
  }

  // Check that begins and ends have one entry per rank and that begins <= ends
  // for all elements. B, E can be any integral index range. ends is only
  // checked for its size: KOKKOS_INVALID_OFFSET is a valid exclusive end.
  // label names the view in the error message; the base View does not exist
  // yet, so it has to be provided by the caller.
  template <typename B, typename E>
  KOKKOS_FUNCTION static void runtime_check_begins_ends(const B& begins,
                                                        const E& ends,
                                                        const char* label) {
    bool valid = has_valid_range(begins) && rank_is_equal_size(ends);
    for (size_t i = 0; valid && i != base_t::rank(); ++i)
      valid = check_subtraction(at(ends, i), at(begins, i)) ==
              subtraction_failure::none;
    if (valid) return;

    Impl::OffsetViewErrorMessage msg;
    msg << "label=(\"" << label << "\")\n";
    if (!has_valid_range(begins)) append_range_error(msg, begins, "begins");
    if (!rank_is_equal_size(ends)) append_rank_error(msg, ends, "ends");

    // If there are no rank errors, then arg_rank == Rank
    // Otherwise, check as much as possible
    for (size_t i = 0; i != base_t::rank(); ++i) {
      subtraction_failure sf = check_subtraction(at(ends, i), at(begins, i));
      if (sf == subtraction_failure::none) continue;
      msg << "(ends[" << i << "] (" << at(ends, i) << ") - begins[" << i
          << "] (" << at(begins, i) << "))"
          << (sf == subtraction_failure::negative ? " must be non-negative\n"
                                                  : " overflows\n");
    }
    Kokkos::abort(msg.c_str());
  }

  // Check begins given alongside already known extents.
  // label names the view in the error message.
  KOKKOS_FUNCTION static void runtime_check_begins(
      const index_list_type& begins, const char* label) {
    if (has_valid_range(begins)) return;

    Impl::OffsetViewErrorMessage msg;
    msg << "label=(\"" << label << "\")\n";
    append_range_error(msg, begins, "begins");
    Kokkos::abort(msg.c_str());
  }

  // Computes the layout after checking begins and ends.
  // label names the view in the error message.
  template <typename B, typename E>
  KOKKOS_FUNCTION static typename traits::array_layout
  compute_layout_from_begins_ends(const B& begins_, const E& ends_,
                                  const char* label) {
    runtime_check_begins_ends(begins_, ends_, label);
    return typename traits::array_layout(
        base_t::rank() > 0 ? at(ends_, 0) - at(begins_, 0)
                           : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        base_t::rank() > 1 ? at(ends_, 1) - at(begins_, 1)
                           : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        base_t::rank() > 2 ? at(ends_, 2) - at(begins_, 2)
                           : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        base_t::rank() > 3 ? at(ends_, 3) - at(begins_, 3)
                           : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        base_t::rank() > 4 ? at(ends_, 4) - at(begins_, 4)
                           : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        base_t::rank() > 5 ? at(ends_, 5) - at(begins_, 5)
                           : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        base_t::rank() > 6 ? at(ends_, 6) - at(begins_, 6)
                           : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        base_t::rank() > 7 ? at(ends_, 7) - at(begins_, 7)
                           : KOKKOS_IMPL_CTOR_DEFAULT_ARG);
  }

  struct begins_ends_tag {};

  // Constructors from view constructor properties after checking begins and
  // ends. Each of B, E can be any integral index range. As for View, the
  // allocating version is host-only while the wrapping one (has_pointer) can
  // also be called on device.
  template <class... P, typename B, typename E>
    requires(!Kokkos::Impl::ViewCtorProp<P...>::has_pointer)
  OffsetView(begins_ends_tag, const Kokkos::Impl::ViewCtorProp<P...>& arg_prop,
             const B& begins_, const E& ends_)
      : base_t(arg_prop, compute_layout_from_begins_ends(
                             begins_, ends_,
                             Kokkos::Impl::get_property<Kokkos::Impl::LabelTag>(
                                 Kokkos::Impl::with_properties_if_unset(
                                     arg_prop, std::string{}))
                                 .c_str())) {
    for (size_t i = 0; i != m_begins.size(); ++i) m_begins[i] = at(begins_, i);
  }

  template <class... P, typename B, typename E>
    requires(Kokkos::Impl::ViewCtorProp<P...>::has_pointer)
  KOKKOS_FUNCTION OffsetView(begins_ends_tag,
                             const Kokkos::Impl::ViewCtorProp<P...>& arg_prop,
                             const B& begins_, const E& ends_)
      : base_t(arg_prop, compute_layout_from_begins_ends(begins_, ends_,
                                                         unmanaged_label())) {
    static_assert(
        std::is_same_v<pointer_type,
                       typename Kokkos::Impl::ViewCtorProp<P...>::pointer_type>,
        "When constructing OffsetView to wrap user memory, you must supply "
        "matching pointer type");
    for (size_t i = 0; i != m_begins.size(); ++i) m_begins[i] = at(begins_, i);
  }

 public:
  // Constructors around unmanaged data. ends_ holds the exclusive end index for
  // each dimension. Named begin/end arguments must be a fixed-size integral
  // index range (std::array, Kokkos::Array, or static-extent std::span) whose
  // compile-time length equals the rank; see IsFixedIntegralIndexRange. The
  // index_list_type overloads accept brace-init lists ({a, b}); their runtime
  // size may differ from the rank, which the range checks validate.
  template <class Begins, class Ends>
    requires(Impl::IsFixedIntegralIndexRange<Begins, base_t::rank()> &&
             Impl::IsFixedIntegralIndexRange<Ends, base_t::rank()>)
  KOKKOS_FUNCTION OffsetView(const pointer_type& p, const Begins& begins_,
                             const Ends& ends_)
      : OffsetView(begins_ends_tag{}, Kokkos::view_wrap(p), begins_, ends_) {}

  template <class Begins>
    requires(Impl::IsFixedIntegralIndexRange<Begins, base_t::rank()>)
  KOKKOS_FUNCTION OffsetView(const pointer_type& p, const Begins& begins_,
                             index_list_type ends_)
      : OffsetView(begins_ends_tag{}, Kokkos::view_wrap(p), begins_, ends_) {}

  template <class Ends>
    requires(Impl::IsFixedIntegralIndexRange<Ends, base_t::rank()>)
  KOKKOS_FUNCTION OffsetView(const pointer_type& p, index_list_type begins_,
                             const Ends& ends_)
      : OffsetView(begins_ends_tag{}, Kokkos::view_wrap(p), begins_, ends_) {}

  KOKKOS_FUNCTION
  OffsetView(const pointer_type& p, index_list_type begins_,
             index_list_type ends_)
      : OffsetView(begins_ends_tag{}, Kokkos::view_wrap(p), begins_, ends_) {}

  // Constructors from view constructor properties using begin/end ranges.
  // They allocate memory, or wrap it if arg_prop holds a pointer (view_wrap).
  // begins_ contains the first valid index for each dimension (inclusive).
  // ends_ contains the exclusive end index for each dimension.
  // index_list_type ({-1, 3}) is preferred for brace-init lists (SCS over UCS).
  // begins_type (Array) overloads accept named Array<int64_t, N> variables.
  template <class... P>
  explicit OffsetView(const Kokkos::Impl::ViewCtorProp<P...>& arg_prop,
                      index_list_type begins_, index_list_type ends_)
      : OffsetView(begins_ends_tag{}, arg_prop, begins_, ends_) {}

  template <Kokkos::Impl::ViewLabel Label>
  explicit OffsetView(const Label& arg_label, index_list_type begins_,
                      index_list_type ends_)
      : OffsetView(Kokkos::Impl::ViewCtorProp<std::string>(arg_label), begins_,
                   ends_) {}

  template <class... P, class Begins, class Ends>
    requires(Impl::IsFixedIntegralIndexRange<Begins, base_t::rank()> &&
             Impl::IsFixedIntegralIndexRange<Ends, base_t::rank()>)
  explicit OffsetView(const Kokkos::Impl::ViewCtorProp<P...>& arg_prop,
                      const Begins& begins_, const Ends& ends_)
      : OffsetView(begins_ends_tag{}, arg_prop, begins_, ends_) {}

  template <Kokkos::Impl::ViewLabel Label, class Begins, class Ends>
    requires(Impl::IsFixedIntegralIndexRange<Begins, base_t::rank()> &&
             Impl::IsFixedIntegralIndexRange<Ends, base_t::rank()>)
  explicit OffsetView(const Label& arg_label, const Begins& begins_,
                      const Ends& ends_)
      : OffsetView(Kokkos::Impl::ViewCtorProp<std::string>(arg_label), begins_,
                   ends_) {}

  // Deprecated: use begin/end range constructors instead.
  template <Kokkos::Impl::ViewLabel Label>
  KOKKOS_DEPRECATED_WITH_COMMENT(
      "OffsetView pair constructors are deprecated. Use begins/ends range "
      "constructors instead: OffsetView(label, begins, ends) where begins and "
      "ends are arrays of first and exclusive-end indices per dimension.")
  explicit OffsetView(
      const Label& arg_label, const std::pair<int64_t, int64_t> range0,
      const std::pair<int64_t, int64_t> range1 = KOKKOS_INVALID_INDEX_RANGE,
      const std::pair<int64_t, int64_t> range2 = KOKKOS_INVALID_INDEX_RANGE,
      const std::pair<int64_t, int64_t> range3 = KOKKOS_INVALID_INDEX_RANGE,
      const std::pair<int64_t, int64_t> range4 = KOKKOS_INVALID_INDEX_RANGE,
      const std::pair<int64_t, int64_t> range5 = KOKKOS_INVALID_INDEX_RANGE,
      const std::pair<int64_t, int64_t> range6 = KOKKOS_INVALID_INDEX_RANGE,
      const std::pair<int64_t, int64_t> range7 = KOKKOS_INVALID_INDEX_RANGE)
      : OffsetView(Kokkos::Impl::ViewCtorProp<std::string>(arg_label),
                   typename traits::array_layout(
                       range0.first == KOKKOS_INVALID_OFFSET
                           ? KOKKOS_IMPL_CTOR_DEFAULT_ARG - 1
                           : range0.second - range0.first + 1,
                       range1.first == KOKKOS_INVALID_OFFSET
                           ? KOKKOS_IMPL_CTOR_DEFAULT_ARG
                           : range1.second - range1.first + 1,
                       range2.first == KOKKOS_INVALID_OFFSET
                           ? KOKKOS_IMPL_CTOR_DEFAULT_ARG
                           : range2.second - range2.first + 1,
                       range3.first == KOKKOS_INVALID_OFFSET
                           ? KOKKOS_IMPL_CTOR_DEFAULT_ARG
                           : range3.second - range3.first + 1,
                       range4.first == KOKKOS_INVALID_OFFSET
                           ? KOKKOS_IMPL_CTOR_DEFAULT_ARG
                           : range4.second - range4.first + 1,
                       range5.first == KOKKOS_INVALID_OFFSET
                           ? KOKKOS_IMPL_CTOR_DEFAULT_ARG
                           : range5.second - range5.first + 1,
                       range6.first == KOKKOS_INVALID_OFFSET
                           ? KOKKOS_IMPL_CTOR_DEFAULT_ARG
                           : range6.second - range6.first + 1,
                       range7.first == KOKKOS_INVALID_OFFSET
                           ? KOKKOS_IMPL_CTOR_DEFAULT_ARG
                           : range7.second - range7.first + 1),
                   {range0.first, range1.first, range2.first, range3.first,
                    range4.first, range5.first, range6.first, range7.first}) {
    static_assert(
        base_t::rank() != 2,
        "OffsetView: pair constructors are ambiguous for rank-2 views — "
        "{a,b},{c,d} could mean two 1D ranges or one 2D begins+ends array. "
        "Use begins/ends range constructors instead.");
  }

  template <class... P>
  KOKKOS_DEPRECATED_WITH_COMMENT(
      "OffsetView pair constructors are deprecated. Use begins/ends range "
      "constructors instead: OffsetView(prop, begins, ends) where begins and "
      "ends are arrays of first and exclusive-end indices per dimension.")
  explicit OffsetView(
      const Kokkos::Impl::ViewCtorProp<P...>& arg_prop,
      const std::pair<int64_t, int64_t> range0 = KOKKOS_INVALID_INDEX_RANGE,
      const std::pair<int64_t, int64_t> range1 = KOKKOS_INVALID_INDEX_RANGE,
      const std::pair<int64_t, int64_t> range2 = KOKKOS_INVALID_INDEX_RANGE,
      const std::pair<int64_t, int64_t> range3 = KOKKOS_INVALID_INDEX_RANGE,
      const std::pair<int64_t, int64_t> range4 = KOKKOS_INVALID_INDEX_RANGE,
      const std::pair<int64_t, int64_t> range5 = KOKKOS_INVALID_INDEX_RANGE,
      const std::pair<int64_t, int64_t> range6 = KOKKOS_INVALID_INDEX_RANGE,
      const std::pair<int64_t, int64_t> range7 = KOKKOS_INVALID_INDEX_RANGE)
      : OffsetView(arg_prop,
                   typename traits::array_layout(
                       range0.first == KOKKOS_INVALID_OFFSET
                           ? KOKKOS_IMPL_CTOR_DEFAULT_ARG
                           : range0.second - range0.first + 1,
                       range1.first == KOKKOS_INVALID_OFFSET
                           ? KOKKOS_IMPL_CTOR_DEFAULT_ARG
                           : range1.second - range1.first + 1,
                       range2.first == KOKKOS_INVALID_OFFSET
                           ? KOKKOS_IMPL_CTOR_DEFAULT_ARG
                           : range2.second - range2.first + 1,
                       range3.first == KOKKOS_INVALID_OFFSET
                           ? KOKKOS_IMPL_CTOR_DEFAULT_ARG
                           : range3.second - range3.first + 1,
                       range4.first == KOKKOS_INVALID_OFFSET
                           ? KOKKOS_IMPL_CTOR_DEFAULT_ARG
                           : range4.second - range4.first + 1,
                       range5.first == KOKKOS_INVALID_OFFSET
                           ? KOKKOS_IMPL_CTOR_DEFAULT_ARG
                           : range5.second - range5.first + 1,
                       range6.first == KOKKOS_INVALID_OFFSET
                           ? KOKKOS_IMPL_CTOR_DEFAULT_ARG
                           : range6.second - range6.first + 1,
                       range7.first == KOKKOS_INVALID_OFFSET
                           ? KOKKOS_IMPL_CTOR_DEFAULT_ARG
                           : range7.second - range7.first + 1),
                   {range0.first, range1.first, range2.first, range3.first,
                    range4.first, range5.first, range6.first, range7.first}) {
    static_assert(
        base_t::rank() != 2,
        "OffsetView: pair constructors are ambiguous for rank-2 views — "
        "{a,b},{c,d} could mean two 1D ranges or one 2D begins+ends array. "
        "Use begins/ends range constructors instead.");
  }

  template <class... P>
    requires(Kokkos::Impl::ViewCtorProp<P...>::has_pointer)
  explicit KOKKOS_FUNCTION OffsetView(
      const Kokkos::Impl::ViewCtorProp<P...>& arg_prop,
      typename traits::array_layout const& arg_layout,
      const index_list_type begins)
      : base_t(arg_prop, arg_layout) {
    runtime_check_begins(begins, unmanaged_label());
    for (size_t i = 0; i < begins.size(); ++i) {
      m_begins[i] = begins.begin()[i];
    }
    static_assert(
        std::is_same_v<pointer_type,
                       typename Kokkos::Impl::ViewCtorProp<P...>::pointer_type>,
        "When constructing OffsetView to wrap user memory, you must supply "
        "matching pointer type");
  }

  template <class... P>
    requires(!Kokkos::Impl::ViewCtorProp<P...>::has_pointer)
  explicit OffsetView(const Kokkos::Impl::ViewCtorProp<P...>& arg_prop,
                      typename traits::array_layout const& arg_layout,
                      const index_list_type minIndices)
      : base_t(arg_prop, arg_layout) {
    for (size_t i = 0; i < base_t::rank(); ++i)
      m_begins[i] = minIndices.begin()[i];
  }
};

/** \brief Temporary free function rank()
 *         until rank() is implemented
 *         in the View
 */
template <typename D, class... P>
KOKKOS_INLINE_FUNCTION constexpr unsigned rank(const OffsetView<D, P...>& V) {
  return V.rank();
}  // Temporary until added to view

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
namespace Impl {

template <class T>
KOKKOS_INLINE_FUNCTION std::enable_if_t<std::is_integral_v<T>, T> shift_input(
    const T arg, const int64_t offset) {
  return arg - offset;
}

KOKKOS_INLINE_FUNCTION
Kokkos::ALL_t shift_input(const Kokkos::ALL_t arg, const int64_t /*offset*/) {
  return arg;
}

template <class T>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<std::is_integral_v<T>, Kokkos::pair<T, T>>
    shift_input(const Kokkos::pair<T, T> arg, const int64_t offset) {
  return Kokkos::make_pair<T, T>(arg.first - offset, arg.second - offset);
}
template <class T>
inline std::enable_if_t<std::is_integral_v<T>, std::pair<T, T>> shift_input(
    const std::pair<T, T> arg, const int64_t offset) {
  return std::make_pair<T, T>(arg.first - offset, arg.second - offset);
}

template <size_t N, class Arg, class A>
KOKKOS_INLINE_FUNCTION void map_arg_to_new_begin(
    const size_t i, Kokkos::Array<int64_t, N>& subviewBegins,
    std::enable_if_t<N != 0, const Arg> shiftedArg, const Arg arg,
    const A viewBegins, size_t& counter) {
  if (!std::is_integral_v<Arg>) {
    subviewBegins[counter] = shiftedArg == arg ? viewBegins[i] : 0;
    counter++;
  }
}

template <size_t N, class Arg, class A>
KOKKOS_INLINE_FUNCTION void map_arg_to_new_begin(
    const size_t /*i*/, Kokkos::Array<int64_t, N>& /*subviewBegins*/,
    std::enable_if_t<N == 0, const Arg> /*shiftedArg*/, const Arg /*arg*/,
    const A /*viewBegins*/, size_t& /*counter*/) {}

template <size_t... Idx, class V, class... Slices>
KOKKOS_FUNCTION auto subview_offset(std::index_sequence<Idx...>, const V& src,
                                    Slices... slices) {
  // Create a subview via shifted slices
  auto sub_view = subview(src.view(), shift_input(slices, src.begin(Idx))...);
  using sub_view_t = decltype(sub_view);

  // extract the subview_begins
  Kokkos::Array<int64_t, sub_view_t::rank()> subview_begins;
  size_t counter = 0;
  auto begins    = src.begins();
  (Impl::map_arg_to_new_begin(Idx, subview_begins,
                              shift_input(slices, src.begin(Idx)), slices,
                              begins, counter),
   ...);

  // construct and return the new OffsetView
  return OffsetView<
      typename sub_view_t::data_type, typename sub_view_t::array_layout,
      typename sub_view_t::device_type, typename sub_view_t::memory_traits>(
      sub_view, subview_begins);
}
}  // namespace Impl
}  // namespace Experimental

template <class D, class... P, class... Args>
KOKKOS_INLINE_FUNCTION auto subview(
    const Kokkos::Experimental::OffsetView<D, P...>& src, Args... args) {
  static_assert(
      Kokkos::Experimental::OffsetView<D, P...>::rank() == sizeof...(Args),
      "subview requires one argument for each source OffsetView rank");

  return Kokkos::Experimental::Impl::subview_offset(
      std::make_index_sequence<sizeof...(Args)>(), src, args...);
}

}  // namespace Kokkos
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Experimental {
template <class LT, class... LP, class RT, class... RP>
KOKKOS_INLINE_FUNCTION bool operator==(const OffsetView<LT, LP...>& lhs,
                                       const OffsetView<RT, RP...>& rhs) {
  // Same data, layout, dimensions
  using lhs_traits = ViewTraits<LT, LP...>;
  using rhs_traits = ViewTraits<RT, RP...>;

  return std::is_same_v<typename lhs_traits::const_value_type,
                        typename rhs_traits::const_value_type> &&
         std::is_same_v<typename lhs_traits::array_layout,
                        typename rhs_traits::array_layout> &&
         std::is_same_v<typename lhs_traits::memory_space,
                        typename rhs_traits::memory_space> &&
         lhs.data() == rhs.data() && lhs.span() == rhs.span() &&
         lhs.extents() == rhs.extents() && lhs.begin(0) == rhs.begin(0) &&
         lhs.begin(1) == rhs.begin(1) && lhs.begin(2) == rhs.begin(2) &&
         lhs.begin(3) == rhs.begin(3) && lhs.begin(4) == rhs.begin(4) &&
         lhs.begin(5) == rhs.begin(5) && lhs.begin(6) == rhs.begin(6) &&
         lhs.begin(7) == rhs.begin(7);
}

template <class LT, class... LP, class RT, class... RP>
KOKKOS_INLINE_FUNCTION bool operator!=(const OffsetView<LT, LP...>& lhs,
                                       const OffsetView<RT, RP...>& rhs) {
  return !(operator==(lhs, rhs));
}

template <class LT, class... LP, class RT, class... RP>
KOKKOS_INLINE_FUNCTION bool operator==(const View<LT, LP...>& lhs,
                                       const OffsetView<RT, RP...>& rhs) {
  // Same data, layout, dimensions
  using lhs_traits = ViewTraits<LT, LP...>;
  using rhs_traits = ViewTraits<RT, RP...>;

  return std::is_same_v<typename lhs_traits::const_value_type,
                        typename rhs_traits::const_value_type> &&
         std::is_same_v<typename lhs_traits::array_layout,
                        typename rhs_traits::array_layout> &&
         std::is_same_v<typename lhs_traits::memory_space,
                        typename rhs_traits::memory_space> &&
         lhs.data() == rhs.data() && lhs.span() == rhs.span() &&
         lhs.extents() == rhs.extents();
}

template <class LT, class... LP, class RT, class... RP>
KOKKOS_INLINE_FUNCTION bool operator==(const OffsetView<LT, LP...>& lhs,
                                       const View<RT, RP...>& rhs) {
  return rhs == lhs;
}

}  // namespace Experimental
} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

template <class DT, class... DP>
inline void deep_copy(const Experimental::OffsetView<DT, DP...>& dst,
                      typename ViewTraits<DT, DP...>::const_value_type& value) {
  static_assert(
      std::is_same_v<typename ViewTraits<DT, DP...>::non_const_value_type,
                     typename ViewTraits<DT, DP...>::value_type>,
      "deep_copy requires non-const type");

  auto dstView = dst.view();
  Kokkos::deep_copy(dstView, value);
}

template <class DT, class... DP, class ST, class... SP>
inline void deep_copy(const Experimental::OffsetView<DT, DP...>& dst,
                      const Experimental::OffsetView<ST, SP...>& value) {
  static_assert(
      std::is_same_v<typename ViewTraits<DT, DP...>::value_type,
                     typename ViewTraits<ST, SP...>::non_const_value_type>,
      "deep_copy requires matching non-const destination type");

  auto dstView = dst.view();
  Kokkos::deep_copy(dstView, value.view());
}
template <class DT, class... DP, class ST, class... SP>
inline void deep_copy(const Experimental::OffsetView<DT, DP...>& dst,
                      const View<ST, SP...>& value) {
  static_assert(
      std::is_same_v<typename ViewTraits<DT, DP...>::value_type,
                     typename ViewTraits<ST, SP...>::non_const_value_type>,
      "deep_copy requires matching non-const destination type");

  auto dstView = dst.view();
  Kokkos::deep_copy(dstView, value);
}

template <class DT, class... DP, class ST, class... SP>
inline void deep_copy(const View<DT, DP...>& dst,
                      const Experimental::OffsetView<ST, SP...>& value) {
  static_assert(
      std::is_same_v<typename ViewTraits<DT, DP...>::value_type,
                     typename ViewTraits<ST, SP...>::non_const_value_type>,
      "deep_copy requires matching non-const destination type");

  Kokkos::deep_copy(dst, value.view());
}

namespace Impl {

// Deduce Mirror Types
template <class Space, class T, class... P>
struct MirrorOffsetViewType {
  // The incoming view_type
  using src_view_type = typename Kokkos::Experimental::OffsetView<T, P...>;
  // The memory space for the mirror view
  using memory_space = typename Space::memory_space;
  // Check whether it is the same memory space
  enum {
    is_same_memspace =
        std::is_same_v<memory_space, typename src_view_type::memory_space>
  };
  // The array_layout
  using array_layout = typename src_view_type::array_layout;
  // The data type (we probably want it non-const since otherwise we can't even
  // deep_copy to it.)
  using data_type = typename src_view_type::non_const_data_type;
  // The destination view type if it is not the same memory space
  using dest_view_type =
      Kokkos::Experimental::OffsetView<data_type, array_layout, Space>;
  // If it is the same memory_space return the existing view_type
  // This will also keep the unmanaged trait if necessary
  using view_type =
      std::conditional_t<is_same_memspace, src_view_type, dest_view_type>;
};

}  // namespace Impl

namespace Impl {

// create a mirror
// private interface that accepts arbitrary view constructor args passed by a
// view_alloc
template <class T, class... P, class... ViewCtorArgs>
inline auto create_mirror(const Kokkos::Experimental::OffsetView<T, P...>& src,
                          const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop) {
  check_view_ctor_args_create_mirror<ViewCtorArgs...>();

  if constexpr (Impl::ViewCtorProp<ViewCtorArgs...>::has_memory_space) {
    using Space = typename Impl::ViewCtorProp<ViewCtorArgs...>::memory_space;

    auto prop_copy = Impl::with_properties_if_unset(
        arg_prop, std::string(src.label()).append("_mirror"));

    return typename Kokkos::Impl::MirrorOffsetViewType<
        Space, T, P...>::dest_view_type(prop_copy, src.layout(),
                                        {src.begin(0), src.begin(1),
                                         src.begin(2), src.begin(3),
                                         src.begin(4), src.begin(5),
                                         src.begin(6), src.begin(7)});
  } else {
    return typename Kokkos::Experimental::OffsetView<T, P...>::host_mirror_type(
        Kokkos::create_mirror(arg_prop, src.view()), src.begins());
  }
}

}  // namespace Impl

// public interface
template <class T, class... P>
inline auto create_mirror(
    const Kokkos::Experimental::OffsetView<T, P...>& src) {
  return Impl::create_mirror(src, Impl::ViewCtorProp<>{});
}

// public interface that accepts a without initializing flag
template <class T, class... P>
inline auto create_mirror(
    Kokkos::Impl::WithoutInitializing_t wi,
    const Kokkos::Experimental::OffsetView<T, P...>& src) {
  return Impl::create_mirror(src, Kokkos::view_alloc(wi));
}

// public interface that accepts a space
template <class Space, class T, class... P>
  requires Kokkos::is_space<Space>::value
inline auto create_mirror(
    const Space&, const Kokkos::Experimental::OffsetView<T, P...>& src) {
  return Impl::create_mirror(
      src, Kokkos::view_alloc(typename Space::memory_space{}));
}

// public interface that accepts a space and a without initializing flag
template <class Space, class T, class... P>
  requires Kokkos::is_space<Space>::value
inline auto create_mirror(
    Kokkos::Impl::WithoutInitializing_t wi, const Space&,
    const Kokkos::Experimental::OffsetView<T, P...>& src) {
  return Impl::create_mirror(
      src, Kokkos::view_alloc(typename Space::memory_space{}, wi));
}

// public interface that accepts arbitrary view constructor args passed by a
// view_alloc
template <class T, class... P, class... ViewCtorArgs>
inline auto create_mirror(
    const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop,
    const Kokkos::Experimental::OffsetView<T, P...>& src) {
  return Impl::create_mirror(src, arg_prop);
}

namespace Impl {

// create a mirror view
// private interface that accepts arbitrary view constructor args passed by a
// view_alloc
template <class T, class... P, class... ViewCtorArgs>
inline auto create_mirror_view(
    const Kokkos::Experimental::OffsetView<T, P...>& src,
    [[maybe_unused]] const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop) {
  if constexpr (!Impl::ViewCtorProp<ViewCtorArgs...>::has_memory_space) {
    if constexpr (std::is_same_v<
                      typename Kokkos::Experimental::OffsetView<
                          T, P...>::memory_space,
                      typename Kokkos::Experimental::OffsetView<
                          T, P...>::host_mirror_type::memory_space> &&
                  std::is_same_v<typename Kokkos::Experimental::OffsetView<
                                     T, P...>::data_type,
                                 typename Kokkos::Experimental::OffsetView<
                                     T, P...>::host_mirror_type::data_type>) {
      return
          typename Kokkos::Experimental::OffsetView<T, P...>::host_mirror_type(
              src);
    } else {
      return Kokkos::Impl::create_mirror(src, arg_prop);
    }
  } else {
    if constexpr (Impl::MirrorOffsetViewType<typename Impl::ViewCtorProp<
                                                 ViewCtorArgs...>::memory_space,
                                             T, P...>::is_same_memspace) {
      return typename Impl::MirrorOffsetViewType<
          typename Impl::ViewCtorProp<ViewCtorArgs...>::memory_space, T,
          P...>::view_type(src);
    } else {
      return Kokkos::Impl::create_mirror(src, arg_prop);
    }
  }
}

}  // namespace Impl

// public interface
template <class T, class... P>
inline auto create_mirror_view(
    const typename Kokkos::Experimental::OffsetView<T, P...>& src) {
  return Impl::create_mirror_view(src, Impl::ViewCtorProp<>{});
}

// public interface that accepts a without initializing flag
template <class T, class... P>
inline auto create_mirror_view(
    Kokkos::Impl::WithoutInitializing_t wi,
    const typename Kokkos::Experimental::OffsetView<T, P...>& src) {
  return Impl::create_mirror_view(src, Kokkos::view_alloc(wi));
}

// public interface that accepts a space
template <class Space, class T, class... P>
  requires Kokkos::is_space<Space>::value
inline auto create_mirror_view(
    const Space&, const Kokkos::Experimental::OffsetView<T, P...>& src) {
  return Impl::create_mirror_view(
      src, Kokkos::view_alloc(typename Space::memory_space{}));
}

// public interface that accepts a space and a without initializing flag
template <class Space, class T, class... P>
  requires Kokkos::is_space<Space>::value
inline auto create_mirror_view(
    Kokkos::Impl::WithoutInitializing_t wi, const Space&,
    const Kokkos::Experimental::OffsetView<T, P...>& src) {
  return Impl::create_mirror_view(
      src, Kokkos::view_alloc(typename Space::memory_space{}, wi));
}

// public interface that accepts arbitrary view constructor args passed by a
// view_alloc
template <class T, class... P, class... ViewCtorArgs>
inline auto create_mirror_view(
    const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop,
    const Kokkos::Experimental::OffsetView<T, P...>& src) {
  return Impl::create_mirror_view(src, arg_prop);
}

// create a mirror view and deep copy it
// public interface that accepts arbitrary view constructor args passed by a
// view_alloc
template <class... ViewCtorArgs, class T, class... P>
typename Kokkos::Impl::MirrorOffsetViewType<
    typename Impl::ViewCtorProp<ViewCtorArgs...>::memory_space, T,
    P...>::view_type
create_mirror_view_and_copy(
    const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop,
    const Kokkos::Experimental::OffsetView<T, P...>& src) {
  return {create_mirror_view_and_copy(arg_prop, src.view()), src.begins()};
}

template <class Space, class T, class... P>
  requires Kokkos::is_space<Space>::value
typename Kokkos::Impl::MirrorOffsetViewType<Space, T, P...>::view_type
create_mirror_view_and_copy(
    const Space& space, const Kokkos::Experimental::OffsetView<T, P...>& src,
    std::string const& name = "") {
  return {create_mirror_view_and_copy(space, src.view(), name), src.begins()};
}

template <class T, class... P>
auto create_mirror_view_and_copy(
    const Kokkos::Experimental::OffsetView<T, P...>& src) {
  return create_mirror_view_and_copy(
      typename Kokkos::Experimental::OffsetView<
          T, P...>::host_mirror_type::memory_space{},
      src);
}

} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#ifdef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_OFFSETVIEW
#undef KOKKOS_IMPL_PUBLIC_INCLUDE
#undef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_OFFSETVIEW
#endif
#endif /* KOKKOS_OFFSETVIEW_HPP_ */
