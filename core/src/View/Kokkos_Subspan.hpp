#include <utility>
#include <Kokkos_Pair.hpp>

namespace Kokkos {
template <class T, class Extents, class Layout, class Accessor, class... Args>
auto subspan(
    Layout,
    const std::experimental::basic_mdspan<T, Extents, Layout, Accessor>& A,
    const Args&... args) {
  return subspan(A, args...);
}

namespace Impl {
template <class T>
struct is_integral_extent_type {
  enum : bool {
    value = std::is_same<T, std::experimental::all_type>::value ? 1 : 0
  };
};

template <class iType>
struct is_integral_extent_type<std::pair<iType, iType>> {
  enum : bool { value = std::is_integral<iType>::value ? 1 : 0 };
};

template <class iType>
struct is_integral_extent_type<Kokkos::pair<iType, iType>> {
  enum : bool { value = std::is_integral<iType>::value ? 1 : 0 };
};

// Assuming '2 == initializer_list<iType>::size()'
template <class iType>
struct is_integral_extent_type<std::initializer_list<iType>> {
  enum : bool { value = std::is_integral<iType>::value ? 1 : 0 };
};

template <unsigned I, class... Args>
struct is_integral_extent {
  // get_type is void when sizeof...(Args) <= I
  using type = typename std::remove_cv<typename std::remove_reference<
      typename Kokkos::Impl::get_type<I, Args...>::type>::type>::type;

  enum : bool { value = is_integral_extent_type<type>::value };

  static_assert(value || std::is_integral<type>::value ||
                    std::is_same<type, void>::value,
                "subview argument must be either integral or integral extent");
};

// Rules for subview arguments and layouts matching

template <class LayoutDest, class LayoutSrc, int RankDest, int RankSrc,
          int CurrentArg, class... SubViewArgs>
struct SubviewLegalArgsCompileTime;

// Rules which allow LayoutLeft to LayoutLeft assignment

template <int RankDest, int RankSrc, int CurrentArg, class Arg,
          class... SubViewArgs>
struct SubviewLegalArgsCompileTime<Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                   RankDest, RankSrc, CurrentArg, Arg,
                                   SubViewArgs...> {
  enum {
    value = (((CurrentArg == RankDest - 1) &&
              (Kokkos::Impl::is_integral_extent_type<Arg>::value)) ||
             ((CurrentArg >= RankDest) && (std::is_integral<Arg>::value)) ||
             ((CurrentArg < RankDest) &&
              (std::is_same<Arg, std::experimental::all_type>::value)) ||
             ((CurrentArg == 0) &&
              (Kokkos::Impl::is_integral_extent_type<Arg>::value))) &&
            (SubviewLegalArgsCompileTime<Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                         RankDest, RankSrc, CurrentArg + 1,
                                         SubViewArgs...>::value)
  };
};

template <int RankDest, int RankSrc, int CurrentArg, class Arg>
struct SubviewLegalArgsCompileTime<Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                   RankDest, RankSrc, CurrentArg, Arg> {
  enum {
    value = ((CurrentArg == RankDest - 1) || (std::is_integral<Arg>::value)) &&
            (CurrentArg == RankSrc - 1)
  };
};

// Rules which allow LayoutRight to LayoutRight assignment

template <int RankDest, int RankSrc, int CurrentArg, class Arg,
          class... SubViewArgs>
struct SubviewLegalArgsCompileTime<Kokkos::LayoutRight, Kokkos::LayoutRight,
                                   RankDest, RankSrc, CurrentArg, Arg,
                                   SubViewArgs...> {
  enum {
    value = (((CurrentArg == RankSrc - RankDest) &&
              (Kokkos::Impl::is_integral_extent_type<Arg>::value)) ||
             ((CurrentArg < RankSrc - RankDest) &&
              (std::is_integral<Arg>::value)) ||
             ((CurrentArg >= RankSrc - RankDest) &&
              (std::is_same<Arg, std::experimental::all_type>::value))) &&
            (SubviewLegalArgsCompileTime<Kokkos::LayoutRight,
                                         Kokkos::LayoutRight, RankDest, RankSrc,
                                         CurrentArg + 1, SubViewArgs...>::value)
  };
};

template <int RankDest, int RankSrc, int CurrentArg, class Arg>
struct SubviewLegalArgsCompileTime<Kokkos::LayoutRight, Kokkos::LayoutRight,
                                   RankDest, RankSrc, CurrentArg, Arg> {
  enum {
    value = ((CurrentArg == RankSrc - 1) &&
             (std::is_same<Arg, std::experimental::all_type>::value))
  };
};

// Rules which allow assignment to LayoutStride

template <int RankDest, int RankSrc, int CurrentArg, class... SubViewArgs>
struct SubviewLegalArgsCompileTime<Kokkos::LayoutStride, Kokkos::LayoutLeft,
                                   RankDest, RankSrc, CurrentArg,
                                   SubViewArgs...> {
  enum : bool { value = true };
};

template <int RankDest, int RankSrc, int CurrentArg, class... SubViewArgs>
struct SubviewLegalArgsCompileTime<Kokkos::LayoutStride, Kokkos::LayoutRight,
                                   RankDest, RankSrc, CurrentArg,
                                   SubViewArgs...> {
  enum : bool { value = true };
};

template <int RankDest, int RankSrc, int CurrentArg, class... SubViewArgs>
struct SubviewLegalArgsCompileTime<Kokkos::LayoutStride, Kokkos::LayoutStride,
                                   RankDest, RankSrc, CurrentArg,
                                   SubViewArgs...> {
  enum : bool { value = true };
};

// Subspan Extents are actually independent of Layout and Accessor
template <class T, class Extents, class... Args>
struct DeduceSubSpanExtents {
  using generic_mdspan_type =
      std::experimental::basic_mdspan<T, Extents,
                                      std::experimental::layout_left,
                                      std::experimental::accessor_basic<T>>;
  using generic_submdspan_type = decltype(std::experimental::subspan(
      std::declval<generic_mdspan_type>(), std::declval<Args>()...));
  using extents_type           = typename generic_submdspan_type::extents_type;
};

// Helpers to figure out whether an argument is a range
template <class... Args>
struct FirstArgIsRange : std::false_type {};

template <class T1, class T2, class... Args>
struct FirstArgIsRange<std::pair<T1, T2>, Args...> : std::true_type {};

template <class T1, class T2, class... Args>
struct FirstArgIsRange<Kokkos::pair<T1, T2>, Args...> : std::true_type {};

template <class... Args>
struct FirstArgIsRange<std::experimental::all_type, Args...> : std::true_type {
};

template <class mdspan_type, class sub_mdspan_type, int R, int Rsub>
struct ConstructSubSpan {
  // Overloads where the target layout only needs dynamic extents to be created
  template <class Arg, class... Args>
  KOKKOS_INLINE_FUNCTION static std::enable_if_t<std::is_integral<Arg>::value,
                                                 sub_mdspan_type>
  create(ptrdiff_t offset, mdspan_type org, Arg, Args... args) {
    return ConstructSubSpan<mdspan_type, sub_mdspan_type, R + 1, Rsub>::create(
        offset, org, args...);
  }
  template <class Arg, class... Args>
  KOKKOS_INLINE_FUNCTION static std::enable_if_t<
      std::is_same<Arg, std::experimental::all_type>::value &&
          sub_mdspan_type::static_extent(Rsub) == -1,
      sub_mdspan_type>
  create(ptrdiff_t offset, mdspan_type org, Arg, Args... args) {
    return ConstructSubSpan<mdspan_type, sub_mdspan_type, R + 1,
                            Rsub + 1>::create(offset, org, args...,
                                              org.extent(R));
  }
  template <class Arg, class... Args>
  KOKKOS_INLINE_FUNCTION static std::enable_if_t<
      std::is_same<Arg, std::experimental::all_type>::value &&
          sub_mdspan_type::static_extent(Rsub) != -1,
      sub_mdspan_type>
  create(ptrdiff_t offset, mdspan_type org, Arg, Args... args) {
    return ConstructSubSpan<mdspan_type, sub_mdspan_type, R + 1,
                            Rsub + 1>::create(offset, org, args...);
  }

  // Overloads where the target layout needs dynamic extents and the strides
  using strides_t = std::array<ptrdiff_t,sub_mdspan_type::rank()>;

  template <class Arg, class... Args>
  KOKKOS_INLINE_FUNCTION static std::enable_if_t<std::is_integral<Arg>::value,
                                                 sub_mdspan_type>
  create(strides_t& strides, ptrdiff_t offset, mdspan_type org, Arg, Args... args) {
    return ConstructSubSpan<mdspan_type, sub_mdspan_type, R + 1, Rsub>::create(
        strides, offset, org, args...);
  }
  template <class Arg, class... Args>
  KOKKOS_INLINE_FUNCTION static std::enable_if_t<
      std::is_same<Arg, std::experimental::all_type>::value &&
          sub_mdspan_type::static_extent(Rsub) == -1,
      sub_mdspan_type>
  create(strides_t& strides, ptrdiff_t offset, mdspan_type org, Arg, Args... args) {
    strides[Rsub] = org.stride(R);
    return ConstructSubSpan<mdspan_type, sub_mdspan_type, R + 1,
                            Rsub + 1>::create(strides, offset, org, args...,
                                              org.extent(R));
  }
  template <class Arg, class... Args>
  KOKKOS_INLINE_FUNCTION static std::enable_if_t<
      std::is_same<Arg, std::experimental::all_type>::value &&
          sub_mdspan_type::static_extent(Rsub) != -1,
      sub_mdspan_type>
  create(strides_t& strides, ptrdiff_t offset, mdspan_type org, Arg, Args... args) {
    strides[Rsub] = org.stride(R);
    return ConstructSubSpan<mdspan_type, sub_mdspan_type, R + 1,
                            Rsub + 1>::create(strides, offset, org, args...);
  }
};

template <class mdspan_type, class sub_mdspan_type>
struct ConstructSubSpan<mdspan_type, sub_mdspan_type, int(mdspan_type::rank()),
                        int(sub_mdspan_type::rank())> {

  template <class... Args>
  KOKKOS_INLINE_FUNCTION static sub_mdspan_type create(ptrdiff_t offset,
                                                       mdspan_type org,
                                                       Args... args) {
    return sub_mdspan_type(org.accessor().offset(org.data(), offset), args...);
  }

  // Overloads where the target layout needs dynamic extents and the strides
  using strides_t = std::array<ptrdiff_t,sub_mdspan_type::rank()>;
  using layout_t = typename sub_mdspan_type::layout_type;
  using extents_t = typename sub_mdspan_type::extents_type;
  using mapping_t = typename layout_t::template mapping<extents_t>;
  template <class... Args>
  KOKKOS_INLINE_FUNCTION static sub_mdspan_type create(strides_t& strides, ptrdiff_t offset,
                                                       mdspan_type org,
                                                       Args... args) {
    return sub_mdspan_type(org.accessor().offset(org.data(), offset), mapping_t(extents_t(args...),strides));
  }
};

template<class mdspan_type, class TSub, class ExtSub, class AccSub>
struct ConstructSubSpan<mdspan_type, std::experimental::basic_mdspan<TSub,ExtSub,Kokkos::LayoutStride,AccSub>, -1, -1> {
  using sub_mdspan_type = std::experimental::basic_mdspan<TSub,ExtSub,Kokkos::LayoutStride,AccSub>;
  template <class ... Args>
  KOKKOS_INLINE_FUNCTION static sub_mdspan_type create(
   ptrdiff_t offset,
   mdspan_type org, Args... args) {
     std::array<ptrdiff_t,sub_mdspan_type::rank()> strides{};
     return ConstructSubSpan<mdspan_type, sub_mdspan_type, 0, 0>::
             create(strides, offset, org, args...);
  }
};

template<class mdspan_type, class sub_mdspan_type>
struct ConstructSubSpan<mdspan_type, sub_mdspan_type, -1, -1> {
  template <class ... Args>
  KOKKOS_INLINE_FUNCTION static sub_mdspan_type create(
   ptrdiff_t offset,
   mdspan_type org, Args... args) {
     return ConstructSubSpan<mdspan_type, sub_mdspan_type, 0, 0>::
             create(offset, org, args...);
  }
};

template <class mdspan_type, int R>
struct ComputeSubSpanOffset {
  template <class Arg, class... Args>
  KOKKOS_INLINE_FUNCTION static std::enable_if_t<std::is_integral<Arg>::value,
                                                 ptrdiff_t>
  compute(mdspan_type org, Arg idx, Args... args) {
    return ComputeSubSpanOffset<mdspan_type, R + 1>::compute(org, args..., idx);
  }
  template <class Arg, class... Args>
  KOKKOS_INLINE_FUNCTION static std::enable_if_t<
      std::is_same<Arg, std::experimental::all_type>::value, ptrdiff_t>
  compute(mdspan_type org, Arg, Args... args) {
    return ComputeSubSpanOffset<mdspan_type, R + 1>::compute(org, args...,
                                                             ptrdiff_t(0));
  }
};

template <class mdspan_type>
struct ComputeSubSpanOffset<mdspan_type, int(mdspan_type::rank())> {
  template <class... Args>
  KOKKOS_INLINE_FUNCTION static ptrdiff_t compute(mdspan_type org,
                                                  Args... args) {
    return org.mapping()(args...);
  }
};

}  // namespace Impl

template <class T, class Extents, class Accessor, class... Args>
auto subspan(Kokkos::LayoutLeft,
             const std::experimental::basic_mdspan<
                 T, Extents, Kokkos::LayoutLeft, Accessor>& A,
             const Args&... args) {
  using mdspan_type =
      std::experimental::basic_mdspan<T, Extents, Kokkos::LayoutLeft, Accessor>;
  using sub_extents_type =
      typename Impl::DeduceSubSpanExtents<T, Extents, Args...>::extents_type;
  using sub_array_layout = typename std::conditional<
      (                                /* Same array layout IF */
       (sub_extents_type::rank() == 0) /* output rank zero */
       || Impl::SubviewLegalArgsCompileTime<
              Kokkos::LayoutLeft, Kokkos::LayoutLeft, sub_extents_type::rank(),
              Extents::rank(), 0, Args...>::value ||
       (sub_extents_type::rank() <= 2 &&
        Impl::FirstArgIsRange<Args...>::value)),
      Kokkos::LayoutLeft, Kokkos::LayoutStride>::type;
  using sub_mdspan_type =
      std::experimental::basic_mdspan<T, sub_extents_type, sub_array_layout,
                                      Accessor>;

  return Impl::ConstructSubSpan<mdspan_type, sub_mdspan_type, -1, -1>::create(
      Impl::ComputeSubSpanOffset<mdspan_type, 0>::compute(A, args...), A,
      args...);
}
}  // namespace Kokkos
