#include <utility>
#include <Kokkos_Pair.hpp>
#include <impl/Kokkos_AllTag.hpp>

namespace Kokkos {

template <class T, class Extents, class Layout, class Accessor, class... Args>
auto submdspan(
    Layout,
    const std::experimental::mdspan<T, Extents, Layout, Accessor>& A,
    const Args&... args) {
  return submdspan(A, args...);
}

namespace Impl {


template<class T1, class T2>
std::pair<T1,T2> convert_subview_args(const Kokkos::pair<T1,T2>& v) { return reinterpret_cast<const std::pair<T1,T2>&>(v); }
inline std::experimental::full_extent_t convert_subview_args(const Kokkos::Impl::ALL_t&) { return std::experimental::full_extent; }
template<class T>
const T& convert_subview_args(const T& v) { return v; }

template <class T>
struct is_integral_extent_type {
  enum : bool {
    value = std::is_same<T, std::experimental::full_extent_t>::value ? 1 : 0
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
              (std::is_same<Arg, std::experimental::full_extent_t>::value)) ||
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
              (std::is_same<Arg, std::experimental::full_extent_t>::value))) &&
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
             (std::is_same<Arg, std::experimental::full_extent_t>::value))
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
  using Kokkos::submdspan;
  using std::experimental::submdspan;
template <class T, class Extents, class... Args>
struct DeduceSubSpanExtents {
  using generic_mdspan_type =
      std::experimental::mdspan<T, Extents,
                                      std::experimental::layout_left,
                                      std::experimental::default_accessor<T>>;
  using generic_submdspan_type = decltype(submdspan(
      std::declval<generic_mdspan_type>(), Kokkos::Impl::convert_subview_args(std::declval<Args>())...));
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
struct FirstArgIsRange<std::experimental::full_extent_t, Args...> : std::true_type {
};

template <class... Args>
struct LastArgIsRange;

template <class T1, class... Args>
struct LastArgIsRange<T1, Args...> : LastArgIsRange<Args...> {};

template <class T1, class T2>
struct LastArgIsRange<Kokkos::pair<T1, T2>> : std::true_type {};

template <class T1, class T2>
struct LastArgIsRange<std::pair<T1, T2>> : std::true_type {};

template <>
struct LastArgIsRange<std::experimental::full_extent_t> : std::true_type {
};

template<class T>
struct LastArgIsRange<T> : std::false_type {};

// This converts the stride arguments into constructor arguments for the
// the new subspan, by replacing them one by one in create call.
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
      std::is_same<Arg, std::experimental::full_extent_t>::value &&
          sub_mdspan_type::static_extent(Rsub) == -1,
      sub_mdspan_type>
  create(ptrdiff_t offset, mdspan_type org, Arg, Args... args) {
    return ConstructSubSpan<mdspan_type, sub_mdspan_type, R + 1,
                            Rsub + 1>::create(offset, org, args...,
                                              org.extent(R));
  }
  template <class Arg, class... Args>
  KOKKOS_INLINE_FUNCTION static std::enable_if_t<
      std::is_same<Arg, std::experimental::full_extent_t>::value &&
          sub_mdspan_type::static_extent(Rsub) != -1,
      sub_mdspan_type>
  create(ptrdiff_t offset, mdspan_type org, Arg, Args... args) {
    return ConstructSubSpan<mdspan_type, sub_mdspan_type, R + 1,
                            Rsub + 1>::create(offset, org, args...);
  }
  template <class T1, class T2, class... Args>
  KOKKOS_INLINE_FUNCTION static sub_mdspan_type
  create(ptrdiff_t offset, mdspan_type org, std::pair<T1,T2> arg, Args... args) {
    return ConstructSubSpan<mdspan_type, sub_mdspan_type, R + 1,
                            Rsub + 1>::create(offset, org, args...,
                                              arg.second-arg.first);
  }

  // Overloads where the target layout only needs dynamic extents to be created
  template <class Arg, class... Args>
  KOKKOS_INLINE_FUNCTION static std::enable_if_t<std::is_integral<Arg>::value,
                                                 sub_mdspan_type>
  create(ptrdiff_t stride, ptrdiff_t offset, mdspan_type org, Arg, Args... args) {
    if(R<mdspan_type::rank()-1) stride *= org.extent(R);
    printf("%i %i %li\n",R,Rsub,stride);
    return ConstructSubSpan<mdspan_type, sub_mdspan_type, R + 1, Rsub>::create(
        stride, offset, org, args...);
  }
  template <class Arg, class... Args>
  KOKKOS_INLINE_FUNCTION static std::enable_if_t<
      std::is_same<Arg, std::experimental::full_extent_t>::value &&
          sub_mdspan_type::static_extent(Rsub) == -1,
      sub_mdspan_type>
  create(ptrdiff_t stride, ptrdiff_t offset, mdspan_type org, Arg, Args... args) {
  if(R<mdspan_type::rank()-1) stride=1;
    printf("%i %i %li\n",R,Rsub,stride);
    return ConstructSubSpan<mdspan_type, sub_mdspan_type, R + 1,
                            Rsub + 1>::create(stride, offset, org, args...,
                                              org.extent(R));
  }
  template <class Arg, class... Args>
  KOKKOS_INLINE_FUNCTION static std::enable_if_t<
      std::is_same<Arg, std::experimental::full_extent_t>::value &&
          sub_mdspan_type::static_extent(Rsub) != -1,
      sub_mdspan_type>
  create(ptrdiff_t stride, ptrdiff_t offset, mdspan_type org, Arg, Args... args) {
  if(R<mdspan_type::rank()-1) stride=1;
    printf("%i %i %li\n",R,Rsub,stride);
    return ConstructSubSpan<mdspan_type, sub_mdspan_type, R + 1,
                            Rsub + 1>::create(stride, offset, org, args...);
  }

  template <class T1, class T2, class... Args>
  KOKKOS_INLINE_FUNCTION static sub_mdspan_type
  create(ptrdiff_t& stride, ptrdiff_t offset, mdspan_type org, std::pair<T1,T2> arg, Args... args) {
    return ConstructSubSpan<mdspan_type, sub_mdspan_type, R + 1,
                            Rsub + 1>::create(stride, offset, org, args...,
                                              arg.second-arg.first);
  }

  // Overloads where the target layout needs dynamic extents and the strides
  using strides_t = std::array<ptrdiff_t,sub_mdspan_type::rank()>;

  template <class Arg, class... Args>
  KOKKOS_INLINE_FUNCTION static std::enable_if_t<std::is_integral<Arg>::value,
                                                 sub_mdspan_type>
  create(strides_t& strides, ptrdiff_t offset, mdspan_type org, Arg, Args... args) {
    printf("Strides Skip: %i %i %i\n",R,Rsub,int(org.stride(R)));
    return ConstructSubSpan<mdspan_type, sub_mdspan_type, R + 1, Rsub>::create(
        strides, offset, org, args...);
  }
  template <class Arg, class... Args>
  KOKKOS_INLINE_FUNCTION static std::enable_if_t<
      std::is_same<Arg, std::experimental::full_extent_t>::value &&
          sub_mdspan_type::static_extent(Rsub) == std::experimental::dynamic_extent,
      sub_mdspan_type>
  create(strides_t& strides, ptrdiff_t offset, mdspan_type org, Arg, Args... args) {
    printf("Strides: %i %i %i\n",R,Rsub,int(org.stride(R)));
    strides[Rsub] = org.stride(R);
    return ConstructSubSpan<mdspan_type, sub_mdspan_type, R + 1,
                            Rsub + 1>::create(strides, offset, org, args...,
                                              org.extent(R));
  }
  template <class Arg, class... Args>
  KOKKOS_INLINE_FUNCTION static std::enable_if_t<
      std::is_same<Arg, std::experimental::full_extent_t>::value &&
          sub_mdspan_type::static_extent(Rsub) != std::experimental::dynamic_extent,
      sub_mdspan_type>
  create(strides_t& strides, ptrdiff_t offset, mdspan_type org, Arg, Args... args) {
    strides[Rsub] = org.stride(R);
    return ConstructSubSpan<mdspan_type, sub_mdspan_type, R + 1,
                            Rsub + 1>::create(strides, offset, org, args...);
  }
  template <class T1, class T2, class... Args>
  KOKKOS_INLINE_FUNCTION static sub_mdspan_type
  create(strides_t& strides, ptrdiff_t offset, mdspan_type org, std::pair<T1,T2> arg, Args... args) {
    printf("Strides: %i %i %i\n",R,Rsub,int(org.stride(R)));
    strides[Rsub] = org.stride(R);
    return ConstructSubSpan<mdspan_type, sub_mdspan_type, R + 1,
                            Rsub + 1>::create(strides, offset, org, args...,
                                              arg.second-arg.first);
  }
};

template <class mdspan_type, class sub_mdspan_type>
struct ConstructSubSpan<mdspan_type, sub_mdspan_type, int(mdspan_type::rank()),
                        int(sub_mdspan_type::rank())> {

  using layout_t = typename sub_mdspan_type::layout_type;
  using extents_t = typename sub_mdspan_type::extents_type;
  using mapping_t = typename layout_t::template mapping<extents_t>;
  // Overload where the target layout needs dynamic extents only (layout_left/layout_right)
  template <class... Args>
  KOKKOS_INLINE_FUNCTION static sub_mdspan_type create(ptrdiff_t offset,
                                                       mdspan_type org,
                                                       Args... args) {
    return sub_mdspan_type(org.accessor().offset(org.data(), offset), mapping_t(extents_t(args...)));
  }

  // Overload where the target layout needs dynamic extents and a single stride (LayoutLeft/LayoutRight)
  template <class... Args>
  KOKKOS_INLINE_FUNCTION static sub_mdspan_type create(ptrdiff_t stride,
                                                       ptrdiff_t offset,
                                                       mdspan_type org,
                                                       Args... args) {
    return sub_mdspan_type(org.accessor().offset(org.data(), offset), mapping_t(extents_t(args...), size_t(stride)*org.mapping().stride(1)));
  }

  // Overloads where the target layout needs dynamic extents and the strides (LayoutStride,layout_stride)
  using strides_t = std::array<ptrdiff_t,sub_mdspan_type::rank()>;
  template <class... Args>
  KOKKOS_INLINE_FUNCTION static sub_mdspan_type create(strides_t& strides, ptrdiff_t offset,
                                                       mdspan_type org,
                                                       Args... args) {
    return sub_mdspan_type(org.accessor().offset(org.data(), offset), mapping_t(extents_t(args...),strides));
  }
};

template<class mdspan_type, class TSub, class ExtSub, class AccSub>
struct ConstructSubSpan<mdspan_type, std::experimental::mdspan<TSub,ExtSub,Kokkos::LayoutStride,AccSub>, -1, -1> {
  using sub_mdspan_type = std::experimental::mdspan<TSub,ExtSub,Kokkos::LayoutStride,AccSub>;
  template <class ... Args>
  KOKKOS_INLINE_FUNCTION static sub_mdspan_type create(
   ptrdiff_t offset,
   mdspan_type org, Args... args) {
     std::array<ptrdiff_t,sub_mdspan_type::rank()> strides{};
     return ConstructSubSpan<mdspan_type, sub_mdspan_type, 0, 0>::
             create(strides, offset, org, args...);
  }
};

template<class mdspan_type, class TSub, class ExtSub, class AccSub>
struct ConstructSubSpan<mdspan_type, std::experimental::mdspan<TSub,ExtSub,Kokkos::LayoutLeft,AccSub>, -1, -1> {
  using sub_mdspan_type = std::experimental::mdspan<TSub,ExtSub,Kokkos::LayoutLeft,AccSub>;

  template <class ... Args>
  KOKKOS_INLINE_FUNCTION static std::enable_if_t<
    !(sub_mdspan_type::extents_type::rank() <=2 &&
      Impl::FirstArgIsRange<Args...>::value),
          sub_mdspan_type> create(
   ptrdiff_t offset,
   mdspan_type org, Args... args) {
     return ConstructSubSpan<mdspan_type, sub_mdspan_type, 0, 0>::
             create(offset, org, args...);
  }

  template <class ... Args>
  KOKKOS_INLINE_FUNCTION static std::enable_if_t<
     (sub_mdspan_type::extents_type::rank() <=2 &&
      Impl::FirstArgIsRange<Args...>::value),
          sub_mdspan_type> create(
   ptrdiff_t offset,
   mdspan_type org, Args... args) {
     ptrdiff_t stride = 1;
     return ConstructSubSpan<mdspan_type, sub_mdspan_type, 0, 0>::
             create(stride, offset, org, args...);
  }
};

template<class mdspan_type, class sub_mdspan_type>
struct ConstructSubSpan<mdspan_type, sub_mdspan_type, -1, -1> {
  template <class ... Args>
  KOKKOS_INLINE_FUNCTION static sub_mdspan_type create(
   ptrdiff_t offset,
   mdspan_type org, Args... args) {
     ptrdiff_t stride = 1;
     return ConstructSubSpan<mdspan_type, sub_mdspan_type, 0, 0>::
             create(stride, offset, org, args...);
  }
};

// Computes offset based on submdspan arguments and the original view
// by isolating each offset index in each dimension, and then feed
// that at the bottom into the mapping of the original view to compute
// the offset.
// For example submdspan(mdspan, 5, make_pair<3,2>, full_extent, 2) will call
// mapping(5,3,0,2) to compute the offset. 

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
      std::is_same<Arg, std::experimental::full_extent_t>::value, ptrdiff_t>
  compute(mdspan_type org, Arg, Args... args) {
    return ComputeSubSpanOffset<mdspan_type, R + 1>::compute(org, args...,
                                                             ptrdiff_t(0));
  }
  template <class T1, class T2, class... Args>
  KOKKOS_INLINE_FUNCTION static ptrdiff_t
  compute(mdspan_type org, const std::pair<T1,T2>& range, Args... args) {
    return ComputeSubSpanOffset<mdspan_type, R + 1>::compute(org, args...,
                                                             ptrdiff_t(range.first));
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
auto submdspan(Kokkos::LayoutLeft,
             const std::experimental::mdspan<
                 T, Extents, Kokkos::LayoutLeft, Accessor>& A,
             const Args&... args) {
        printf("HUCH\n");
  using mdspan_type =
      std::experimental::mdspan<T, Extents, Kokkos::LayoutLeft, Accessor>;
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
      std::experimental::mdspan<T, sub_extents_type, sub_array_layout,
                                      Accessor>;

  return Impl::ConstructSubSpan<mdspan_type, sub_mdspan_type, -1, -1>::create(
      Impl::ComputeSubSpanOffset<mdspan_type, 0>::compute(A, Kokkos::Impl::convert_subview_args(args)...), A,
      Kokkos::Impl::convert_subview_args(args)...);
}



template <class T, class Extents, class Accessor, class... Args>
auto submdspan(Kokkos::LayoutRight,
             const std::experimental::mdspan<
                 T, Extents, Kokkos::LayoutRight, Accessor>& A,
             const Args&... args) {
        printf("HUCH\n");
  using mdspan_type =
      std::experimental::mdspan<T, Extents, Kokkos::LayoutRight, Accessor>;
  using sub_extents_type =
      typename Impl::DeduceSubSpanExtents<T, Extents, Args...>::extents_type;
  using sub_array_layout = typename std::conditional<
      (                                /* Same array layout IF */
       (sub_extents_type::rank() == 0) /* output rank zero */
       || Impl::SubviewLegalArgsCompileTime<
              Kokkos::LayoutRight, Kokkos::LayoutRight, sub_extents_type::rank(),
              Extents::rank(), 0, Args...>::value ||
       (sub_extents_type::rank() <= 2 &&
        Impl::LastArgIsRange<Args...>::value)),
      Kokkos::LayoutRight, Kokkos::LayoutStride>::type;
  using sub_mdspan_type =
      std::experimental::mdspan<T, sub_extents_type, sub_array_layout,
                                      Accessor>;

  return Impl::ConstructSubSpan<mdspan_type, sub_mdspan_type, -1, -1>::create(
      Impl::ComputeSubSpanOffset<mdspan_type, 0>::compute(A, Kokkos::Impl::convert_subview_args(args)...), A,
      Kokkos::Impl::convert_subview_args(args)...);
}

template<class T, class Extents, class Accessor, class... Args>
auto submdspan(const std::experimental::mdspan<
                  T, Extents, Kokkos::LayoutRight, Accessor>& A,
               const Args&... args) {
   return submdspan(Kokkos::LayoutRight(), A, args...);
}

template<class T, class Extents, class Accessor, class... Args>
auto submdspan(const std::experimental::mdspan<
                  T, Extents, Kokkos::LayoutLeft, Accessor>& A,
               const Args&... args) {
   return submdspan(Kokkos::LayoutLeft(), A, args...);
}
}  // namespace Kokkos
