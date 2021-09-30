
namespace Kokkos {
namespace Impl {
template <class ViewType, class SubMDSpanType>
struct SubViewTypeDeduction {
  using layout_type   = typename SubMDSpanType::layout_type;
  using device_type   = typename ViewType::device_type;
  using memory_traits = typename ViewType::memory_traits;
  using value_type    = typename ViewType::value_type;
  using extents_type  = typename SubMDSpanType::extents_type;
  using data_type =
      typename DataTypeFromExtents<value_type, extents_type>::type;
  using type = Kokkos::View<data_type, layout_type, device_type, memory_traits>;
};


}  // namespace Impl

template <class ViewType, class... Args>
auto subview(const ViewType& v, Args... args) {
  using Kokkos::submdspan;
  using std::experimental::submdspan;
  return typename Impl::SubViewTypeDeduction<
      ViewType, decltype(submdspan(
                              typename ViewType::mdspan_type::layout_type{},
                                 v.get_mdspan(), Impl::convert_subview_args(args)...))>::type(v, args...);
}
template <class MemoryTraits, class ViewType, class... Args>
auto subview(const ViewType& v, Args... args) {
  using Kokkos::submdspan;
  using std::experimental::submdspan;
  using ViewTypeMod = Kokkos::View<typename ViewType::data_type, typename ViewType::array_layout, typename ViewType::device_type, MemoryTraits>;
  return typename Impl::SubViewTypeDeduction<
      ViewTypeMod, decltype(submdspan(
                              typename ViewTypeMod::mdspan_type::layout_type{},
                                 v.get_mdspan(), Impl::convert_subview_args(args)...))>::type(v, args...);
}
}  // namespace Kokkos
