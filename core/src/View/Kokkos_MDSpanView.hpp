/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_MDSPANVIEW_HPP
#define KOKKOS_MDSPANVIEW_HPP

#include <View/Kokkos_View_fwd.hpp>

#include <View/Kokkos_NormalizeView.hpp>
#include <View/Kokkos_ExtractExtents.hpp>
#include <View/Kokkos_MDSpanLayout.hpp>
//#include <View/Kokkos_MDSpanAccessor.hpp>
#include <View/Kokkos_ViewVerify.hpp>
#include <View/Kokkos_IsView.hpp>
#include <View/Kokkos_ViewValueFunctor.hpp>
//#include <View/Kokkos_ViewAllocation.hpp>

#include <Kokkos_ViewAlloc.hpp>
#include <Kokkos_ViewWrap.hpp>

#include <impl/Kokkos_ViewTracker.hpp>
#include <impl/Kokkos_ViewUniformType.hpp>
#include <impl/Kokkos_ViewCtor.hpp>

#include <impl/Kokkos_Utilities.hpp>  // type_list

#include <experimental/mdspan>

#if defined(KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)

#define KOKKOS_IMPL_SINK(ARG) ARG

// TODO @mdspan bounds checking in the layout
#define KOKKOS_IMPL_VIEW_OPERATOR_VERIFY(ARG) \
  /* Kokkos::Impl::view_verify_operator_bounds<memory_space> ARG; */

#else

#define KOKKOS_IMPL_SINK(ARG)

#define KOKKOS_IMPL_VIEW_OPERATOR_VERIFY(ARG)

#endif

namespace Kokkos {

// A lot of the analysis of view type traits that leads to member types of
// BasicView is useful as part of customization point constraints (like
// MDSpanAccessorFromKokkosMemoryTraits), so it's useful to have them
// instantiated before the BasicView class itself gets instantiated.
template <class BasicViewType>
struct ViewTypeTraits;

template <class DataType, class Layout, class Space, class MemoryTraits>
struct ViewTypeTraits<BasicView<DataType, Layout, Space, MemoryTraits>> {
 protected:
  //----------------------------------------------------------------------------
  // <editor-fold desc="private member types"> {{{2

  using _extracted = Impl::ExtractExtents<DataType>;

  // </editor-fold> end private member types }}}2
  //----------------------------------------------------------------------------

 public:
  //----------------------------------------------------------------------------
  // <editor-fold desc="public member types"> {{{2

  using mdspan_element_type = typename _extracted::value_type;
  using mdspan_extents_type = typename _extracted::extents_type;

  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // <editor-fold desc="Spaces, memory access, mirrors, etc."> {{{3

  using execution_space = typename Space::execution_space;
  using memory_space    = typename Space::memory_space;
  using device_type     = Kokkos::Device<execution_space, memory_space>;
  using array_layout    = Layout;
  using memory_traits   = MemoryTraits;
  using size_type       = typename memory_space::size_type;

  using host_mirror_space =
      typename Kokkos::Impl::HostMirror<Space>::Space::memory_space;
  using anonymous_device_type =
      Kokkos::Device<execution_space, Kokkos::AnonymousSpace>;

  // </editor-fold> end Spaces, memory access, mirrors, etc. }}}3
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // <editor-fold desc="Value type traits"> {{{3

  using value_type           = mdspan_element_type;
  using const_value_type     = std::add_const_t<value_type>;
  using non_const_value_type = std::remove_cv_t<value_type>;

  using scalar_array_type =
      typename Impl::DataTypeFromExtents<value_type, mdspan_extents_type>::type;
  using const_scalar_array_type =
      typename Impl::DataTypeFromExtents<const_value_type,
                                         mdspan_extents_type>::type;
  using non_const_scalar_array_type =
      typename Impl::DataTypeFromExtents<non_const_value_type,
                                         mdspan_extents_type>::type;
  using data_type           = scalar_array_type;
  using const_data_type     = const_scalar_array_type;
  using non_const_data_type = non_const_scalar_array_type;

  // </editor-fold> end Value type traits }}}3
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // <editor-fold desc="Analogous View types"> {{{3

  /** \brief  Compatible view of array of scalar types */
  using array_type =
      View<scalar_array_type, array_layout, device_type, memory_traits>;

  /** \brief  Compatible view of const data type */
  using const_type =
      View<const_data_type, array_layout, device_type, memory_traits>;

  /** \brief  Compatible view of non-const data type */
  using non_const_type =
      View<non_const_data_type, array_layout, device_type, memory_traits>;

  /** \brief  Compatible HostMirror view */
  using HostMirror = View<non_const_data_type, array_layout,
                          Device<DefaultHostExecutionSpace,
                                 typename host_mirror_space::memory_space>>;

  /** \brief  Compatible HostMirror view */
  using host_mirror_type =
      View<non_const_data_type, array_layout, host_mirror_space>;

  // </editor-fold> end Analogous View types }}}3
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // <editor-fold desc="uniform View types"> {{{3

  using uniform_type =
      Kokkos::View<data_type, array_layout, device_type, memory_traits>;
  using uniform_const_type =
      Kokkos::View<const_data_type, array_layout, device_type, memory_traits>;
  using uniform_nomemspace_type =
      Kokkos::View<data_type, array_layout, anonymous_device_type,
                   memory_traits>;
  using uniform_const_nomemspace_type =
      Kokkos::View<const_data_type, array_layout, anonymous_device_type,
                   memory_traits>;

  // </editor-fold> end uniform View types }}}3
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  // For legacy reasons:
  using specialize = void;

  // </editor-fold> end public member types }}}2
  //----------------------------------------------------------------------------
};

//==============================================================================
// <editor-fold desc="BasicView"> {{{1

template <class DataType, class Layout, class Space, class MemoryTraits>
class BasicView
    : public ViewTypeTraits<BasicView<DataType, Layout, Space, MemoryTraits>> {
 public:
  using traits =
      ViewTypeTraits<BasicView<DataType, Layout, Space, MemoryTraits>>;

  //----------------------------------------------------------------------------
  // <editor-fold desc="new member types"> {{{2

  using mdspan_layout_type  = Layout;
  using mdspan_mapping_type = typename mdspan_layout_type::template mapping<
      typename traits::mdspan_extents_type>;
  using mdspan_accessor_type =
      std::experimental::default_accessor<typename traits::value_type>;
  // typename Impl::MDSpanAccessorFromKokkosMemoryTraits<traits,
  //                                                    MemoryTraits>::type;

  using mdspan_type =
      std::experimental::mdspan<typename traits::mdspan_element_type,
                                      typename traits::mdspan_extents_type,
                                      mdspan_layout_type, mdspan_accessor_type>;

  // </editor-fold> end private member types }}}2
  //----------------------------------------------------------------------------

 protected:
  //----------------------------------------------------------------------------
  // <editor-fold desc="friends"> {{{2

  template <class>
  friend struct Impl::ViewTracker;

  template <class, class, class, class>
  friend class BasicView;

  // </editor-fold> end friends }}}2
  //----------------------------------------------------------------------------

  //----------------------------------------------------------------------------
  // <editor-fold desc="private data members"> {{{2

  mdspan_type m_data;

  using track_type = Kokkos::Impl::ViewTracker<BasicView>;
  track_type m_track;

  // </editor-fold> end private data members }}}2
  //----------------------------------------------------------------------------

 public:
  //----------------------------------------------------------------------------
  // <editor-fold desc="public member types"> {{{2

  using reference_type = typename mdspan_type::reference;
  using pointer_type   = typename mdspan_type::pointer;

  using runtime_data_type =
      typename Impl::ViewScalarToDataType<typename traits::value_type,
                                          mdspan_type::rank()>::type;
  using runtime_const_data_type =
      typename Impl::ViewScalarToDataType<typename traits::const_value_type,
                                          mdspan_type::rank()>::type;

  using uniform_runtime_type =
      Kokkos::View<runtime_data_type, typename traits::array_layout,
                   typename traits::device_type,
                   typename traits::memory_traits>;
  using uniform_runtime_const_type =
      Kokkos::View<runtime_const_data_type, typename traits::array_layout,
                   typename traits::device_type,
                   typename traits::memory_traits>;
  using uniform_runtime_nomemspace_type =
      Kokkos::View<runtime_data_type, typename traits::array_layout,
                   typename traits::anonymous_device_type,
                   typename traits::memory_traits>;
  using uniform_runtime_const_nomemspace_type =
      Kokkos::View<runtime_const_data_type, typename traits::array_layout,
                   typename traits::anonymous_device_type,
                   typename traits::memory_traits>;

  // </editor-fold> end public member types }}}2
  //----------------------------------------------------------------------------

  //----------------------------------------------------------------------------
  // <editor-fold desc="inline static data members"> {{{2

  enum : bool {
    is_hostspace = std::is_same<typename traits::memory_space, HostSpace>::value
  };
  enum : bool { is_managed = MemoryTraits::is_unmanaged == 0 };
  enum : bool { is_random_access = MemoryTraits::is_random_access == 1 };

  enum : int { rank = mdspan_type::rank() };
  enum : int { Rank = rank };  // Yay legacy code!!!
  enum : int { rank_dynamic = mdspan_type::rank_dynamic() };
  enum : bool {
    reference_type_is_lvalue_reference =
        std::is_lvalue_reference<reference_type>::value
  };

  // </editor-fold> end inline static data members }}}2
  //----------------------------------------------------------------------------

 private:
  //----------------------------------------------------------------------------
  // <editor-fold desc="private member functions"> {{{2

  template <std::size_t... Idxs>
  KOKKOS_FUNCTION constexpr typename traits::array_layout _layout_helper(
      std::true_type /* is_extent_constructible */,
      std::integer_sequence<std::size_t, Idxs...>) const {
    return typename traits::array_layout{m_data.extent(Idxs)...};
  }

  template <std::size_t... Idxs>
  KOKKOS_FUNCTION constexpr typename traits::array_layout _layout_helper(
      std::false_type /* is_extent_constructible */,
      std::integer_sequence<std::size_t, Idxs...>) const {
    // This is the best way I can think of to rifle (zip + flatten) a parameter
    // pack:
    return typename traits::array_layout{
        (Idxs % 2 == 0 ? m_data.extent(Idxs / 2) : m_data.stride(Idxs / 2))...};
  }

  template <std::size_t... Idxs, class... Args>
  KOKKOS_FORCEINLINE_FUNCTION constexpr reference_type _access_helper_impl(
      Impl::repeated_type<typename traits::size_type, Idxs>... idxs,
      Args... KOKKOS_IMPL_SINK(args)) const {
#if defined(KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)
    Impl::view_verify_operator_bounds<typename traits::memory_space>(
        m_data, idxs..., args...);
#endif
    return m_data(idxs...);
  }

  template <std::size_t... Idxs, class... Args>
  KOKKOS_FORCEINLINE_FUNCTION constexpr reference_type _access_helper(
      std::integer_sequence<std::size_t, Idxs...>, Args... args) const {
    return _access_helper_impl<Idxs...>(std::size_t(args)...);
  }

  // </editor-fold> end private member functions }}}2
  //----------------------------------------------------------------------------

 public:
  //----------------------------------------------------------------------------
  // <editor-fold desc="constructors, destructor, and assignment"> {{{2

  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // <editor-fold desc="defaulted rule of 6 special functions"> {{{3

  KOKKOS_DEFAULTED_FUNCTION ~BasicView() = default;

  KOKKOS_DEFAULTED_FUNCTION BasicView() = default;

  KOKKOS_DEFAULTED_FUNCTION BasicView(const BasicView&) = default;

  KOKKOS_DEFAULTED_FUNCTION BasicView(BasicView&&) = default;

  KOKKOS_DEFAULTED_FUNCTION BasicView& operator=(const BasicView&) = default;

  KOKKOS_DEFAULTED_FUNCTION BasicView& operator=(BasicView&&) = default;

  // </editor-fold> end defaulted rule of 6 special functions }}}3
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // <editor-fold desc="compatible View conversion"> {{{3

  template <
      class RT, class RL, class RS, class RMP,
      //----------------------------------------
      /* requires
       *   std::is_convertible_v<
       *     BasicView<RT, RL, RS, RMP>::mdspan_type,
       *     mdspan_type>
       */
      std::enable_if_t<
          std::is_convertible<typename BasicView<RT, RL, RS, RMP>::mdspan_type,
                              mdspan_type>::value,
          int> = 0
      //----------------------------------------
      >
  KOKKOS_FUNCTION BasicView(BasicView<RT, RL, RS, RMP> const& rhs)
      : m_data(rhs.m_data), m_track(rhs) {
    // TODO @mdspan check runtime dimension compatibility (check in the mapping)
  }

  // TODO @mdspan does this need to be SFINAE, or should we just static assert?
  template <class RT, class RL, class RS, class RMP,
            //----------------------------------------
            /* requires
             *   std::is_assignable_v<
             *     mdspan_type,
             *     BasicView<RT, RL, RS, RMP>::mdspan_type
             *   >
             */
            std::enable_if_t<
                std::is_assignable<
                    mdspan_type,
                    typename BasicView<RT, RL, RS, RMP>::mdspan_type>::value,
                int> = 0
            //----------------------------------------
            >
  KOKKOS_FUNCTION BasicView& operator=(BasicView<RT, RL, RS, RMP> const& rhs) {
    m_data = rhs.m_data;
    m_track.assign(rhs);
    // TODO @mdspan check runtime dimension compatibility!!!
    return *this;
  }

  template <class... Args>
  KOKKOS_FUNCTION BasicView(std::experimental::mdspan<Args...> const& rhs,
                            const track_type& track = track_type())
      : m_data(rhs), m_track(track) {}

  // </editor-fold> end compatible View conversion }}}3
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  template <class RT, class RL, class RS, class RMP, class Arg, class... Args>
  KOKKOS_FUNCTION BasicView(const BasicView<RT, RL, RS, RMP>& rhs, Arg first_arg, Args... args)
  //: m_data(submdspan(rhs.get_mdspan(),args...)),m_track(rhs) {}
  {
    m_data = Kokkos::submdspan(
        typename BasicView<RT, RL, RS, RMP>::mdspan_type::layout_type(),
        // TODO: why do I need to conver here for
        // Kokkos::View<int***,Kokkos::LayoutRight>(a_org,5,7,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        // This works without it: Kokkos::View<int***,Kokkos::LayoutRight>(a_org,5,7,stdex::full_extent,stdex::full_extent,stdex::full_extent);
        rhs.get_mdspan(), Kokkos::Impl::convert_subview_args(first_arg), Kokkos::Impl::convert_subview_args(args)...);
    m_track.assign(rhs);
  }

  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // <editor-fold desc="View Ctor Prop constructors"> {{{3

  // Readability note: almost all of the Kokkos constructors delegate to
  // one of these first two constructors

  // The view_wrap() and layout version
  template <class... P,
            //----------------------------------------
            /* requires
             *   Impl::ViewConstructorDescription<P...>::has_pointer
             */
            std::enable_if_t<
                Impl::ViewConstructorDescription<P...>::has_pointer, int> = 0
            //----------------------------------------
            >
  inline BasicView(Impl::ViewConstructorDescription<P...> const& arg_desc,
                   mdspan_mapping_type const& arg_mapping)
      : m_data(arg_desc.get_pointer(), mdspan_mapping_type{arg_mapping},
               mdspan_accessor_type{}),
        m_track() {
    using desc_type = typename Impl::ViewConstructorDescription<P...>;
    static_assert(
        std::is_convertible<typename desc_type::pointer_type,
                            typename mdspan_type::pointer>::value,
        "Constructing View to wrap user memory must supply matching pointer "
        "type");
    /*Impl::verify_pointer_construction_for_accessor(m_data.accessor(),
                                                   arg_desc.get_pointer());*/
  }

  // The view_alloc() and layout version
  template <class... P,
            //----------------------------------------
            /* requires
             *   !Impl::ViewConstructorDescription<P...>::has_pointer
             *
             *   (Note: basically, this currently means that the first argument
             *   is the return of a call to `Kokkos::view_alloc()`)
             */
            std::enable_if_t<
                !Impl::ViewConstructorDescription<P...>::has_pointer, int> = 0
            //----------------------------------------
            >
  inline BasicView(Impl::ViewConstructorDescription<P...> const& arg_desc,
                   mdspan_mapping_type const& mapping)
      : m_data(), m_track() {
    //------------------------------------------------------------
    // add default memory and execution spaces in
    using desc_with_defaults_t = typename Impl::ViewConstructorDescription<
        P...>::template with_default_traits<typename traits::execution_space,
                                            typename traits::memory_space>;
    auto desc_with_defaults = desc_with_defaults_t{arg_desc};

    //------------------------------------------------------------
    // some sanity checks

    static_assert(is_managed,
                  "View allocation constructor requires managed memory");

    if (desc_with_defaults_t::initialize &&
        !desc_with_defaults.get_execution_space().impl_is_initialized()) {
      // If initializing view data then
      // the execution space must be initialized.
      Kokkos::Impl::throw_runtime_exception(
          "Constructing View and initializing data with uninitialized "
          "execution space");
    }

    //------------------------------------------------------------
#if defined(KOKKOS_ENABLE_CUDA)
    // If allocating in CudaUVMSpace must fence before and after
    // the allocation to protect against possible concurrent access
    // on the CPU and the GPU.
    // Fence using the trait's execution space (which will be Kokkos::Cuda)
    // to avoid incomplete type errors from using Kokkos::Cuda directly.
    if (std::is_same<Kokkos::CudaUVMSpace,
                     typename traits::device_type::memory_space>::value) {
      // TODO @mdspan should this actually translate into device synchronize
      //      since it works around a bug that might not be addressed by fencing
      //      the stream?
      desc_with_defaults.get_execution_space().fence();
    }
#endif

    //------------------------------------------------------------
    // Set up the layout mapping

    //    auto mapping = mdspan_mapping_type{arg_layout};
    /*    Impl::HandleLayoutPadding<
            desc_with_defaults_t::allow_padding,
            typename traits::value_type>::apply_padding(mapping);*/

    //------------------------------------------------------------
    // Allocate/initialize the data and set up the

    // Note: This is new behavior in refactored View code. The old code aligned
    // the allocation to an 8 byte boundary and then re-aligned the allocation
    // to a Kokkos::Impl::MEMORY_ALIGNMENT boundary, which is redundant almost
    // always (unless for some reason someone compiled with
    // KOKKOS_MEMORY_ALIGNMENT less than 8, in which case we were still not
    // doing what the user actually asked for).

    using construct_destroy_functor_type =
        Impl::ViewValueFunctor<typename desc_with_defaults_t::execution_space,
                               typename traits::value_type>;
    // note: owning raw pointer
    auto* record =
        Impl::SharedAllocationRecord<typename traits::memory_space,
                                     construct_destroy_functor_type>::
            allocate(typename traits::memory_space{},
                     desc_with_defaults.get_label(),
                     mapping.required_span_size() *
                         sizeof(typename traits::value_type));
    // TODO @mdspan decide whether thish should depend on the required span size
    //      or the number of elements in the View (we could still have nonzero
    //      values for the former even if the latter is 0)
    if (mapping.required_span_size() > 0 && desc_with_defaults_t::initialize) {
      // a better name for m_destroy might be m_construct_and_destroy
      // or something like that
      record->m_destroy = construct_destroy_functor_type{
          desc_with_defaults.get_execution_space(),
          static_cast<typename traits::value_type*>(record->data()),
          typename traits::size_type(mapping.required_span_size()),
          desc_with_defaults.get_label()};
      record->m_destroy.construct_shared_allocation();
    }
    m_track.m_tracker.assign_allocated_record_to_uninitialized(record);

    // note: ownership transfer of the `record` raw pointer
    // auto accessor = mdspan_accessor_type{record};
    auto accessor = mdspan_accessor_type{};

    //------------------------------------------------------------
#if defined(KOKKOS_ENABLE_CUDA)
    if (std::is_same<Kokkos::CudaUVMSpace,
                     typename traits::device_type::memory_space>::value) {
      // TODO @mdspan should this actually translate into device synchronize
      //      since it works around a bug that might not be addressed by fencing
      //      the stream?
      desc_with_defaults.get_execution_space().fence();
    }
#endif

    //------------------------------------------------------------
    // We can now finally construct the mdspan that backs this View
    m_data = mdspan_type{
        reinterpret_cast<typename mdspan_type::pointer>(record->data()),
        std::move(mapping), std::move(accessor)};
  }

  // Properties and dimensions: just forward to prop/layout ctor
  template <class... P, class... IntegralTypes,
            //----------------------------------------
            /* requires
             *   (std::is_integral_v<Integral> && ...)
             */
            std::enable_if_t<_MDSPAN_FOLD_AND(std::is_integral<
                                              IntegralTypes>::value /* && ... */) &&
                              sizeof...(IntegralTypes) == mdspan_type::rank_dynamic(),
                             int> = 0
            //----------------------------------------
            >
  explicit inline BasicView(
      const Impl::ViewConstructorDescription<P...>& arg_prop, IntegralTypes... dims)
      : BasicView(arg_prop,
                  mdspan_mapping_type(
                      typename traits::mdspan_extents_type(dims...))) {
    /* delegating constructor; body must be empty */
  }

  // Allocate with label and layout
  inline BasicView(std::string const& arg_label,
                   mdspan_mapping_type const& arg_mapping)
      : BasicView(Kokkos::view_alloc(arg_label),
                  mdspan_mapping_type{arg_mapping}) {
    /* delegating constructor; body must be empty */
  }

  // Allocate with label + layout dimensions
  template <
      class... IntegralTypes,
      //----------------------------------------
      /* requires
       *   (std::is_integral_v<IntegralTypes> && ...)
       *   && TODO @mdspan layout is constructible from dimensions only
       */
      std::enable_if_t<
// is_integral doesn't work for untyped enum ...
              //          _MDSPAN_FOLD_AND(std::is_integral<IntegralTypes>::value /* && ... */),
             _MDSPAN_FOLD_AND(std::is_convertible<IntegralTypes,ptrdiff_t>::value) &&
             sizeof...(IntegralTypes) == mdspan_type::rank_dynamic(),
     int> = 0
      //----------------------------------------
      >
  inline BasicView(std::string const& arg_label,
                            IntegralTypes... dimensions)
      : BasicView(Kokkos::view_alloc(arg_label),
                  mdspan_mapping_type(
                      typename traits::mdspan_extents_type(dimensions...))) {}

  // </editor-fold> end View Ctor Prop constructor }}}3
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // <editor-fold desc="Pointer and sizes constructor"> {{{3

  explicit BasicView(pointer_type arg_ptr,
                     typename traits::array_layout const& arg_layout)
      : BasicView(Kokkos::view_wrap(arg_ptr), mdspan_mapping_type{arg_layout}) {
    /* delegating ctor; must be empty */
  }

  template <class... IntegralTypes,
            //----------------------------------------
            /* requires
             *   (std::is_convertible_v<IntegralTypes, std::size_t> && ...)
             */
            std::enable_if_t<_MDSPAN_FOLD_AND(std::is_convertible<
                                              IntegralTypes,
                                              std::size_t>::value /* && ... */),
                             int> = 0
            //----------------------------------------
            >
  explicit BasicView(pointer_type arg_ptr, IntegralTypes... arg_dims)
      : BasicView(Kokkos::view_wrap(arg_ptr),
                  typename traits::array_layout(std::size_t(arg_dims)...)) {
    /* delegating ctor; must be empty */
  }

  // </editor-fold> end Pointer and sizes constructor }}}3
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // <editor-fold desc="Scratch memory space constructors"> {{{3

  explicit BasicView(
      typename traits::execution_space::scratch_memory_space const& arg_space,
      mdspan_mapping_type const& arg_mapping)
      // Delegate to the view wrap constructor
      : BasicView(Kokkos::view_wrap(reinterpret_cast<pointer_type>(
                      arg_space.get_shmem_aligned(
                          BasicView::required_allocation_size(arg_mapping),
                          sizeof(typename traits::value_type)))),
                  mdspan_mapping_type{arg_mapping}) {
    /* delegating ctor; must be empty */
  }

  template <class... IntegralTypes,
            //----------------------------------------
            /* requires
             *   (std::is_convertible_v<IntegralTypes, std::size_t> && ...)
             */
            std::enable_if_t<_MDSPAN_FOLD_AND(std::is_convertible<
                                              IntegralTypes,
                                              std::size_t>::value /* && ... */),
                             int> = 0
            //----------------------------------------
            >
  explicit BasicView(
      typename traits::execution_space::scratch_memory_space const& arg_space,
      IntegralTypes... arg_dims)
      : BasicView(arg_space,
                  mdspan_mapping_type(
                      typename traits::mdspan_extents_type(arg_dims...))) {
    /* delegating ctor; must be empty */
#if 0  // TODO @mdspan check rank at runtime??
#ifdef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
    Impl::runtime_check_rank_host(
        rank_dynamic,
        /*std::is_same<typename traits::specialize, void>::value*/ true,
        std::size_t{arg_dims}..., label());
#else
    Impl::runtime_check_rank_device(
        rank_dynamic,
        /*std::is_same<typename traits::specialize, void>::value*/ true,
        std::size_t{arg_dims}..., label());
#endif
#endif
  }

  // </editor-fold> end Scratch memory space constructors }}}3
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  // TODO ViewTracker and mapping constructor???

  // TODO construct View from shared allocation tracker object and map?

  // </editor-fold> end constructors, destructor, and assignment }}}2
  //----------------------------------------------------------------------------

  //----------------------------------------------------------------------------
  // <editor-fold desc="public member functions"> {{{2

  KOKKOS_FUNCTION constexpr typename traits::array_layout layout() const {
    return _layout_helper(
        std::integral_constant<bool,
                               traits::array_layout::is_extent_constructible>{},
        std::make_index_sequence <
                // sorry for the apparently obfuscated code; we need to double
                // length to rifle the sequence if we need the strides also, so:
                traits::array_layout::is_extent_constructible
            ? rank
            : rank * 2 > {});
  }

  KOKKOS_FUNCTION constexpr size_t size() const noexcept {
    return m_data.size();
  }

  KOKKOS_FUNCTION constexpr size_t span() const noexcept {
    return m_data.mapping().required_span_size();
  }

  KOKKOS_FUNCTION constexpr bool span_is_contiguous() const {
    return span() == size();
  }

  KOKKOS_FUNCTION constexpr bool is_allocated() const {
    return m_data.data() != nullptr;
  }

  KOKKOS_FUNCTION constexpr pointer_type data() const { return m_data.data(); }
  KOKKOS_FUNCTION void assign_data(const pointer_type& ptr) {
    m_track.m_tracker.clear();
    m_data = mdspan_type(ptr, m_data.mapping(), m_data.accessor());
  }
  KOKKOS_FUNCTION constexpr mdspan_type get_mdspan() const { return m_data; }
  KOKKOS_FUNCTION constexpr track_type impl_get_tracker() const {
    return m_track;
  }

  // TODO @mdspan assign_data

  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // <editor-fold desc="extents"> {{{3

  template <class IntegralType>
  KOKKOS_FUNCTION constexpr std::size_t extent(IntegralType i) const noexcept {
    // Note: intentionally not using {} construction here to avoid narrowing
    // warnings.
    static_assert(std::is_integral<IntegralType>::value,
                  "Kokkos::View::extent() called with non-integral type.");
    return i<mdspan_type::extents_type::rank()?m_data.extent(std::size_t(i)):1;
  }

  template <class IntegralType>
  KOKKOS_FUNCTION constexpr int extent_int(IntegralType i) const noexcept {
    // Note: intentionally not using {} construction here to avoid narrowing
    // warnings.
    static_assert(std::is_integral<IntegralType>::value,
                  "Kokkos::View::extent_int() called with non-integral type.");
    return i<mdspan_type::extents_type::rank()?int(m_data.extent(std::size_t(i))):1;
  }

  KOKKOS_FUNCTION static constexpr std::size_t static_extent(unsigned i)
      noexcept {
    // Note: intentionally not using {} construction here to avoid narrowing
    // warnings.
    return mdspan_type::extents_type::static_extent(std::size_t(i));
  }

  // </editor-fold> end extents }}}3
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // <editor-fold desc="strides"> {{{3

  // TENTATIVELY DEPRECATED
  KOKKOS_FUNCTION constexpr size_t stride_0() const { return m_data.stride(0); }
  // TENTATIVELY DEPRECATED
  KOKKOS_FUNCTION constexpr size_t stride_1() const { return m_data.stride(1); }
  // TENTATIVELY DEPRECATED
  KOKKOS_FUNCTION constexpr size_t stride_2() const { return m_data.stride(2); }
  // TENTATIVELY DEPRECATED
  KOKKOS_FUNCTION constexpr size_t stride_3() const { return m_data.stride(3); }
  // TENTATIVELY DEPRECATED
  KOKKOS_FUNCTION constexpr size_t stride_4() const { return m_data.stride(4); }
  // TENTATIVELY DEPRECATED
  KOKKOS_FUNCTION constexpr size_t stride_5() const { return m_data.stride(5); }
  // TENTATIVELY DEPRECATED
  KOKKOS_FUNCTION constexpr size_t stride_6() const { return m_data.stride(6); }
  // TENTATIVELY DEPRECATED
  KOKKOS_FUNCTION constexpr size_t stride_7() const { return m_data.stride(7); }

  template <class IntegralType>
  KOKKOS_FUNCTION constexpr std::size_t stride(IntegralType i) const noexcept {
    // Note: intentionally not using {} construction here to avoid narrowing
    // warnings.
    static_assert(std::is_integral<IntegralType>::value,
                  "Kokkos::View::stride() called with non-integral type.");
    return m_data.stride(std::size_t(i));
  }

  // TENTATIVELY DEPRECATED
  template <class IntegralType>
  KOKKOS_FUNCTION constexpr void stride(IntegralType* s) const noexcept {
    static_assert(
        std::is_integral<IntegralType>::value,
        "Kokkos::View::stride() called with pointer to non-integral type.");
    for (int i = 0; i < m_data.rank(); ++i) {
      s[i] = IntegralType(stride(i));
    }
  }

  // </editor-fold> end strides }}}3
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // <editor-fold desc="element access"> {{{3

  template <class... IntegralTypes>
  KOKKOS_FORCEINLINE_FUNCTION constexpr reference_type operator()(
      IntegralTypes... i) const {
    static_assert(
        //_MDSPAN_FOLD_AND(std::is_integral<IntegralTypes>::value /* && ... */),
        _MDSPAN_FOLD_AND(std::is_convertible<IntegralTypes,ptrdiff_t>::value),
        "Kokkos::View::operator() must be called with integral index types.");
    static_assert(sizeof...(IntegralTypes) == rank,
                  "Wrong number of indices given to Kokkos::View::operator() "
                  "(argument count must match rank).");
    Impl::ViewVerifySpace<typename traits::memory_space>::check();
    KOKKOS_IMPL_VIEW_OPERATOR_VERIFY((m_tracker, m_data, i...))
    return m_data(i...);
  }

  template <class IntegralType>
  KOKKOS_FORCEINLINE_FUNCTION reference_type operator[](IntegralType i) const
      noexcept {
    static_assert(std::is_convertible<IntegralType,ptrdiff_t>::value,
                  "Kokkos::View::operator[] called with non-integral type.");
    static_assert(mdspan_type::rank() == 1,
                  "Only a rank 1 View can be dereferenced with operator[].");
    KOKKOS_IMPL_VIEW_OPERATOR_VERIFY((m_tracker, m_data, i))
    return m_data(i);
  }

  template <class... IntegralTypes>
  KOKKOS_FORCEINLINE_FUNCTION reference_type access(IntegralTypes... i) const {
    static_assert(
        _MDSPAN_FOLD_AND(std::is_convertible<IntegralTypes,ptrdiff_t>::value),
        //_MDSPAN_FOLD_AND(std::is_integral<IntegralTypes>::value /* && ... */),
        "Kokkos::View::access() must be called with integral index types.");
    return _access_helper(std::make_index_sequence<mdspan_type::rank()>{},
                          i...);
  }

  // </editor-fold> end element access }}}3
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // <editor-fold desc="allocation tracking"> {{{3

  // TODO @mdspan
  KOKKOS_FUNCTION
  int use_count() const {
    return m_track.m_tracker.use_count();
    // return Impl::use_count_for_accessor(m_data.accessor());
  }

  inline std::string label() const {
    return m_track.m_tracker.template get_label<typename traits::memory_space>();
    //    return Impl::get_label_from_accessor(m_data.accessor());
  }

  // </editor-fold> end allocation tracking }}}3
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  // </editor-fold> end public member functions }}}2
  //----------------------------------------------------------------------------

  //----------------------------------------------------------------------------
  // <editor-fold desc="static member functions"> {{{2

  template<class ... Extents>
  static constexpr size_t required_allocation_size(
      typename mdspan_type::mapping_type const& arg_mapping) {
    return typename mdspan_type::mapping_type{arg_mapping}.required_span_size()*sizeof(typename traits::value_type);
  }

  template <class... IntegralTypes>
  static constexpr size_t required_allocation_size(IntegralTypes... dims) {
    static_assert(
        _MDSPAN_FOLD_AND(
            std::is_convertible<IntegralTypes, std::size_t>::value /* ... */),
        "Arguments to View::required_allocation_size() must be integral");
    static_assert(traits::array_layout::is_extent_constructible,
                  "View::required_allocation_size(Dimensions...) can only be "
                  "called if the layout is constructible from the dimensions "
                  "alone (e.g., not something like LayoutStride)");
    return BasicView::required_allocation_size(
        mdspan_mapping_type{std::size_t(dims)...});
  }

  template <class... IntegralTypes>
  static constexpr size_t shmem_size(IntegralTypes... dims) {
    static_assert(
        _MDSPAN_FOLD_AND(
            std::is_convertible<IntegralTypes, std::size_t>::value /* ... */),
        "Arguments to View::shmem_size(Dimensions...) must be integral");
    static_assert(traits::array_layout::is_extent_constructible,
                  "View::shmem_size(Dimensions...) can only be called if the "
                  "layout is constructible from the dimensions alone (e.g., "
                  "not something like LayoutStride)");
    return BasicView::shmem_size(
        typename traits::array_layout{std::size_t(dims)...});
  }

  static constexpr size_t shmem_size(
      typename traits::array_layout const& arg_layout) {
    return (typename mdspan_type::mapping_type{arg_layout}.required_span_size() + 1) * 
           sizeof(typename traits::value_type);
  }

  // </editor-fold> end static member functions }}}2
  //----------------------------------------------------------------------------
};

// </editor-fold> end BasicView }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="View"> {{{1

namespace Impl {
//TODO: check that the args which are unused right now actually are the default constructor args


template<class Extents>
struct make_extents_from_too_many_args<Extents, 0> {
  static constexpr Extents create(ptrdiff_t n0 = -1, ptrdiff_t n1 = -1, ptrdiff_t n2 = -1, ptrdiff_t n3 = -1, ptrdiff_t n4 = -1, ptrdiff_t n5 = -1, ptrdiff_t n6 = -1, ptrdiff_t n7 = -1) {
    return Extents();
  }
};
template<class Extents>
struct make_extents_from_too_many_args<Extents, 1> {
  static constexpr Extents create(ptrdiff_t n0 = -1, ptrdiff_t n1 = -1, ptrdiff_t n2 = -1, ptrdiff_t n3 = -1, ptrdiff_t n4 = -1, ptrdiff_t n5 = -1, ptrdiff_t n6 = -1, ptrdiff_t n7 = -1) {
    return Extents(n0);
  }
};
template<class Extents>
struct make_extents_from_too_many_args<Extents, 2> {
  static constexpr Extents create(ptrdiff_t n0 = -1, ptrdiff_t n1 = -1, ptrdiff_t n2 = -1, ptrdiff_t n3 = -1, ptrdiff_t n4 = -1, ptrdiff_t n5 = -1, ptrdiff_t n6 = -1, ptrdiff_t n7 = -1) {
    return Extents(n0, n1);
  }
};
template<class Extents>
struct make_extents_from_too_many_args<Extents, 3> {
  static constexpr Extents create(ptrdiff_t n0 = -1, ptrdiff_t n1 = -1, ptrdiff_t n2 = -1, ptrdiff_t n3 = -1, ptrdiff_t n4 = -1, ptrdiff_t n5 = -1, ptrdiff_t n6 = -1, ptrdiff_t n7 = -1) {
    return Extents(n0, n1, n2);
  }
};
template<class Extents>
struct make_extents_from_too_many_args<Extents, 4> {
  static constexpr Extents create(ptrdiff_t n0 = -1, ptrdiff_t n1 = -1, ptrdiff_t n2 = -1, ptrdiff_t n3 = -1, ptrdiff_t n4 = -1, ptrdiff_t n5 = -1, ptrdiff_t n6 = -1, ptrdiff_t n7 = -1) {
    return Extents(n0, n1, n2, n3);
  }
};
template<class Extents>
struct make_extents_from_too_many_args<Extents, 5> {
  static constexpr Extents create(ptrdiff_t n0 = -1, ptrdiff_t n1 = -1, ptrdiff_t n2 = -1, ptrdiff_t n3 = -1, ptrdiff_t n4 = -1, ptrdiff_t n5 = -1, ptrdiff_t n6 = -1, ptrdiff_t n7 = -1) {
    return Extents(n0, n1, n2, n3, n4);
  }
};
template<class Extents>
struct make_extents_from_too_many_args<Extents, 6> {
  static constexpr Extents create(ptrdiff_t n0 = -1, ptrdiff_t n1 = -1, ptrdiff_t n2 = -1, ptrdiff_t n3 = -1, ptrdiff_t n4 = -1, ptrdiff_t n5 = -1, ptrdiff_t n6 = -1, ptrdiff_t n7 = -1) {
    return Extents(n0, n1, n2, n3, n4, n5);
  }
};
template<class Extents>
struct make_extents_from_too_many_args<Extents, 7> {
  static constexpr Extents create(ptrdiff_t n0 = -1, ptrdiff_t n1 = -1, ptrdiff_t n2 = -1, ptrdiff_t n3 = -1, ptrdiff_t n4 = -1, ptrdiff_t n5 = -1, ptrdiff_t n6 = -1, ptrdiff_t n7 = -1) {
    return Extents(n0, n1, n2, n3, n4, n5, n6);
  }
};
template<class Extents>
struct make_extents_from_too_many_args<Extents, 8> {
  static constexpr Extents create(ptrdiff_t n0 = -1, ptrdiff_t n1 = -1, ptrdiff_t n2 = -1, ptrdiff_t n3 = -1, ptrdiff_t n4 = -1, ptrdiff_t n5 = -1, ptrdiff_t n6 = -1, ptrdiff_t n7 = -1) {
    return Extents(n0, n1, n2, n3, n4, n5, n6, n7);
  }
};
}

template <class DataType, class... Properties>
class View : public Impl::NormalizeViewProperties<
                 DataType, Impl::type_list<Properties...>>::type {
 public:
  using basic_view_type = typename Impl::NormalizeViewProperties<
      DataType, Impl::type_list<Properties...>>::type;

  template <
      class... Args,
      std::enable_if_t<std::is_convertible<
                           typename View<Args...>::basic_view_type::mdspan_type,
                           typename basic_view_type::mdspan_type>::value,
                       int> = 0>
  View(const View<Args...>& other_view) {
    basic_view_type::m_data  = other_view.get_mdspan();
     basic_view_type::m_track.assign(other_view);
  }

  template < class ... Args, std::enable_if_t<
          _MDSPAN_FOLD_AND(std::is_convertible<Args,ptrdiff_t>::value)
       && ((sizeof...(Args))!= basic_view_type::mdspan_type::extents_type::rank_dynamic()), int> = 0>
  View(const std::string& label, Args... args): basic_view_type(label, 
               typename basic_view_type::mdspan_type::mapping_type(
                     Kokkos::Impl::make_extents_from_too_many_args<typename basic_view_type::mdspan_type::extents_type>::create(args...))) {}

  template < class ... Args, std::enable_if_t<
          _MDSPAN_FOLD_AND(std::is_convertible<Args,ptrdiff_t>::value)
       && !((sizeof...(Args))!= basic_view_type::mdspan_type::extents_type::rank_dynamic()), int> = 0>
  View(const std::string& label, Args... args): basic_view_type(label, args...) {}

  template <class... P, class... IntegralTypes,
            //----------------------------------------
            /* requires
             *   (std::is_integral_v<Integral> && ...)
             */
            std::enable_if_t<_MDSPAN_FOLD_AND(std::is_integral<
                                              IntegralTypes>::value /* && ... */) &&
                              sizeof...(IntegralTypes) != basic_view_type::mdspan_type::rank_dynamic(),
                             int> = 0
            //----------------------------------------
            >
  explicit inline View(
      const Impl::ViewConstructorDescription<P...>& arg_prop, IntegralTypes... dims)
      : basic_view_type(arg_prop,
               typename basic_view_type::mdspan_type::mapping_type(
                     Kokkos::Impl::make_extents_from_too_many_args<typename basic_view_type::mdspan_type::extents_type>::create(dims...))) {
    /* delegating constructor; body must be empty */
  }

  template <class... P, class... IntegralTypes,
            //----------------------------------------
            /* requires
             *   (std::is_integral_v<Integral> && ...)
             */
            std::enable_if_t<_MDSPAN_FOLD_AND(std::is_integral<
                                              IntegralTypes>::value /* && ... */) &&
                              sizeof...(IntegralTypes) == basic_view_type::mdspan_type::rank_dynamic(),
                             int> = 0
            //----------------------------------------
            >
  explicit inline View(
      const Impl::ViewConstructorDescription<P...>& arg_prop, IntegralTypes... dims)
      : basic_view_type(arg_prop,
                  typename basic_view_type::mdspan_mapping_type(
                      typename basic_view_type::traits::mdspan_extents_type(dims...))) {
    /* delegating constructor; body must be empty */
  }

   //   : basic_view_type(other_view.get_mdspan(),
    //                    other_view.impl_get_tracker()){};
  template <
      class... Args,
      std::enable_if_t<std::is_assignable<
                           typename View<Args...>::basic_view_type::mdspan_type,
                           typename basic_view_type::mdspan_type>::value,
                       int> = 0>
  View& operator=(const View<Args...>& other_view) {
    basic_view_type::m_data  = other_view.get_mdspan();
    basic_view_type::m_track.assign(other_view);
   // = other_view.impl_get_tracker();
    return *this;
  };

 private:
  static constexpr void _deferred_instantiation_static_assertions() {
    // Needs to be here because View needs to be complete for this static
    // assertion.
    // It's important to assert that we're layout-compatible with base_t so that
    // coverting between this type and a compatible `BasicView` is safe and
    // doesn't slice any data.
    static_assert(sizeof(View) == sizeof(basic_view_type),
                  "Kokkos internal implementation error: View type isn't "
                  "layout compatible"
                  " with normalized BasicView type that it's based on.");
  }

 public:
  using basic_view_type::basic_view_type;
};

// </editor-fold> end View }}}1
//==============================================================================

template <class LT, class... LP, class RT, class... RP>
KOKKOS_INLINE_FUNCTION bool operator==(const BasicView<LT, LP...>& lhs,
                                       const BasicView<RT, RP...>& rhs) {
  // Same data, layout, dimensions
  using lhs_traits = ViewTypeTraits<BasicView<LT, LP...>>;
  using rhs_traits = ViewTypeTraits<BasicView<RT, RP...>>;

  return std::is_same<typename lhs_traits::const_value_type,
                      typename rhs_traits::const_value_type>::value &&
         std::is_same<typename lhs_traits::array_layout,
                      typename rhs_traits::array_layout>::value &&
         std::is_same<typename lhs_traits::memory_space,
                      typename rhs_traits::memory_space>::value &&
         unsigned(lhs.rank) == unsigned(rhs.rank) &&
         lhs.data() == rhs.data() && lhs.span() == rhs.span() &&
         lhs.extent(0) == rhs.extent(0) && lhs.extent(1) == rhs.extent(1) &&
         lhs.extent(2) == rhs.extent(2) && lhs.extent(3) == rhs.extent(3) &&
         lhs.extent(4) == rhs.extent(4) && lhs.extent(5) == rhs.extent(5) &&
         lhs.extent(6) == rhs.extent(6) && lhs.extent(7) == rhs.extent(7);
}

template <class LT, class... LP, class RT, class... RP>
KOKKOS_INLINE_FUNCTION bool operator!=(const BasicView<LT, LP...>& lhs,
                                       const BasicView<RT, RP...>& rhs) {
  return !(operator==(lhs, rhs));
}

template <class LT, class... LP, class RT, class... RP>
KOKKOS_INLINE_FUNCTION bool operator==(const View<LT, LP...>& lhs,
                                       const View<RT, RP...>& rhs) {
  return reinterpret_cast<const typename View<LT, LP...>::basic_view_type&>(lhs) ==
         reinterpret_cast<const typename View<RT, RP...>::basic_view_type&>(rhs);
}

template <class LT, class... LP, class RT, class... RP>
KOKKOS_INLINE_FUNCTION bool operator!=(const View<LT, LP...>& lhs,
                                       const View<RT, RP...>& rhs) {
  return !(operator==(lhs, rhs));
}
namespace Impl {
template <class T1, class T2, class Enable = void>
struct is_always_assignable_impl: std::false_type {};

template<class T>
struct is_always_assignable_impl<T,T,void>: std::true_type {};

template<>
struct is_always_assignable_impl<Kokkos::LayoutStride, Kokkos::LayoutLeft,void>: std::true_type {};

template<>
struct is_always_assignable_impl<Kokkos::LayoutStride, Kokkos::LayoutRight,void>: std::true_type {};

template<size_t ... DstExtentVals, size_t ... SrcExtentVals>
struct is_always_assignable_impl<std::experimental::extents<SrcExtentVals...>,
                                 std::experimental::extents<DstExtentVals...>,
                                 std::enable_if_t<!std::is_same<
                                   std::experimental::extents<SrcExtentVals...>,
                                   std::experimental::extents<DstExtentVals...>
                                 >::value,void>> {
  constexpr static bool value =
    _MDSPAN_FOLD_AND(DstExtentVals!=std::experimental::dynamic_extent?DstExtentVals==SrcExtentVals:true);

};

template <class... ViewTDst, class... ViewTSrc>
struct is_always_assignable_impl<Kokkos::BasicView<ViewTDst...>,
                                 Kokkos::BasicView<ViewTSrc...>,
                                 std::enable_if_t<!std::is_same<
                                   Kokkos::BasicView<ViewTDst...>,
                                   Kokkos::BasicView<ViewTSrc...>
                                 >::value, void>> {
  using dst_view_t = Kokkos::BasicView<ViewTDst...>;
  using src_view_t = Kokkos::BasicView<ViewTSrc...>;


  constexpr static bool value =
    std::is_convertible<src_view_t, dst_view_t>::value &&
    is_always_assignable_impl<typename dst_view_t::array_layout,
                              typename src_view_t::array_layout>::value &&
    is_always_assignable_impl<typename dst_view_t::mdspan_type::extents_type,
                              typename src_view_t::mdspan_type::extents_type>::value;
};

}

template <class View1, class View2>
using is_always_assignable = Impl::is_always_assignable_impl<
    typename std::remove_reference<View1>::type::basic_view_type,
    typename std::remove_const<
        typename std::remove_reference<View2>::type>::type::basic_view_type>;

template <class ... ViewTDst, class ... ViewTSrc>
bool is_assignable(const Kokkos::View<ViewTDst...>& dst, const Kokkos::View<ViewTSrc...>& src) {
  // FIXME Return the correct value here
  return std::is_convertible<Kokkos::View<ViewTSrc...>, Kokkos::View<ViewTDst...>>::value;
  // && is_assignable(dst.mapping(), src.mapping()); // this is a runtime check
}

#ifdef KOKKOS_ENABLE_CXX17
template <class T1, class T2>
inline constexpr bool is_always_assignable_v =
    is_always_assignable<T1, T2>::value;
#endif



}  // end namespace Kokkos
#include <View/Kokkos_MDSpanView_Subview.hpp>
#undef KOKKOS_IMPL_VIEW_OPERATOR_VERIFY
#undef KOKKOS_IMPL_SINK

#endif /* #ifndef KOKKOS_MDSPANVIEW_HPP */
