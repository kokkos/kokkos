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

#include <View/Kokkos_NormalizeView.hpp>
#include <View/Kokkos_ExtractExtents.hpp>
#include <View/Kokkos_MDSpanLayout.hpp>
#include <View/Kokkos_MDSpanAccessor.hpp>
#include <View/Kokkos_ViewVerify.hpp>
#include <View/Kokkos_IsView.hpp>
#include <View/Kokkos_ViewAllocation.hpp>

#include <Kokkos_ViewAlloc.hpp>

#include <impl/Kokkos_ViewTracker.hpp>
#include <impl/Kokkos_ViewUniformType.hpp>
#include <impl/Kokkos_ViewCtor.hpp>

#include <impl/Kokkos_Utilities.hpp>  // type_list

#include <experimental/mdspan>

#if defined(KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)

#define KOKKOS_IMPL_SINK(ARG) ARG

#define KOKKOS_IMPL_VIEW_OPERATOR_VERIFY(ARG) \
  Kokkos::Impl::view_verify_operator_bounds<memory_space> ARG;

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

  using _extracted          = Impl::ExtractExtents<DataType>;
  using mdspan_element_type = typename _extracted::value_type;
  using mdspan_extents_type = typename _extracted::extents_type;

  // </editor-fold> end private member types }}}2
  //----------------------------------------------------------------------------

 public:
  //----------------------------------------------------------------------------
  // <editor-fold desc="public member types"> {{{2

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

  // </editor-fold> end public member types }}}2
  //----------------------------------------------------------------------------
};

//==============================================================================
// <editor-fold desc="BasicView"> {{{1

template <class DataType, class Layout, class Space, class MemoryTraits>
class BasicView
    : ViewTypeTraits<BasicView<DataType, Layout, Space, MemoryTraits>> {
 public:
  using traits =
      ViewTypeTraits<BasicView<DataType, Layout, Space, MemoryTraits>>;

 private:
  //----------------------------------------------------------------------------
  // <editor-fold desc="private member types"> {{{2

  using mdspan_layout_type =
      typename Impl::MDSpanLayoutFromKokkosLayout<traits, Layout>::type;
  using mdspan_accessor_type =
      typename Impl::MDSpanAccessorFromKokkosMemoryTraits<traits,
                                                          MemoryTraits>::type;

  using mdspan_type =
      std::experimental::basic_mdspan<typename traits::mdspan_element_type,
                                      typename traits::mdspan_extents_type,
                                      mdspan_layout_type, mdspan_accessor_type>;

  using view_tracker_type = Impl::ViewTracker<BasicView>;

  // </editor-fold> end private member types }}}2
  //----------------------------------------------------------------------------

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

  // </editor-fold> end private data members }}}2
  //----------------------------------------------------------------------------

 public:
  //----------------------------------------------------------------------------
  // <editor-fold desc="public member types"> {{{2

  using reference_type = typename mdspan_type::reference;
  using pointer_type   = typename mdspan_type::pointer;

  using runtime_data_type =
      typename Impl::ViewScalarToDataType<typename traits::value_type,
                                          mdspan_type::rank()>;
  using runtime_const_data_type =
      typename Impl::ViewScalarToDataType<typename traits::const_value_type,
                                          mdspan_type::rank()>;

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
      Args... KOKKOS_IMPL_SINK(args)) {
#if defined(KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)
    Impl::view_verify_operator_bounds<typename traits::memory_space>(
        m_data, idxs..., args...);
#endif
    return m_data(idxs...);
  }

  template <std::size_t... Idxs, class... Args>
  KOKKOS_FORCEINLINE_FUNCTION constexpr reference_type _access_helper(
      std::integer_sequence<std::size_t, Idxs...>, Args... args) {
    return _access_helper_impl<Idxs...>(size_type(args)...);
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
      : m_data(rhs.m_data) {
    // TODO @mdspan check runtime dimension compatibility (check in the mapping)
  }

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
    // TODO @mdspan check runtime dimension compatibility!!!
    return *this;
  }

  // </editor-fold> end compatible View conversion }}}3
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  // TODO subview constructor

  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // <editor-fold desc="View Ctor Prop constructors"> {{{3

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
  explicit inline BasicView(
      Impl::ViewConstructorDescription<P...> const& arg_desc,
      typename traits::array_layout const& arg_layout)
      : m_data(arg_desc.get_pointer(), arg_layout,
               mdspan_accessor_type(
                   Impl::runtime_non_owning_accessor_tag_t{},
                   // Potentially needed to check alignment, etc.
                   arg_desc.get_pointer()
               ) {
    static_assert(
        std::is_same<typename traits::pointer_type,
                     typename Impl::ViewConstructorDescription<
                         P...>::pointer_type>::value,
        "Constructing View to wrap user memory must supply matching pointer "
        "type");
  }

  // The view_alloc() and layout version
  template <class... P,
            //----------------------------------------
            /* requires
             *   !Impl::ViewConstructorDescription<P...>::has_pointer
             */
            std::enable_if_t<
                !Impl::ViewConstructorDescription<P...>::has_pointer, int> = 0
            //----------------------------------------
            >
  explicit inline BasicView(Impl::ViewConstructorDescription<P...> const& arg_desc,
                            typename traits::array_layout const& arg_layout)
      : m_data() {
    using alloc_prop = Impl::ViewConstructorDescription<P...>;
    static_assert(is_managed,
                  "View allocation constructor requires managed memory");

    if (alloc_prop::initialize &&
        !arg_desc.get_execution_space().impl_is_initialized()) {
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
      // TODO @mdspan should this actually translate into device synchronize?
      arg_desc.get_execution_space().fence();
    }
#endif
    //------------------------------------------------------------

    // TODO handle padding in the layout (incorporate arg_desc::allow_padding)

    auto* record = ViewAllocationMechanism<ValueType>::allocate_shared(
        arg_desc.get_execution_space(), typename traits::memory_space{},
        arg_desc.get_label(), arg_layout.span(), arg_desc::initialize);

//------------------------------------------------------------
#if defined(KOKKOS_ENABLE_CUDA)
    if (std::is_same<Kokkos::CudaUVMSpace,
                     typename traits::device_type::memory_space>::value) {
      // TODO @mdspan should this actually translate into device synchronize?
      arg_desc.get_execution_space().fence();
    }
#endif
    //------------------------------------------------------------

    m_data = mdspan_type {
      record->data(), arg_layout, mdspan_accessor_type { record }
    }

    // Setup and initialization complete, start tracking
    // m_track.m_tracker.assign_allocated_record_to_uninitialized(record);
  }

  // Properties and dimensions: just forward to prop/layout ctor
  template <class... P, class... Integral,
            //----------------------------------------
            /* requires
             *   (std::is_integral_v<Integral> && ...)
             */
            std::enable_if_t<_MDSPAN_FOLD_AND(std::is_integral<
                                              Integral>::value /* && ... */),
                             int> = 0
            //----------------------------------------
            >
  explicit inline BasicView(const Impl::ViewCtorProp<P...>& arg_prop,
                            Integral... dims)
      : BasicView(arg_prop, array_layout(dims...)) {
  // TODO @mdspan Rewrite runtime_check_rank_*()
#ifdef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
    Impl::runtime_check_rank_host(
        rank_dynamic,
        /*std::is_same<typename traits::specialize, void>::value*/ true,
        dims..., label());
#else
    Impl::runtime_check_rank_device(
        rank_dynamic,
        /*std::is_same<typename traits::specialize, void>::value*/ true,
        dims..., label());
#endif
  }

  // Allocate with label and layout
  explicit inline BasicView(std::string const& arg_label,
                            array_layout const& arg_layout)
      : BasicView(Impl::ViewCtorProp<std::string>(arg_label), arg_layout) {}

  // Allocate with label + layout dimensions
  template <
      class... IntegralTypes,
      //----------------------------------------
      /* requires
       *   (std::is_integral_v<IntegralTypes> && ...)
       *   && TODO layout is constructible from dimensions only
       */
      std::enable_if_t<
          _MDSPAN_FOLD_AND(std::is_integral<IntegralTypes>::value /* && ... */),
          int> = 0
      //----------------------------------------
      >
  explicit inline BasicView(std::string const& arg_label,
                            IntegralTypes... dimensions)
      : BasicView(Impl::ViewCtorProp<std::string>(arg_label),
                  array_layout(dimensions...)) {}

  // </editor-fold> end View Ctor Prop constructor }}}3
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // <editor-fold desc="Pointer and sizes constructor"> {{{3

  explicit BasicView(pointer_type arg_ptr, array_layout const& arg_layout)
      : BasicView(Impl::ViewCtorProp<pointer_type>(arg_ptr), arg_layout) {}

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
      : BasicView(Impl::ViewCtorProp<pointer_type>(arg_ptr),
                  array_layout(std::size_t{arg_dims}...)) {
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
  }

  // </editor-fold> end Pointer and sizes constructor }}}3
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // <editor-fold desc="Scratch memory space constructors"> {{{3

  explicit BasicView(
      typename execution_space::scratch_memory_space const& arg_space,
      array_layout const& arg_layout)
      : BasicView(
            Impl::ViewCtorProp<pointer_type>{
                reinterpret_cast<pointer_type>(arg_space.get_shmem_aligned(
                    BasicView::required_allocation_size(arg_layout),
                    sizeof(value_type)))},
            arg_layout) {}

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
      typename execution_space::scratch_memory_space const& arg_space,
      IntegralTypes... arg_dims)
      : BasicView(arg_space, array_layout{arg_dims...}) {
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
  }

  // </editor-fold> end Scratch memory space constructors }}}3
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  // TODO ViewTracker and mapping constructor???

  // TODO construct View from shared allocation tracker object and map?

  // </editor-fold> end constructors, destructor, and assignment }}}2
  //----------------------------------------------------------------------------

  //----------------------------------------------------------------------------
  // <editor-fold desc="public member functions"> {{{2

  KOKKOS_FUNCTION constexpr array_layout layout() const {
    return _layout_helper(
        std::integral_constant<bool, array_layout::is_extent_constructible>{},
        std::make_index_sequence <
                // sorry for the apparently obfuscated code; we need to double
                // length to rifle the sequence if we need the strides also, so:
                array_layout::is_extent_constructible
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

  KOKKOS_FUNCTION constexpr pointer_type data() const {
    return m_data.data(); }

  // TODO @mdspan assign_data

  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // <editor-fold desc="extents"> {{{3

  template <class IntegralType>
  KOKKOS_FUNCTION constexpr std::size_t extent(IntegralType i) const noexcept {
    // Note: intentionally not using {} construction here to avoid narrowing
    // warnings.
    static_assert(std::is_integral<IntegralType>::value,
                  "Kokkos::View::extent() called with non-integral type.");
    return m_data.extent(std::size_t(i));
  }

  template <class IntegralType>
  KOKKOS_FUNCTION constexpr int extent_int(IntegralType i) const noexcept {
    // Note: intentionally not using {} construction here to avoid narrowing
    // warnings.
    static_assert(std::is_integral<IntegralType>::value,
                  "Kokkos::View::extent_int() called with non-integral type.");
    return int(m_data.extent(std::size_t(i)));
  }

  KOKKOS_FUNCTION constexpr std::size_t static_extent(
      unsigned i) const noexcept {
    // Note: intentionally not using {} construction here to avoid narrowing
    // warnings.
    return m_data.extent(std::size_t(i));
  }

  // </editor-fold> end extents }}}3
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // <editor-fold desc="strides"> {{{3

  // TENTATIVELY DEPRECATED
  KOKKOS_FUNCTION constexpr size_t stride_0() const {
    return m_data.stride(0); }
  // TENTATIVELY DEPRECATED
  KOKKOS_FUNCTION constexpr size_t stride_1() const {
    return m_data.stride(1); }
  // TENTATIVELY DEPRECATED
  KOKKOS_FUNCTION constexpr size_t stride_2() const {
    return m_data.stride(2); }
  // TENTATIVELY DEPRECATED
  KOKKOS_FUNCTION constexpr size_t stride_3() const {
    return m_data.stride(3); }
  // TENTATIVELY DEPRECATED
  KOKKOS_FUNCTION constexpr size_t stride_4() const {
    return m_data.stride(4); }
  // TENTATIVELY DEPRECATED
  KOKKOS_FUNCTION constexpr size_t stride_5() const {
    return m_data.stride(5); }
  // TENTATIVELY DEPRECATED
  KOKKOS_FUNCTION constexpr size_t stride_6() const {
    return m_data.stride(6); }
  // TENTATIVELY DEPRECATED
  KOKKOS_FUNCTION constexpr size_t stride_7() const {
    return m_data.stride(7); }

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
        _MDSPAN_FOLD_AND(std::is_integral<IntegralTypes>::value /* && ... */),
        "Kokkos::View::operator() must be called with integral index types.");
    static_assert(sizeof...(IntegralTypes) == m_data.rank(),
                  "Wrong number of indices given to Kokkos::View::operator() "
                  "(argument count must match rank).");
    Impl::ViewVerifySpace<memory_space>::check();
    KOKKOS_IMPL_VIEW_OPERATOR_VERIFY((m_tracker, m_data, i...))
    return m_data(i...);
  }

  template <class IntegralType>
  KOKKOS_FORCEINLINE_FUNCTION std::size_t operator[](
      IntegralType i) const noexcept {
    static_assert(std::is_integral<IntegralType>::value,
                  "Kokkos::View::operator[] called with non-integral type.");
    static_assert(mdspan_type::rank() == 1,
                  "Only a rank 1 View can be dereferenced with operator[].");
    KOKKOS_IMPL_VIEW_OPERATOR_VERIFY((m_tracker, m_data, i))
    return m_data(i);
  }

  template <class... IntegralTypes>
  KOKKOS_FORCEINLINE_FUNCTION reference_type access(IntegralTypes... i) const {
    static_assert(
        _MDSPAN_FOLD_AND(std::is_integral<IntegralTypes>::value /* && ... */),
        "Kokkos::View::access() must be called with integral index types.");
    return _access_helper(std::make_index_sequence<mdspan_type::rank()>{},
                          i...);
  }

  // </editor-fold> end element access }}}3
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // <editor-fold desc="allocation tracking"> {{{3

  KOKKOS_FUNCTION
  int use_count() const {
    return m_tracker.m_tracker.use_count(); }

  inline const std::string label() const {
    return m_tracker.m_tracker.template get_label<memory_space>();
  }

  // </editor-fold> end allocation tracking }}}3
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  // </editor-fold> end public member functions }}}2
  //----------------------------------------------------------------------------

  //----------------------------------------------------------------------------
  // <editor-fold desc="static member functions"> {{{2

  static constexpr size_t required_allocation_size(
      array_layout const& arg_layout) {
    return typename mdspan_type::mapping_type{arg_layout}.required_span_size();
  }

  template <class... IntegralTypes>
  static constexpr size_t required_allocation_size(IntegralTypes... dims) {
    static_assert(
        _MDSPAN_FOLD_AND(
            std::is_convertible<IntegralTypes, std::size_t>::value /* ... */),
        "Arguments to View::required_allocation_size() must be integral");
    static_assert(array_layout::is_extent_constructible,
                  "View::required_allocation_size(Dimensions...) can only be "
                  "called if the layout is constructible from the dimensions "
                  "alone (e.g., not something like LayoutStride)");
    return BasicView::required_allocation_size(
        array_layout{std::size_t{dims}...});
  }

  template <class... IntegralTypes>
  static constexpr size_t shmem_size(IntegralTypes... dims) {
    static_assert(
        _MDSPAN_FOLD_AND(
            std::is_convertible<IntegralTypes, std::size_t>::value /* ... */),
        "Arguments to View::shmem_size(Dimensions...) must be integral");
    static_assert(array_layout::is_extent_constructible,
                  "View::shmem_size(Dimensions...) can only be called if the "
                  "layout is constructible from the dimensions alone (e.g., "
                  "not something like LayoutStride)");
    return BasicView::shmem_size(array_layout{std::size_t{dims}...});
  }

  static constexpr size_t shmem_size(array_layout const& arg_layout) {
    return typename mdspan_type::mapping_type{arg_layout}.required_span_size() +
           sizeof(value_type);
  }

  // </editor-fold> end static member functions }}}2
  //----------------------------------------------------------------------------
};

// </editor-fold> end BasicView }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="View"> {{{1

template <class DataType, class... Properties>
class View : public Impl::NormalizeViewProperties<
                 DataType, Impl::type_list<Properties...>>::type {
 private:
  using base_t = typename Impl::NormalizeViewProperties<
      DataType, Impl::type_list<Properties...>>::type;
  static constexpr void _deferred_instantiation_static_assertions() {
    // Needs to be here because View needs to be complete for this static
    // assertion.
    // It's important to assert that we're layout-compatible with base_t so that
    // coverting between this type and a compatible `BasicView` is safe and
    // doesn't slice any data.
    static_assert(sizeof(View) == sizeof(base_t),
                  "Kokkos internal implementation error: View type isn't "
                  "layout compatible"
                  " with normalized BasicView type that it's based on.");
  }

 public:
  using base_t::base_t;
};

// </editor-fold> end View }}}1
//==============================================================================

}  // end namespace Kokkos

#undef KOKKOS_IMPL_VIEW_OPERATOR_VERIFY
#undef KOKKOS_IMPL_SINK

#endif /* #ifndef KOKKOS_MDSPANVIEW_HPP */
