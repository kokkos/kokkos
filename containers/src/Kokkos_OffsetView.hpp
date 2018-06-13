/*
 * Kokkos_OffsetView.hpp
 *
 *  Created on: Apr 23, 2018
 *      Author: swbova
 */

#ifndef KOKKOS_OFFSETVIEW_HPP_
#define KOKKOS_OFFSETVIEW_HPP_


#include <Kokkos_Core.hpp>

#include <Kokkos_View.hpp>

namespace Kokkos {

  //----------------------------------------------------------------------------
  //----------------------------------------------------------------------------

  template< class DataType , class ... Properties >
  class OffsetView ;

  template< class > struct is_offset_view : public std::false_type {};

  template< class D, class ... P >
  struct is_offset_view< OffsetView<D,P...> > : public std::true_type {};

  template< class D, class ... P >
  struct is_offset_view< const OffsetView<D,P...> > : public std::true_type {};

#define KOKKOS_INVALID_OFFSET int64_t(0)
#define KOKKOS_INVALID_INDEX_RANGE {KOKKOS_INVALID_OFFSET, KOKKOS_INVALID_OFFSET}

  template <typename iType, typename std::enable_if< std::is_integral<iType>::value &&
  std::is_signed<iType>::value, iType >::type = 0>
  using IndexRange  = Kokkos::Array<iType, 2>;


  using index_list_type = std::initializer_list<int64_t>;


  //  template <typename iType,
  //    typename std::enable_if< std::is_integral<iType>::value &&
  //      std::is_signed<iType>::value, iType >::type = 0> using min_index_type = std::initializer_list<iType>;

  namespace Impl {

#ifdef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST


KOKKOS_INLINE_FUNCTION
void runtime_check_rank_host(const size_t rank_dynamic, const size_t rank,
                               const index_list_type minIndices, const std::string & label)
    {
  bool isBad = false;
      std::string message = "Kokkos::OffsetView ERROR: for OffsetView labeled '" + label + "':";
      if (rank_dynamic != rank) {
          message += "The full rank must be the same as the dynamic rank. full rank = ";
          message += std::to_string(rank) + " dynamic rank = " + std::to_string(rank_dynamic) + "\n";
          isBad = true;
      }

      size_t numOffsets = 0;
      for(size_t i = 0; i < minIndices.size(); ++i ){
          if( minIndices.begin()[i] != -KOKKOS_INVALID_OFFSET) numOffsets++;
      }
      if (numOffsets != rank_dynamic) {
          message += "The number of offsets provided ( " + std::to_string(numOffsets) +
              " ) must equal the dynamic rank ( " + std::to_string(rank_dynamic) + " ).";
          isBad = true;
      }

      if(isBad) Kokkos::abort(message.c_str());
    }
#endif

KOKKOS_INLINE_FUNCTION
void runtime_check_rank_device(const size_t rank_dynamic, const size_t rank,
                               const index_list_type minIndices)
    {
      if (rank_dynamic != rank) {
          Kokkos::abort("The full rank of an OffsetView must be the same as the dynamic rank.");
      }
      size_t numOffsets = 0;
      for(size_t i = 0; i < minIndices.size(); ++i ){
          if( minIndices.begin()[i] != -KOKKOS_INVALID_OFFSET) numOffsets++;
      }
      if (numOffsets != rank) {
          Kokkos::abort("The number of offsets provided to an OffsetView constructor must equal the dynamic rank.");
      }

    }
  }

  template< class DataType , class ... Properties >
  class OffsetView : public ViewTraits< DataType , Properties ... > {
  public:

    typedef ViewTraits< DataType , Properties ... > traits ;



  private:

    template< class , class ... > friend class OffsetView ;
    template< class , class ... > friend class View ;
    template< class , class ... > friend class Kokkos::Impl::ViewMapping ;


    typedef Kokkos::Impl::ViewMapping< traits , void > map_type ;
    typedef Kokkos::Impl::SharedAllocationTracker      track_type ;
  public:
    enum { Rank = map_type::Rank };
    typedef Kokkos::Array<int64_t, Rank>  begins_type ;


    template <typename iType, typename std::enable_if< std::is_integral<iType>::value, iType>::type = 0>
    KOKKOS_INLINE_FUNCTION
    int64_t begin(const iType dimension) const {
      return dimension < Rank ? m_begins[dimension] : 0;
    }

    template <typename iType, typename std::enable_if< std::is_integral<iType>::value, iType>::type = 0>
    KOKKOS_INLINE_FUNCTION
    int64_t end(const iType dimension) const {return begin(dimension) + m_map.extent(dimension);}


  private:
    track_type  m_track ;
    map_type    m_map ;
    begins_type  m_begins;

  public:
    //----------------------------------------
    /** \brief  Compatible view of array of scalar types */
    typedef OffsetView< typename traits::scalar_array_type ,
        typename traits::array_layout ,
        typename traits::device_type ,
        typename traits::memory_traits >
    array_type ;

    /** \brief  Compatible view of const data type */
    typedef OffsetView< typename traits::const_data_type ,
        typename traits::array_layout ,
        typename traits::device_type ,
        typename traits::memory_traits >
    const_type ;

    /** \brief  Compatible view of non-const data type */
    typedef OffsetView< typename traits::non_const_data_type ,
        typename traits::array_layout ,
        typename traits::device_type ,
        typename traits::memory_traits >
    non_const_type ;

    /** \brief  Compatible HostMirror view */
    typedef OffsetView< typename traits::non_const_data_type ,
        typename traits::array_layout ,
        typename traits::host_mirror_space >
    HostMirror ;

    //----------------------------------------
    // Domain rank and extents

    /** \brief rank() to be implemented
     */
    //KOKKOS_INLINE_FUNCTION
    //static
    //constexpr unsigned rank() { return map_type::Rank; }

    template< typename iType >
    KOKKOS_INLINE_FUNCTION constexpr
    typename std::enable_if< std::is_integral<iType>::value , size_t >::type
    extent( const iType & r ) const
    { return m_map.extent(r); }

    template< typename iType >
    KOKKOS_INLINE_FUNCTION constexpr
    typename std::enable_if< std::is_integral<iType>::value , int >::type
    extent_int( const iType & r ) const
    { return static_cast<int>(m_map.extent(r)); }

    KOKKOS_INLINE_FUNCTION constexpr
    typename traits::array_layout layout() const
    { return m_map.layout(); }


    KOKKOS_INLINE_FUNCTION constexpr size_t size() const { return m_map.dimension_0() *
        m_map.dimension_1() *
        m_map.dimension_2() *
        m_map.dimension_3() *
        m_map.dimension_4() *
        m_map.dimension_5() *
        m_map.dimension_6() *
        m_map.dimension_7(); }

    KOKKOS_INLINE_FUNCTION constexpr size_t stride_0() const { return m_map.stride_0(); }
    KOKKOS_INLINE_FUNCTION constexpr size_t stride_1() const { return m_map.stride_1(); }
    KOKKOS_INLINE_FUNCTION constexpr size_t stride_2() const { return m_map.stride_2(); }
    KOKKOS_INLINE_FUNCTION constexpr size_t stride_3() const { return m_map.stride_3(); }
    KOKKOS_INLINE_FUNCTION constexpr size_t stride_4() const { return m_map.stride_4(); }
    KOKKOS_INLINE_FUNCTION constexpr size_t stride_5() const { return m_map.stride_5(); }
    KOKKOS_INLINE_FUNCTION constexpr size_t stride_6() const { return m_map.stride_6(); }
    KOKKOS_INLINE_FUNCTION constexpr size_t stride_7() const { return m_map.stride_7(); }

    template< typename iType >
    KOKKOS_INLINE_FUNCTION constexpr
    typename std::enable_if< std::is_integral<iType>::value , size_t >::type
    stride(iType r) const {
      return (r == 0 ? m_map.stride_0() :
          (r == 1 ? m_map.stride_1() :
              (r == 2 ? m_map.stride_2() :
                  (r == 3 ? m_map.stride_3() :
                      (r == 4 ? m_map.stride_4() :
                          (r == 5 ? m_map.stride_5() :
                              (r == 6 ? m_map.stride_6() :
                                  m_map.stride_7())))))));
    }

    template< typename iType >
    KOKKOS_INLINE_FUNCTION void stride( iType * const s ) const { m_map.stride(s); }

    //----------------------------------------
    // Range span is the span which contains all members.

    typedef typename map_type::reference_type  reference_type ;
    typedef typename map_type::pointer_type    pointer_type ;

    enum { reference_type_is_lvalue_reference = std::is_lvalue_reference< reference_type >::value };

    KOKKOS_INLINE_FUNCTION constexpr size_t span() const { return m_map.span(); }
    KOKKOS_INLINE_FUNCTION bool span_is_contiguous() const { return m_map.span_is_contiguous(); }
    KOKKOS_INLINE_FUNCTION constexpr pointer_type data() const { return m_map.data(); }

    //----------------------------------------
    // Allow specializations to query their specialized map

    KOKKOS_INLINE_FUNCTION
    const Kokkos::Impl::ViewMapping< traits , void > &
    implementation_map() const { return m_map ; }

    //----------------------------------------

  private:

    enum {
      is_layout_left = std::is_same< typename traits::array_layout
      , Kokkos::LayoutLeft >::value ,

      is_layout_right = std::is_same< typename traits::array_layout
      , Kokkos::LayoutRight >::value ,

      is_layout_stride = std::is_same< typename traits::array_layout
      , Kokkos::LayoutStride >::value ,

      is_default_map =
          std::is_same< typename traits::specialize , void >::value &&
          ( is_layout_left || is_layout_right || is_layout_stride )
    };

    template< class Space , bool = Kokkos::Impl::MemorySpaceAccess< Space , typename traits::memory_space >::accessible > struct verify_space
        { KOKKOS_FORCEINLINE_FUNCTION static void check() {} };

    template< class Space > struct verify_space<Space,false>
    { KOKKOS_FORCEINLINE_FUNCTION static void check()
    { Kokkos::abort("Kokkos::View ERROR: attempt to access inaccessible memory space");
    };
    };

#if defined( KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK )

#define KOKKOS_IMPL_OFFSETVIEW_OPERATOR_VERIFY( ARG ) \
    OffsetView::template verify_space< Kokkos::Impl::ActiveExecutionMemorySpace >::check(); \
    Kokkos::Impl::offsetview_verify_operator_bounds< typename traits::memory_space > ARG ;

#else

#define KOKKOS_IMPL_OFFSETVIEW_OPERATOR_VERIFY( ARG ) \
    OffsetView::template verify_space< Kokkos::Impl::ActiveExecutionMemorySpace >::check();

#endif
  public:

    //------------------------------
    // Rank 0 operator()

    KOKKOS_FORCEINLINE_FUNCTION
    reference_type
    operator()() const
    {
      return m_map.reference();
    }
    //------------------------------
    // Rank 1 operator()


    template< typename I0>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
    ( Kokkos::Impl::are_integral<I0>::value
        && ( 1 == Rank )
        && ! is_default_map
    ), reference_type >::type
    operator()( const I0 & i0) const
    {

      KOKKOS_IMPL_OFFSETVIEW_OPERATOR_VERIFY( (m_track,m_map, m_begins, i0) )
      const size_t j0 = i0 - m_begins[0];
      return m_map.reference(j0);
    }

    template< typename I0>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
    ( Kokkos::Impl::are_integral<I0>::value
        && ( 1 == Rank )
        && is_default_map
        && ! is_layout_stride
    ), reference_type >::type
    operator()( const I0 & i0 ) const
    {
      KOKKOS_IMPL_OFFSETVIEW_OPERATOR_VERIFY( (m_track,m_map, m_begins, i0) )
      const size_t j0 = i0 - m_begins[0];
      return m_map.m_impl_handle[ j0 ];
    }

    template< typename I0 >
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
    ( Kokkos::Impl::are_integral<I0>::value
        && ( 1 == Rank )
        && is_default_map
        && is_layout_stride
    ), reference_type >::type
    operator()( const I0 & i0) const
    {
      KOKKOS_IMPL_OFFSETVIEW_OPERATOR_VERIFY( (m_track,m_map, m_begins, i0) )
      const size_t j0 = i0 - m_begins[0];
      return m_map.m_impl_handle[ m_map.m_impl_offset.m_stride.S0 * j0 ];
    }
    //------------------------------
    // Rank 1 operator[]

    template< typename I0 >
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
    ( Kokkos::Impl::are_integral<I0>::value
        && ( 1 == Rank )
        && ! is_default_map
    ), reference_type >::type
    operator[]( const I0 & i0 ) const
    {
      KOKKOS_IMPL_OFFSETVIEW_OPERATOR_VERIFY( (m_track,m_map, m_begins, i0) )
      const size_t j0 = i0 - m_begins[0];
      return m_map.reference(j0);
    }

    template< typename I0 >
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
    ( Kokkos::Impl::are_integral<I0>::value
        && ( 1 == Rank )
        && is_default_map
        && ! is_layout_stride
    ), reference_type >::type
    operator[]( const I0 & i0 ) const
    {
      KOKKOS_IMPL_OFFSETVIEW_OPERATOR_VERIFY( (m_track,m_map, m_begins, i0) )
      const size_t j0 = i0 - m_begins[0];
      return m_map.m_impl_handle[ j0 ];
    }

    template< typename I0 >
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
    ( Kokkos::Impl::are_integral<I0>::value
        && ( 1 == Rank )
        && is_default_map
        && is_layout_stride
    ), reference_type >::type
    operator[]( const I0 & i0 ) const
    {
      KOKKOS_IMPL_OFFSETVIEW_OPERATOR_VERIFY( (m_track,m_map, m_begins, i0) )
      const size_t j0 = i0 - m_begins[0];
      return m_map.m_impl_handle[ m_map.m_impl_offset.m_stride.S0 * j0 ];
    }


    //------------------------------
    // Rank 2

    template< typename I0 , typename I1 >
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
    ( Kokkos::Impl::are_integral<I0,I1>::value
        && ( 2 == Rank )
        && ! is_default_map
    ), reference_type >::type
    operator()( const I0 & i0 , const I1 & i1) const
    {
      KOKKOS_IMPL_OFFSETVIEW_OPERATOR_VERIFY( (m_track,m_map, m_begins, i0,i1) )
      const size_t j0 = i0 - m_begins[0];
      const size_t j1 = i1 - m_begins[1];
      return m_map.reference(j0,j1);
    }

    template< typename I0 , typename I1 >
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
    ( Kokkos::Impl::are_integral<I0,I1>::value
        && ( 2 == Rank )
        && is_default_map
        && is_layout_left && ( traits::rank_dynamic == 0 )
    ), reference_type >::type
    operator()( const I0 & i0 , const I1 & i1) const
    {
      KOKKOS_IMPL_OFFSETVIEW_OPERATOR_VERIFY( (m_track,m_map, m_begins, i0,i1) )
      const size_t j0 = i0 - m_begins[0];
      const size_t j1 = i1 - m_begins[1];
      return m_map.m_impl_handle[ j0 + m_map.m_impl_offset.m_dim.N0 * j1 ];
    }

    template< typename I0 , typename I1>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
    ( Kokkos::Impl::are_integral<I0,I1>::value
        && ( 2 == Rank )
        && is_default_map
        && is_layout_left && ( traits::rank_dynamic != 0 )
    ), reference_type >::type
    operator()( const I0 & i0 , const I1 & i1) const
    {
      KOKKOS_IMPL_OFFSETVIEW_OPERATOR_VERIFY( (m_track,m_map, m_begins, i0,i1) )
      const size_t j0 = i0 - m_begins[0];
      const size_t j1 = i1 - m_begins[1];
      return m_map.m_impl_handle[ j0 + m_map.m_impl_offset.m_stride * j1 ];
    }

    template< typename I0 , typename I1 >
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
    ( Kokkos::Impl::are_integral<I0,I1>::value
        && ( 2 == Rank )
        && is_default_map
        && is_layout_right && ( traits::rank_dynamic == 0 )
    ), reference_type >::type
    operator()( const I0 & i0 , const I1 & i1 ) const
    {
      KOKKOS_IMPL_OFFSETVIEW_OPERATOR_VERIFY( (m_track,m_map, m_begins, i0,i1) )
      const size_t j0 = i0 - m_begins[0];
      const size_t j1 = i1 - m_begins[1];
      return m_map.m_impl_handle[ j1 + m_map.m_impl_offset.m_dim.N1 * j0 ];
    }

    template< typename I0 , typename I1 >
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
    ( Kokkos::Impl::are_integral<I0,I1>::value
        && ( 2 == Rank )
        && is_default_map
        && is_layout_right && ( traits::rank_dynamic != 0 )
    ), reference_type >::type
    operator()( const I0 & i0 , const I1 & i1 ) const
    {
      KOKKOS_IMPL_OFFSETVIEW_OPERATOR_VERIFY( (m_track,m_map, m_begins, i0,i1) )
      const size_t j0 = i0 - m_begins[0];
      const size_t j1 = i1 - m_begins[1];
      return m_map.m_impl_handle[ j1 + m_map.m_impl_offset.m_stride * j0 ];
    }

    template< typename I0 , typename I1>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
    ( Kokkos::Impl::are_integral<I0,I1>::value
        && ( 2 == Rank )
        && is_default_map
        && is_layout_stride
    ), reference_type >::type
    operator()( const I0 & i0 , const I1 & i1 ) const
    {
      KOKKOS_IMPL_OFFSETVIEW_OPERATOR_VERIFY( (m_track,m_map, m_begins, i0,i1) )
      const size_t j0 = i0 - m_begins[0];
      const size_t j1 = i1 - m_begins[1];
      return m_map.m_impl_handle[ j0 * m_map.m_impl_offset.m_stride.S0 +
                                  j1 * m_map.m_impl_offset.m_stride.S1 ];
    }

    //------------------------------
    // Rank 3

    template< typename I0 , typename I1 , typename I2 >
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
    ( Kokkos::Impl::are_integral<I0,I1,I2>::value
        && ( 3 == Rank )
        && is_default_map
    ), reference_type >::type
    operator()( const I0 & i0 , const I1 & i1 , const I2 & i2) const
    {
      KOKKOS_IMPL_OFFSETVIEW_OPERATOR_VERIFY( (m_track,m_map, m_begins, i0,i1, i2) )
      const size_t j0 = i0 - m_begins[0];
      const size_t j1 = i1 - m_begins[1];
      const size_t j2 = i2 - m_begins[2];
      return m_map.m_impl_handle[ m_map.m_impl_offset(j0, j1, j2) ];
    }

    template< typename I0 , typename I1 , typename I2>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
    ( Kokkos::Impl::are_integral<I0,I1,I2>::value
        && ( 3 == Rank )
        && ! is_default_map
    ), reference_type >::type
    operator()( const I0 & i0 , const I1 & i1 , const I2 & i2) const
    {
      KOKKOS_IMPL_OFFSETVIEW_OPERATOR_VERIFY( (m_track,m_map,m_begins, i0,i1, i2) )
      const size_t j0 = i0 - m_begins[0];
      const size_t j1 = i1 - m_begins[1];
      const size_t j2 = i2 - m_begins[2];
      return m_map.reference(j0, j1, j2);
    }

    //------------------------------
    // Rank 4

    template< typename I0 , typename I1 , typename I2 , typename I3>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
    ( Kokkos::Impl::are_integral<I0,I1,I2,I3>::value
        && ( 4 == Rank )
        && is_default_map
    ), reference_type >::type
    operator()( const I0 & i0 , const I1 & i1 , const I2 & i2 , const I3 & i3) const
    {
      KOKKOS_IMPL_OFFSETVIEW_OPERATOR_VERIFY( (m_track,m_map, m_begins, i0,i1, i2, i3) )
      const size_t j0 = i0 - m_begins[0];
      const size_t j1 = i1 - m_begins[1];
      const size_t j2 = i2 - m_begins[2];
      const size_t j3 = i3 - m_begins[3];
      return m_map.m_impl_handle[ m_map.m_impl_offset(j0,j1,j2,j3) ];
    }

    template< typename I0 , typename I1 , typename I2 , typename I3 >
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
    ( Kokkos::Impl::are_integral<I0,I1,I2,I3>::value
        && ( 4 == Rank )
        && ! is_default_map
    ), reference_type >::type
    operator()( const I0 & i0 , const I1 & i1 , const I2 & i2 , const I3 & i3) const
    {
      KOKKOS_IMPL_OFFSETVIEW_OPERATOR_VERIFY( (m_track,m_map, m_begins, i0,i1, i2, i3) )
      const size_t j0 = i0 - m_begins[0];
      const size_t j1 = i1 - m_begins[1];
      const size_t j2 = i2 - m_begins[2];
      const size_t j3 = i3 - m_begins[3];
      return m_map.reference(j0,j1,j2,j3);
    }

    //------------------------------
    // Rank 5

    template< typename I0 , typename I1 , typename I2 , typename I3
    , typename I4>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
    ( Kokkos::Impl::are_integral<I0,I1,I2,I3,I4>::value
        && ( 5 == Rank )
        && is_default_map
    ), reference_type >::type
    operator()( const I0 & i0 , const I1 & i1 , const I2 & i2 , const I3 & i3
                , const I4 & i4 ) const
    {
      KOKKOS_IMPL_OFFSETVIEW_OPERATOR_VERIFY( (m_track,m_map, m_begins, i0,i1, i2, i3, i4) )
      const size_t j0 = i0 - m_begins[0];
      const size_t j1 = i1 - m_begins[1];
      const size_t j2 = i2 - m_begins[2];
      const size_t j3 = i3 - m_begins[3];
      const size_t j4 = i4 - m_begins[4];
      return m_map.m_impl_handle[ m_map.m_impl_offset(j0, j1,j2, j3, j4) ];
    }

    template< typename I0 , typename I1 , typename I2 , typename I3
    , typename I4>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
    ( Kokkos::Impl::are_integral<I0,I1,I2,I3,I4>::value
        && ( 5 == Rank )
        && ! is_default_map
    ), reference_type >::type
    operator()( const I0 & i0 , const I1 & i1 , const I2 & i2 , const I3 & i3
                , const I4 & i4) const
    {
      KOKKOS_IMPL_OFFSETVIEW_OPERATOR_VERIFY( (m_track,m_map, m_begins, i0,i1, i2, i3, i4) )
      const size_t j0 = i0 - m_begins[0];
      const size_t j1 = i1 - m_begins[1];
      const size_t j2 = i2 - m_begins[2];
      const size_t j3 = i3 - m_begins[3];
      const size_t j4 = i4 - m_begins[4];
      return m_map.reference(j0,j1,j2,j3,j4);
    }

    //------------------------------
    // Rank 6

    template< typename I0 , typename I1 , typename I2 , typename I3
    , typename I4 , typename I5 >
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
    ( Kokkos::Impl::are_integral<I0,I1,I2,I3,I4,I5>::value
        && ( 6 == Rank )
        && is_default_map
    ), reference_type >::type
    operator()( const I0 & i0 , const I1 & i1 , const I2 & i2 , const I3 & i3
                , const I4 & i4 , const I5 & i5 ) const
    {
      KOKKOS_IMPL_OFFSETVIEW_OPERATOR_VERIFY( (m_track,m_map,m_begins, i0,i1, i2, i3, i4, i5) )
      const size_t j0 = i0 - m_begins[0];
      const size_t j1 = i1 - m_begins[1];
      const size_t j2 = i2 - m_begins[2];
      const size_t j3 = i3 - m_begins[3];
      const size_t j4 = i4 - m_begins[4];
      const size_t j5 = i5 - m_begins[5];
      return m_map.m_impl_handle[ m_map.m_impl_offset(j0,j1,j2,j3,j4,j5) ];
    }

    template< typename I0 , typename I1 , typename I2 , typename I3
    , typename I4 , typename I5>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
    ( Kokkos::Impl::are_integral<I0,I1,I2,I3,I4,I5>::value
        && ( 6 == Rank )
        && ! is_default_map
    ), reference_type >::type
    operator()( const I0 & i0 , const I1 & i1 , const I2 & i2 , const I3 & i3
                , const I4 & i4 , const I5 & i5) const
    {
      KOKKOS_IMPL_OFFSETVIEW_OPERATOR_VERIFY( (m_track,m_map,m_begins, i0,i1, i2, i3, i4, i5) )
      const size_t j0 = i0 - m_begins[0];
      const size_t j1 = i1 - m_begins[1];
      const size_t j2 = i2 - m_begins[2];
      const size_t j3 = i3 - m_begins[3];
      const size_t j4 = i4 - m_begins[4];
      const size_t j5 = i5 - m_begins[5];
      return m_map.reference(j0,j1,j2,j3,j4,j5);
    }

    //------------------------------
    // Rank 7

    template< typename I0 , typename I1 , typename I2 , typename I3
    , typename I4 , typename I5 , typename I6>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
    ( Kokkos::Impl::are_integral<I0,I1,I2,I3,I4,I5,I6>::value
        && ( 7 == Rank )
        && is_default_map
    ), reference_type >::type
    operator()( const I0 & i0 , const I1 & i1 , const I2 & i2 , const I3 & i3
                , const I4 & i4 , const I5 & i5 , const I6 & i6) const
    {
      KOKKOS_IMPL_OFFSETVIEW_OPERATOR_VERIFY( (m_track,m_map,m_begins, i0,i1, i2, i3, i4, i5, i6) )
      const size_t j0 = i0 - m_begins[0];
      const size_t j1 = i1 - m_begins[1];
      const size_t j2 = i2 - m_begins[2];
      const size_t j3 = i3 - m_begins[3];
      const size_t j4 = i4 - m_begins[4];
      const size_t j5 = i5 - m_begins[5];
      const size_t j6 = i6 - m_begins[6];
      return m_map.m_impl_handle[ m_map.m_impl_offset(j0,j1,j2,j3,j4,j5,j6) ];
    }

    template< typename I0 , typename I1 , typename I2 , typename I3
    , typename I4 , typename I5 , typename I6 >
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
    ( Kokkos::Impl::are_integral<I0,I1,I2,I3,I4,I5,I6>::value
        && ( 7 == Rank )
        && ! is_default_map
    ), reference_type >::type
    operator()( const I0 & i0 , const I1 & i1 , const I2 & i2 , const I3 & i3
                , const I4 & i4 , const I5 & i5 , const I6 & i6) const
    {
      KOKKOS_IMPL_OFFSETVIEW_OPERATOR_VERIFY( (m_track,m_map,m_begins, i0,i1, i2, i3, i4, i5, i6) )
      const size_t j0 = i0 - m_begins[0];
      const size_t j1 = i1 - m_begins[1];
      const size_t j2 = i2 - m_begins[2];
      const size_t j3 = i3 - m_begins[3];
      const size_t j4 = i4 - m_begins[4];
      const size_t j5 = i5 - m_begins[5];
      const size_t j6 = i6 - m_begins[6];
      return m_map.reference(j0,j1,j2,j3,j4,j5,j6);
    }

    //------------------------------
    // Rank 8

    template< typename I0 , typename I1 , typename I2 , typename I3
    , typename I4 , typename I5 , typename I6 , typename I7 >
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
    ( Kokkos::Impl::are_integral<I0,I1,I2,I3,I4,I5,I6,I7>::value
        && ( 8 == Rank )
        && is_default_map
    ), reference_type >::type
    operator()( const I0 & i0 , const I1 & i1 , const I2 & i2 , const I3 & i3
                , const I4 & i4 , const I5 & i5 , const I6 & i6 , const I7 & i7) const
    {
      KOKKOS_IMPL_OFFSETVIEW_OPERATOR_VERIFY( (m_track,m_map,m_begins, i0,i1, i2, i3, i4, i5, i6, i7) )
      const size_t j0 = i0 - m_begins[0];
      const size_t j1 = i1 - m_begins[1];
      const size_t j2 = i2 - m_begins[2];
      const size_t j3 = i3 - m_begins[3];
      const size_t j4 = i4 - m_begins[4];
      const size_t j5 = i5 - m_begins[5];
      const size_t j6 = i6 - m_begins[6];
      const size_t j7 = i7 - m_begins[7];
      return m_map.m_impl_handle[ m_map.m_impl_offset(j0,j1,j2,j3,j4,j5,j6,j7) ];
    }

    template< typename I0 , typename I1 , typename I2 , typename I3
    , typename I4 , typename I5 , typename I6 , typename I7>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
    ( Kokkos::Impl::are_integral<I0,I1,I2,I3,I4,I5,I6,I7>::value
        && ( 8 == Rank )
        && ! is_default_map
    ), reference_type >::type
    operator()( const I0 & i0 , const I1 & i1 , const I2 & i2 , const I3 & i3
                , const I4 & i4 , const I5 & i5 , const I6 & i6 , const I7 & i7 ) const
    {
      KOKKOS_IMPL_OFFSETVIEW_OPERATOR_VERIFY( (m_track,m_map,m_begins, i0,i1, i2, i3, i4, i5, i6, i7) )
      const size_t j0 = i0 - m_begins[0];
      const size_t j1 = i1 - m_begins[1];
      const size_t j2 = i2 - m_begins[2];
      const size_t j3 = i3 - m_begins[3];
      const size_t j4 = i4 - m_begins[4];
      const size_t j5 = i5 - m_begins[5];
      const size_t j6 = i6 - m_begins[6];
      const size_t j7 = i7 - m_begins[7];
      return m_map.reference(j0,j1,j2,j3,j4,j5,j6,j7);
    }


#undef KOKKOS_IMPL_OFFSETVIEW_OPERATOR_VERIFY

    //----------------------------------------
    // Standard destructor, constructors, and assignment operators

    KOKKOS_INLINE_FUNCTION
    ~OffsetView() {}

    KOKKOS_INLINE_FUNCTION
    OffsetView() : m_track(), m_map() {

      for(size_t i = 0; i < Rank; ++i) m_begins[i] = KOKKOS_INVALID_INDEX;
    }

    KOKKOS_INLINE_FUNCTION
    OffsetView( const OffsetView & rhs ) : m_track( rhs.m_track, traits::is_managed ), m_map( rhs.m_map ),
    m_begins(rhs.m_begins) {}

    KOKKOS_INLINE_FUNCTION
    OffsetView( OffsetView && rhs ) : m_track( std::move(rhs.m_track) ),
    m_map( std::move(rhs.m_map)), m_begins(std::move(rhs.m_begins)) {}

    KOKKOS_INLINE_FUNCTION
    OffsetView & operator = ( const OffsetView & rhs ) {
      m_track = rhs.m_track ;
      m_map = rhs.m_map ;
      m_begins = rhs.m_begins;
      return *this ;
    }

    KOKKOS_INLINE_FUNCTION
    OffsetView & operator = ( OffsetView && rhs ) {
      m_track = std::move(rhs.m_track) ;
      m_map = std::move(rhs.m_map) ;
      m_begins = std::move(rhs.m_begins) ;
      return *this ;
    }

    //interoperability with View
  private:
    typedef View< typename traits::scalar_array_type ,
        typename traits::array_layout ,
        typename traits::device_type ,
        typename traits::memory_traits > view_type;
  public:

    KOKKOS_INLINE_FUNCTION
    view_type view() {

      view_type v(m_track, m_map);
      return v ;
    }

    template<class RT, class ... RP>
    KOKKOS_INLINE_FUNCTION
    OffsetView( const View<RT, RP...> & view) :
                m_track(view.impl_track()), m_map(){

      typedef typename OffsetView<RT,RP...>::traits  SrcTraits ;
      typedef Kokkos::Impl::ViewMapping< traits , SrcTraits , void >  Mapping ;
       static_assert( Mapping::is_assignable , "Incompatible OffsetView copy construction" );
       Mapping::assign( m_map , view.impl_map() , m_track );

      for (size_t i = 0; i < view.Rank; ++i) {
          m_begins[i] = 0;
      }
    }

    template<class RT, class ... RP>
    KOKKOS_INLINE_FUNCTION
    OffsetView( const View<RT, RP...> & view
                ,const index_list_type & minIndices) :
                m_track(view.impl_track()), m_map(){

      typedef typename OffsetView<RT,RP...>::traits  SrcTraits ;
      typedef Kokkos::Impl::ViewMapping< traits , SrcTraits , void >  Mapping ;
       static_assert( Mapping::is_assignable , "Incompatible OffsetView copy construction" );
       Mapping::assign( m_map , view.impl_map() , m_track );

#ifdef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
          Impl::runtime_check_rank_host(traits::rank_dynamic, Rank, minIndices, label());
#else
          Impl::runtime_check_rank_device(traits::rank_dynamic, Rank, minIndices);

#endif

      for (size_t i = 0; i < minIndices.size(); ++i) {
          m_begins[i] = minIndices.begin()[i];
      }
    }

    // may assign unmanaged from managed.

#if 0
    template< class RT , class ... RP >
    KOKKOS_INLINE_FUNCTION
    OffsetView( const OffsetView<RT,RP...> & rhs )
    : m_track( rhs.m_track , traits::is_managed )
    , m_map()
    , m_begins(rhs.m_begins)
    {
      typedef typename OffsetView<RT,RP...>::traits  SrcTraits ;
      typedef Kokkos::Impl::ViewMapping< traits , SrcTraits , void >  Mapping ;
      static_assert( Mapping::is_assignable , "Incompatible OffsetView copy construction" );
      Mapping::assign( m_map , rhs.m_map , rhs.m_track );  //swb what about assign?
    }

    template< class RT , class ... RP >
    KOKKOS_INLINE_FUNCTION
    View & operator = ( const View<RT,RP...> & rhs )
    {
      typedef typename View<RT,RP...>::traits  SrcTraits ;
      typedef Kokkos::Impl::ViewMapping< traits , SrcTraits , void >  Mapping ;
      static_assert( Mapping::is_assignable , "Incompatible View copy assignment" );
      Mapping::assign( m_map , rhs.m_map , rhs.m_track );
      m_track.assign( rhs.m_track , traits::is_managed );
      return *this ;
    }

    //----------------------------------------
    // Compatible subview constructor
    // may assign unmanaged from managed.

    template< class RT , class ... RP , class Arg0 , class ... Args >
    KOKKOS_INLINE_FUNCTION
    View( const View< RT , RP... > & src_view
          , const Arg0 & arg0 , Args ... args )
    : m_track( src_view.m_track , traits::is_managed )
    , m_map()
    {
      typedef View< RT , RP... > SrcType ;

      typedef Kokkos::Impl::ViewMapping
          < void /* deduce destination view type from source view traits */
          , typename SrcType::traits
          , Arg0 , Args... > Mapping ;

      typedef typename Mapping::type DstType ;

      static_assert( Kokkos::Impl::ViewMapping< traits , typename DstType::traits , void >::is_assignable
                     , "Subview construction requires compatible view and subview arguments" );

      Mapping::assign( m_map, src_view.m_map, arg0 , args... );
    }
#endif

    //----------------------------------------
    // Allocation tracking properties
    KOKKOS_INLINE_FUNCTION
    int use_count() const
    { return m_track.use_count(); }

    inline
    const std::string label() const
    { return m_track.template get_label< typename traits::memory_space >(); }

#if 0
    //----------------------------------------
    // Allocation according to allocation properties and array layout

    template< class ... P >
    explicit inline
    View( const Impl::ViewCtorProp< P ... > & arg_prop
          , typename std::enable_if< ! Impl::ViewCtorProp< P... >::has_pointer
          , typename traits::array_layout
          >::type const & arg_layout
    )
    : m_track()
    , m_map()
    {
      // Append layout and spaces if not input
      typedef Impl::ViewCtorProp< P ... > alloc_prop_input ;

      // use 'std::integral_constant<unsigned,I>' for non-types
      // to avoid duplicate class error.
      typedef Impl::ViewCtorProp
          < P ...
          , typename std::conditional
          < alloc_prop_input::has_label
          , std::integral_constant<unsigned,0>
      , typename std::string
      >::type
      , typename std::conditional
      < alloc_prop_input::has_memory_space
      , std::integral_constant<unsigned,1>
      , typename traits::device_type::memory_space
      >::type
      , typename std::conditional
      < alloc_prop_input::has_execution_space
      , std::integral_constant<unsigned,2>
      , typename traits::device_type::execution_space
      >::type
      > alloc_prop ;

      static_assert( traits::is_managed
                     , "OffsetView allocation constructor requires managed memory" );

      if ( alloc_prop::initialize &&
          ! alloc_prop::execution_space::impl_is_initialized()
      ) {
          // If initializing view data then
          // the execution space must be initialized.
          Kokkos::Impl::throw_runtime_exception("Constructing OffsetView and initializing data with uninitialized execution space");
      }

      // Copy the input allocation properties with possibly defaulted properties
      alloc_prop prop( arg_prop );

      //------------------------------------------------------------
#if defined( KOKKOS_ENABLE_CUDA )
      // If allocating in CudaUVMSpace must fence before and after
      // the allocation to protect against possible concurrent access
      // on the CPU and the GPU.
      // Fence using the trait's executon space (which will be Kokkos::Cuda)
      // to avoid incomplete type errors from usng Kokkos::Cuda directly.
      if ( std::is_same< Kokkos::CudaUVMSpace , typename traits::device_type::memory_space >::value ) {
          traits::device_type::memory_space::execution_space::fence();
      }
#endif
      //------------------------------------------------------------

      Kokkos::Impl::SharedAllocationRecord<> *
      record = m_map.allocate_shared( prop , arg_layout );

      //------------------------------------------------------------
#if defined( KOKKOS_ENABLE_CUDA )
      if ( std::is_same< Kokkos::CudaUVMSpace , typename traits::device_type::memory_space >::value ) {
          traits::device_type::memory_space::execution_space::fence();
      }
#endif
      //------------------------------------------------------------

      // Setup and initialization complete, start tracking
      m_track.assign_allocated_record_to_uninitialized( record );
    }

    KOKKOS_INLINE_FUNCTION
    void assign_data( pointer_type arg_data )
    {
      m_track.clear();
      m_map.assign_data( arg_data );
    }

    // Wrap memory according to properties and array layout
    template< class ... P >
    explicit KOKKOS_INLINE_FUNCTION
    View( const Impl::ViewCtorProp< P ... > & arg_prop
          , typename std::enable_if< Impl::ViewCtorProp< P... >::has_pointer
          , typename traits::array_layout
          >::type const & arg_layout
    )
    : m_track() // No memory tracking
    , m_map( arg_prop , arg_layout )
    {
      static_assert(
          std::is_same< pointer_type
          , typename Impl::ViewCtorProp< P... >::pointer_type
          >::value ,
          "Constructing OffsetView to wrap user memory must supply matching pointer type" );
    }

    // Simple dimension-only layout
    template< class ... P >
    explicit inline
    View( const Impl::ViewCtorProp< P ... > & arg_prop
          , typename std::enable_if< ! Impl::ViewCtorProp< P... >::has_pointer
          , size_t
          >::type const arg_N0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
    )
    : View( arg_prop
            , typename traits::array_layout
            ( arg_N0 , arg_N1 , arg_N2 , arg_N3
              , arg_N4 , arg_N5 , arg_N6 , arg_N7 )
    )
    {
#ifdef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
      Impl::runtime_check_rank_host(traits::rank_dynamic, arg_N0, arg_N1, arg_N2, arg_N3,
                                    arg_N4, arg_N5, arg_N6, arg_N7, label());
#else
      Impl::runtime_check_rank_device(traits::rank_dynamic, arg_N0, arg_N1, arg_N2, arg_N3,
                                      arg_N4, arg_N5, arg_N6, arg_N7);

#endif

    }

    template< class ... P >
    explicit KOKKOS_INLINE_FUNCTION
    View( const Impl::ViewCtorProp< P ... > & arg_prop
          , typename std::enable_if< Impl::ViewCtorProp< P... >::has_pointer
          , size_t
          >::type const arg_N0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
    )
    : View( arg_prop
            , typename traits::array_layout
            ( arg_N0 , arg_N1 , arg_N2 , arg_N3
              , arg_N4 , arg_N5 , arg_N6 , arg_N7 )
    )
    {
#ifdef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
      Impl::runtime_check_rank_host(traits::rank_dynamic, arg_N0, arg_N1, arg_N2, arg_N3,
                                    arg_N4, arg_N5, arg_N6, arg_N7, label());
#else
      Impl::runtime_check_rank_device(traits::rank_dynamic, arg_N0, arg_N1, arg_N2, arg_N3,
                                      arg_N4, arg_N5, arg_N6, arg_N7);

#endif

    }

    // Allocate with label and layout
    template< typename Label >
    explicit inline
    View( const Label & arg_label
          , typename std::enable_if<
          Kokkos::Impl::is_view_label<Label>::value ,
          typename traits::array_layout >::type const & arg_layout
    )
    : View( Impl::ViewCtorProp< std::string >( arg_label ) , arg_layout )
    {}

    // Allocate label and layout, must disambiguate from subview constructor.
    template< typename Label >
    explicit inline
    View( const Label & arg_label
          , typename std::enable_if<
          Kokkos::Impl::is_view_label<Label>::value ,
          const size_t >::type arg_N0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
    )
    : View( Impl::ViewCtorProp< std::string >( arg_label )
            , typename traits::array_layout
            ( arg_N0 , arg_N1 , arg_N2 , arg_N3
              , arg_N4 , arg_N5 , arg_N6 , arg_N7 )
    )
    {

#ifdef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
      Impl::runtime_check_rank_host(traits::rank_dynamic, arg_N0, arg_N1, arg_N2, arg_N3,
                                    arg_N4, arg_N5, arg_N6, arg_N7, label());
#else
      Impl::runtime_check_rank_device(traits::rank_dynamic, arg_N0, arg_N1, arg_N2, arg_N3,
                                      arg_N4, arg_N5, arg_N6, arg_N7);

#endif



    }

    // For backward compatibility
    explicit inline
    View( const ViewAllocateWithoutInitializing & arg_prop
          , const typename traits::array_layout & arg_layout
    )
    : View( Impl::ViewCtorProp< std::string , Kokkos::Impl::WithoutInitializing_t >( arg_prop.label , Kokkos::WithoutInitializing )
            , arg_layout
    )
    {}

    explicit inline
    View( const ViewAllocateWithoutInitializing & arg_prop
          , const size_t arg_N0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
    )
    : View( Impl::ViewCtorProp< std::string , Kokkos::Impl::WithoutInitializing_t >( arg_prop.label , Kokkos::WithoutInitializing )
            , typename traits::array_layout
            ( arg_N0 , arg_N1 , arg_N2 , arg_N3
              , arg_N4 , arg_N5 , arg_N6 , arg_N7 )
    )
    {
#ifdef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
      Impl::runtime_check_rank_host(traits::rank_dynamic, arg_N0, arg_N1, arg_N2, arg_N3,
                                    arg_N4, arg_N5, arg_N6, arg_N7, label());
#else
      Impl::runtime_check_rank_device(traits::rank_dynamic, arg_N0, arg_N1, arg_N2, arg_N3,
                                      arg_N4, arg_N5, arg_N6, arg_N7);

#endif

    }

    //----------------------------------------
    // Memory span required to wrap these dimensions.
    static constexpr size_t required_allocation_size(
        const size_t arg_N0 = 0
        , const size_t arg_N1 = 0
        , const size_t arg_N2 = 0
        , const size_t arg_N3 = 0
        , const size_t arg_N4 = 0
        , const size_t arg_N5 = 0
        , const size_t arg_N6 = 0
        , const size_t arg_N7 = 0
    )
    {
      return map_type::memory_span(
          typename traits::array_layout
          ( arg_N0 , arg_N1 , arg_N2 , arg_N3
            , arg_N4 , arg_N5 , arg_N6 , arg_N7 ) );
    }

    explicit KOKKOS_INLINE_FUNCTION
    View( pointer_type arg_ptr
          , const size_t arg_N0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
    )
    : View( Impl::ViewCtorProp<pointer_type>(arg_ptr)
            , typename traits::array_layout
            ( arg_N0 , arg_N1 , arg_N2 , arg_N3
              , arg_N4 , arg_N5 , arg_N6 , arg_N7 )
    )
    {
#ifdef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
      Impl::runtime_check_rank_host(traits::rank_dynamic, arg_N0, arg_N1, arg_N2, arg_N3,
                                    arg_N4, arg_N5, arg_N6, arg_N7, label());
#else
      Impl::runtime_check_rank_device(traits::rank_dynamic, arg_N0, arg_N1, arg_N2, arg_N3,
                                      arg_N4, arg_N5, arg_N6, arg_N7);

#endif
    }

    explicit KOKKOS_INLINE_FUNCTION
    View( pointer_type arg_ptr
          , const typename traits::array_layout & arg_layout
    )
    : View( Impl::ViewCtorProp<pointer_type>(arg_ptr) , arg_layout )
    {

    }

    //----------------------------------------
    // Shared scratch memory constructor

    static inline
    size_t
    shmem_size( const size_t arg_N0 = KOKKOS_INVALID_INDEX,
                const size_t arg_N1 = KOKKOS_INVALID_INDEX,
                const size_t arg_N2 = KOKKOS_INVALID_INDEX,
                const size_t arg_N3 = KOKKOS_INVALID_INDEX,
                const size_t arg_N4 = KOKKOS_INVALID_INDEX,
                const size_t arg_N5 = KOKKOS_INVALID_INDEX,
                const size_t arg_N6 = KOKKOS_INVALID_INDEX,
                const size_t arg_N7 = KOKKOS_INVALID_INDEX )
    {
      if ( is_layout_stride ) {
          Kokkos::abort( "Kokkos::View::shmem_size(extents...) doesn't work with LayoutStride. Pass a LayoutStride object instead" );
      }
      const size_t num_passed_args = Impl::count_valid_integers(arg_N0, arg_N1, arg_N2, arg_N3,
                                                                arg_N4, arg_N5, arg_N6, arg_N7);

      if ( std::is_same<typename traits::specialize,void>::value && num_passed_args != traits::rank_dynamic ) {
          Kokkos::abort( "Kokkos::View::shmem_size() rank_dynamic != number of arguments.\n" );
      }

      return View::shmem_size(
          typename traits::array_layout
          ( arg_N0 , arg_N1 , arg_N2 , arg_N3
            , arg_N4 , arg_N5 , arg_N6 , arg_N7 ) );
    }

    static inline
    size_t shmem_size( typename traits::array_layout const& arg_layout )
    {
      return map_type::memory_span( arg_layout );
    }

    explicit KOKKOS_INLINE_FUNCTION
    View( const typename traits::execution_space::scratch_memory_space & arg_space
          , const typename traits::array_layout & arg_layout )
    : View( Impl::ViewCtorProp<pointer_type>(
        reinterpret_cast<pointer_type>(
            arg_space.get_shmem( map_type::memory_span( arg_layout ) ) ) )
            , arg_layout )
    {}

    explicit KOKKOS_INLINE_FUNCTION
    View( const typename traits::execution_space::scratch_memory_space & arg_space
          , const size_t arg_N0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG
          , const size_t arg_N7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG )
    : View( Impl::ViewCtorProp<pointer_type>(
        reinterpret_cast<pointer_type>(
            arg_space.get_shmem(
                map_type::memory_span(
                    typename traits::array_layout
                    ( arg_N0 , arg_N1 , arg_N2 , arg_N3
                      , arg_N4 , arg_N5 , arg_N6 , arg_N7 ) ) ) ) )
            , typename traits::array_layout
            ( arg_N0 , arg_N1 , arg_N2 , arg_N3
              , arg_N4 , arg_N5 , arg_N6 , arg_N7 )
    )
    {

#ifdef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
      Impl::runtime_check_rank_host(traits::rank_dynamic, arg_N0, arg_N1, arg_N2, arg_N3,
                                    arg_N4, arg_N5, arg_N6, arg_N7, label());
#else
      Impl::runtime_check_rank_device(traits::rank_dynamic, arg_N0, arg_N1, arg_N2, arg_N3,
                                      arg_N4, arg_N5, arg_N6, arg_N7);

#endif
    }
#endif

    template< typename Label>
    explicit inline
    OffsetView( const Label & arg_label
                ,typename std::enable_if<Kokkos::Impl::is_view_label<Label>::value , const index_list_type >::type
                                      range0 = KOKKOS_INVALID_INDEX_RANGE
                ,const index_list_type range1 = KOKKOS_INVALID_INDEX_RANGE
                ,const index_list_type range2 = KOKKOS_INVALID_INDEX_RANGE
                ,const index_list_type range3 = KOKKOS_INVALID_INDEX_RANGE
                ,const index_list_type range4 = KOKKOS_INVALID_INDEX_RANGE
                ,const index_list_type range5 = KOKKOS_INVALID_INDEX_RANGE
                ,const index_list_type range6 = KOKKOS_INVALID_INDEX_RANGE
                ,const index_list_type range7 = KOKKOS_INVALID_INDEX_RANGE

    ) : OffsetView( Impl::ViewCtorProp< std::string >( arg_label ),
                    typename traits::array_layout
                    ( range0.begin()[1] - range0.begin()[0] + 1, range1.begin()[1] - range1.begin()[0] + 1 ,
                      range2.begin()[1] - range2.begin()[0] + 1, range3.begin()[1] - range3.begin()[0] + 1,
                      range4.begin()[1] - range4.begin()[0] + 1, range5.begin()[1] - range5.begin()[0] + 1 ,
                      range6.begin()[1] - range6.begin()[0] + 1, range7.begin()[1] - range7.begin()[0] + 1 ),
                      {range0.begin()[0], range1.begin()[0], range2.begin()[0], range3.begin()[0], range4.begin()[0],
                          range5.begin()[0], range6.begin()[0], range7.begin()[0] })
    {

    }



    template<class ... P >
    explicit KOKKOS_INLINE_FUNCTION
    OffsetView( const Impl::ViewCtorProp< P ... > & arg_prop
                ,typename std::enable_if< Impl::ViewCtorProp< P... >::has_pointer , typename traits::array_layout >::type const & arg_layout
                ,const index_list_type minIndices
    )
    : m_track() // No memory tracking
    , m_map( arg_prop , arg_layout )
    {


      for (size_t i = 0; i < minIndices.size(); ++i) {
          m_begins[i] = minIndices.begin()[i];
      }
      static_assert(
          std::is_same< pointer_type
          , typename Impl::ViewCtorProp< P... >::pointer_type
          >::value ,
          "When constructing OffsetView to wrap user memory, you must supply matching pointer type" );
    }

    template<class ... P >
    explicit inline
    OffsetView( const Impl::ViewCtorProp< P ... > & arg_prop
                , typename std::enable_if< ! Impl::ViewCtorProp< P... >::has_pointer , typename traits::array_layout>::type const & arg_layout
                ,const index_list_type minIndices
    )
    : m_track()
    , m_map()

    {

      for(size_t i = 0; i < Rank; ++i)
        m_begins[i] = minIndices.begin()[i];

      // Append layout and spaces if not input
      typedef Impl::ViewCtorProp< P ... > alloc_prop_input ;

      // use 'std::integral_constant<unsigned,I>' for non-types
      // to avoid duplicate class error.
      typedef Impl::ViewCtorProp
          < P ..., typename std::conditional < alloc_prop_input::has_label
          , std::integral_constant<unsigned,0>, typename std::string >::type
          , typename std::conditional
      < alloc_prop_input::has_memory_space
      , std::integral_constant<unsigned,1>
      , typename traits::device_type::memory_space
      >::type
      , typename std::conditional
      < alloc_prop_input::has_execution_space
      , std::integral_constant<unsigned,2>
      , typename traits::device_type::execution_space
      >::type
      > alloc_prop ;

      static_assert( traits::is_managed
                     , "OffsetView allocation constructor requires managed memory" );

      if ( alloc_prop::initialize &&
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE
           ! alloc_prop::execution_space::is_initialized()
#else
           ! alloc_prop::execution_space::impl_is_initialized()
#endif
      ) {
          // If initializing view data then
          // the execution space must be initialized.
          Kokkos::Impl::throw_runtime_exception("Constructing OffsetView and initializing data with uninitialized execution space");
      }

      // Copy the input allocation properties with possibly defaulted properties
      alloc_prop prop( arg_prop );

      //------------------------------------------------------------
#if defined( KOKKOS_ENABLE_CUDA )
      // If allocating in CudaUVMSpace must fence before and after
      // the allocation to protect against possible concurrent access
      // on the CPU and the GPU.
      // Fence using the trait's executon space (which will be Kokkos::Cuda)
      // to avoid incomplete type errors from usng Kokkos::Cuda directly.
      if ( std::is_same< Kokkos::CudaUVMSpace , typename traits::device_type::memory_space >::value ) {
          traits::device_type::memory_space::execution_space::fence();
      }
#endif
      //------------------------------------------------------------

      Kokkos::Impl::SharedAllocationRecord<> *
      record = m_map.allocate_shared( prop , arg_layout );

      //------------------------------------------------------------
#if defined( KOKKOS_ENABLE_CUDA )
      if ( std::is_same< Kokkos::CudaUVMSpace , typename traits::device_type::memory_space >::value ) {
          traits::device_type::memory_space::execution_space::fence();
      }
#endif
      //------------------------------------------------------------

      // Setup and initialization complete, start tracking
      m_track.assign_allocated_record_to_uninitialized( record );

#ifdef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
          Impl::runtime_check_rank_host(traits::rank_dynamic, Rank, minIndices, label());
#else
          Impl::runtime_check_rank_device(traits::rank_dynamic, Rank, minIndices);

#endif

    }


  };



  /** \brief Temporary free function rank()
   *         until rank() is implemented
   *         in the View
   */
  template < typename D , class ... P >
  KOKKOS_INLINE_FUNCTION
  constexpr unsigned rank( const OffsetView<D , P...> & V ) { return V.Rank; } //Temporary until added to view

  //----------------------------------------------------------------------------
  //----------------------------------------------------------------------------

#if 0
  template< class D, class ... P , class ... Args >
  KOKKOS_INLINE_FUNCTION
  typename Kokkos::Impl::ViewMapping
  < void /* deduce subview type from source view traits */
  , ViewTraits< D , P... >
  , Args ...
  >::type
  subview( const View< D, P... > & src , Args ... args )
  {
    static_assert( View< D , P... >::Rank == sizeof...(Args) ,
                   "subview requires one argument for each source View rank" );

    return typename
        Kokkos::Impl::ViewMapping
        < void /* deduce subview type from source view traits */
        , ViewTraits< D , P ... >
    , Args ... >::type( src , args ... );
  }

  template< class MemoryTraits , class D, class ... P , class ... Args >
  KOKKOS_INLINE_FUNCTION
  typename Kokkos::Impl::ViewMapping
  < void /* deduce subview type from source view traits */
  , ViewTraits< D , P... >
  , Args ...
  >::template apply< MemoryTraits >::type
  subview( const View< D, P... > & src , Args ... args )
  {
    static_assert( View< D , P... >::Rank == sizeof...(Args) ,
                   "subview requires one argument for each source View rank" );

    return typename
        Kokkos::Impl::ViewMapping
        < void /* deduce subview type from source view traits */
        , ViewTraits< D , P ... >
    , Args ... >
    ::template apply< MemoryTraits >
    ::type( src , args ... );
  }
#endif
} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

  template< class LT , class ... LP , class RT , class ... RP >
  KOKKOS_INLINE_FUNCTION
  bool operator == ( const OffsetView<LT,LP...> & lhs ,
      const OffsetView<RT,RP...> & rhs )
      {
    // Same data, layout, dimensions
    typedef ViewTraits<LT,LP...>  lhs_traits ;
    typedef ViewTraits<RT,RP...>  rhs_traits ;

    return
        std::is_same< typename lhs_traits::const_value_type ,
        typename rhs_traits::const_value_type >::value &&
        std::is_same< typename lhs_traits::array_layout ,
        typename rhs_traits::array_layout >::value &&
        std::is_same< typename lhs_traits::memory_space ,
        typename rhs_traits::memory_space >::value &&
        unsigned(lhs_traits::rank) == unsigned(rhs_traits::rank) &&
        lhs.data()        == rhs.data() &&
        lhs.span()        == rhs.span() &&
        lhs.extent(0) == rhs.extent(0) &&
        lhs.extent(1) == rhs.extent(1) &&
        lhs.extent(2) == rhs.extent(2) &&
        lhs.extent(3) == rhs.extent(3) &&
        lhs.extent(4) == rhs.extent(4) &&
        lhs.extent(5) == rhs.extent(5) &&
        lhs.extent(6) == rhs.extent(6) &&
        lhs.extent(7) == rhs.extent(7) &&
        lhs.begin(0) == rhs.begin(0) &&
        lhs.begin(1) == rhs.begin(1) &&
        lhs.begin(2) == rhs.begin(2) &&
        lhs.begin(3) == rhs.begin(3) &&
        lhs.begin(4) == rhs.begin(4) &&
        lhs.begin(5) == rhs.begin(5) &&
        lhs.begin(6) == rhs.begin(6) &&
        lhs.begin(7) == rhs.begin(7)
                                        ;
      }

  template< class LT , class ... LP , class RT , class ... RP >
  KOKKOS_INLINE_FUNCTION
  bool operator != ( const OffsetView<LT,LP...> & lhs ,
      const OffsetView<RT,RP...> & rhs )
      {
    return ! ( operator==(lhs,rhs) );
      }

  template< class LT , class ... LP , class RT , class ... RP >
    KOKKOS_INLINE_FUNCTION
    bool operator == ( const View<LT,LP...> & lhs ,
        const OffsetView<RT,RP...> & rhs )
        {
      // Same data, layout, dimensions
      typedef ViewTraits<LT,LP...>  lhs_traits ;
      typedef ViewTraits<RT,RP...>  rhs_traits ;

      return
          std::is_same< typename lhs_traits::const_value_type ,
          typename rhs_traits::const_value_type >::value &&
          std::is_same< typename lhs_traits::array_layout ,
          typename rhs_traits::array_layout >::value &&
          std::is_same< typename lhs_traits::memory_space ,
          typename rhs_traits::memory_space >::value &&
          unsigned(lhs_traits::rank) == unsigned(rhs_traits::rank) &&
          lhs.data()        == rhs.data() &&
          lhs.span()        == rhs.span() &&
          lhs.extent(0) == rhs.extent(0) &&
          lhs.extent(1) == rhs.extent(1) &&
          lhs.extent(2) == rhs.extent(2) &&
          lhs.extent(3) == rhs.extent(3) &&
          lhs.extent(4) == rhs.extent(4) &&
          lhs.extent(5) == rhs.extent(5) &&
          lhs.extent(6) == rhs.extent(6) &&
          lhs.extent(7) == rhs.extent(7)
                                          ;
        }

  template< class LT , class ... LP , class RT , class ... RP >
  KOKKOS_INLINE_FUNCTION
    bool operator == ( const OffsetView<LT,LP...> & lhs ,
        const View<RT,RP...> & rhs )
        { return rhs == lhs;}


} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#if 0 //swb dont think we need this
namespace Kokkos {
  namespace Impl {

    inline
    void shared_allocation_tracking_disable()
    { Kokkos::Impl::SharedAllocationRecord<void,void>::tracking_disable(); }

    inline
    void shared_allocation_tracking_enable()
    { Kokkos::Impl::SharedAllocationRecord<void,void>::tracking_enable(); }

  } /* namespace Impl */
} /* namespace Kokkos */
#endif
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
  namespace Impl {

    // Deduce Mirror Types
    template<class Space, class T, class ... P>
    struct MirrorOffsetViewType {
      // The incoming view_type
      typedef typename Kokkos::OffsetView<T,P...> src_view_type;
      // The memory space for the mirror view
      typedef typename Space::memory_space memory_space;
      // Check whether it is the same memory space
      enum { is_same_memspace = std::is_same<memory_space,typename src_view_type::memory_space>::value };
      // The array_layout
      typedef typename src_view_type::array_layout array_layout;
      // The data type (we probably want it non-const since otherwise we can't even deep_copy to it.
      typedef typename src_view_type::non_const_data_type data_type;
      // The destination view type if it is not the same memory space
      typedef Kokkos::OffsetView<data_type,array_layout,Space> dest_view_type;
      // If it is the same memory_space return the existsing view_type
      // This will also keep the unmanaged trait if necessary
      typedef typename std::conditional<is_same_memspace,src_view_type,dest_view_type>::type view_type;
    };

    template<class Space, class T, class ... P>
    struct MirrorOffsetType {
      // The incoming view_type
      typedef typename Kokkos::OffsetView<T,P...> src_view_type;
      // The memory space for the mirror view
      typedef typename Space::memory_space memory_space;
      // Check whether it is the same memory space
      enum { is_same_memspace = std::is_same<memory_space,typename src_view_type::memory_space>::value };
      // The array_layout
      typedef typename src_view_type::array_layout array_layout;
      // The data type (we probably want it non-const since otherwise we can't even deep_copy to it.
      typedef typename src_view_type::non_const_data_type data_type;
      // The destination view type if it is not the same memory space
      typedef Kokkos::OffsetView<data_type,array_layout,Space> view_type;
    };

  }

  template< class T , class ... P >
  inline
  typename Kokkos::OffsetView<T,P...>::HostMirror
  create_mirror( const Kokkos::OffsetView<T,P...> & src
                 , typename std::enable_if<
                 ! std::is_same< typename Kokkos::ViewTraits<T,P...>::array_layout
                 , Kokkos::LayoutStride >::value
                 >::type * = 0
  )
  {
    typedef OffsetView<T,P...>             src_type ;
    typedef typename src_type::HostMirror  dst_type ;

    return dst_type( std::string( src.label() ).append("_mirror"),
                     typename Kokkos::ViewTraits<T,P...>::array_layout
                     ( src.extent(0), src.extent(1), src.extent(2), src.extent(3), src.extent(4),
                       src.extent(5), src.extent(6), src.extent(7) ),
                     { src.begin(0), src.begin(1), src.begin(2), src.begin(3), src.begin(4),
                         src.begin(5), src.begin(6), src.begin(7) });
  }

  template< class T , class ... P >
  inline
  typename Kokkos::OffsetView<T,P...>::HostMirror
  create_mirror( const Kokkos::OffsetView<T,P...> & src
                 , typename std::enable_if<
                 std::is_same< typename Kokkos::ViewTraits<T,P...>::array_layout
                 , Kokkos::LayoutStride >::value
                 >::type * = 0
  )
  {
    typedef OffsetView<T,P...>             src_type ;
    typedef typename src_type::HostMirror  dst_type ;

    Kokkos::LayoutStride layout ;

    layout.dimension[0] = src.extent(0);
    layout.dimension[1] = src.extent(1);
    layout.dimension[2] = src.extent(2);
    layout.dimension[3] = src.extent(3);
    layout.dimension[4] = src.extent(4);
    layout.dimension[5] = src.extent(5);
    layout.dimension[6] = src.extent(6);
    layout.dimension[7] = src.extent(7);

    layout.stride[0] = src.stride_0();
    layout.stride[1] = src.stride_1();
    layout.stride[2] = src.stride_2();
    layout.stride[3] = src.stride_3();
    layout.stride[4] = src.stride_4();
    layout.stride[5] = src.stride_5();
    layout.stride[6] = src.stride_6();
    layout.stride[7] = src.stride_7();

    return dst_type( std::string( src.label() ).append("_mirror") , layout,
                     { src.begin(0), src.begin(1), src.begin(2), src.begin(3), src.begin(4),
                         src.begin(5), src.begin(6), src.begin(7) } );
  }


  // Create a mirror in a new space (specialization for different space)
  template<class Space, class T, class ... P>
  typename Impl::MirrorOffsetType<Space,T,P ...>::view_type create_mirror(const Space& , const Kokkos::OffsetView<T,P...> & src) {
    return typename Impl::MirrorOffsetType<Space,T,P ...>::view_type(src.label(),src.layout(),
                                                               { src.begin(0), src.begin(1), src.begin(2), src.begin(3), src.begin(4),
                                                                   src.begin(5), src.begin(6), src.begin(7) } );
  }

  template< class T , class ... P >
  inline
  typename Kokkos::OffsetView<T,P...>::HostMirror
  create_mirror_view( const Kokkos::OffsetView<T,P...> & src
                      , typename std::enable_if<(
                          std::is_same< typename Kokkos::OffsetView<T,P...>::memory_space
                          , typename Kokkos::OffsetView<T,P...>::HostMirror::memory_space
                          >::value
                          &&
                          std::is_same< typename Kokkos::OffsetView<T,P...>::data_type
                          , typename Kokkos::OffsetView<T,P...>::HostMirror::data_type
                          >::value
                      )>::type * = 0
  )
  {
    return src ;
  }

  template< class T , class ... P >
  inline
  typename Kokkos::OffsetView<T,P...>::HostMirror
  create_mirror_view( const Kokkos::OffsetView<T,P...> & src
                      , typename std::enable_if< ! (
                          std::is_same< typename Kokkos::OffsetView<T,P...>::memory_space
                          , typename Kokkos::OffsetView<T,P...>::HostMirror::memory_space
                          >::value
                          &&
                          std::is_same< typename Kokkos::OffsetView<T,P...>::data_type
                          , typename Kokkos::OffsetView<T,P...>::HostMirror::data_type
                          >::value
                      )>::type * = 0
  )
  {
    return Kokkos::create_mirror( src );
  }

  // Create a mirror view in a new space (specialization for same space)
  template<class Space, class T, class ... P>
  typename Impl::MirrorOffsetViewType<Space,T,P ...>::view_type
  create_mirror_view(const Space& , const Kokkos::OffsetView<T,P...> & src
                     , typename std::enable_if<Impl::MirrorOffsetViewType<Space,T,P ...>::is_same_memspace>::type* = 0 ) {
    return src;
  }

  // Create a mirror view in a new space (specialization for different space)
  template<class Space, class T, class ... P>
  typename Impl::MirrorOffsetViewType<Space,T,P ...>::view_type
  create_mirror_view(const Space& , const Kokkos::OffsetView<T,P...> & src
                     , typename std::enable_if<!Impl::MirrorOffsetViewType<Space,T,P ...>::is_same_memspace>::type* = 0 ) {
    return typename Impl::MirrorOffsetViewType<Space,T,P ...>::view_type(src.label(),src.layout(),
                                                                   { src.begin(0), src.begin(1), src.begin(2), src.begin(3), src.begin(4),
                                                                       src.begin(5), src.begin(6), src.begin(7) } );
  }
//
//  // Create a mirror view and deep_copy in a new space (specialization for same space)
//  template<class Space, class T, class ... P>
//  typename Impl::MirrorViewType<Space,T,P ...>::view_type
//  create_mirror_view_and_copy(const Space& , const Kokkos::OffsetView<T,P...> & src
//                              , std::string const& name = ""
//                                  , typename std::enable_if<Impl::MirrorViewType<Space,T,P ...>::is_same_memspace>::type* = 0 ) {
//    (void)name;
//    return src;
//  }
//
//  // Create a mirror view and deep_copy in a new space (specialization for different space)
//  template<class Space, class T, class ... P>
//  typename Impl::MirrorViewType<Space,T,P ...>::view_type
//  create_mirror_view_and_copy(const Space& , const Kokkos::OffsetView<T,P...> & src
//                              , std::string const& name = ""
//                                  , typename std::enable_if<!Impl::MirrorViewType<Space,T,P ...>::is_same_memspace>::type* = 0 ) {
//    using Mirror = typename Impl::MirrorViewType<Space,T,P ...>::view_type;
//    std::string label = name.empty() ? src.label() : name;
//    auto mirror = Mirror(ViewAllocateWithoutInitializing(label), src.layout(),
//                         { src.begin(0), src.begin(1), src.begin(2), src.begin(3), src.begin(4),
//                             src.begin(5), src.begin(6), src.begin(7) });
//    deep_copy(mirror, src);
//    return mirror;
//  }

} /* namespace Kokkos */


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
#if 0
namespace Kokkos { namespace Impl {

  template < class Specialize, typename A, typename B >
  struct CommonViewValueType;

  template < typename A, typename B >
  struct CommonViewValueType< void, A, B >
  {
    using value_type = typename std::common_type< A , B >::type;
  };


  template < class Specialize, class ValueType >
  struct CommonViewAllocProp;

  template < class ValueType >
  struct CommonViewAllocProp< void, ValueType >
  {
    using value_type = ValueType;
    using scalar_array_type = ValueType;

    template < class ... Views >
    KOKKOS_INLINE_FUNCTION
    CommonViewAllocProp( const Views & ... ) {}
  };


  template < class ... Views >
  struct DeduceCommonViewAllocProp;

  // Base case must provide types for:
  // 1. specialize  2. value_type  3. is_view  4. prop_type
  template < class FirstView >
  struct DeduceCommonViewAllocProp< FirstView >
  {
    using specialize = typename FirstView::traits::specialize;

    using value_type = typename FirstView::traits::value_type;

    enum : bool { is_view = is_view< FirstView >::value };

    using prop_type = CommonViewAllocProp< specialize, value_type >;
  };


  template < class FirstView, class ... NextViews >
  struct DeduceCommonViewAllocProp< FirstView, NextViews... >
  {
    using NextTraits = DeduceCommonViewAllocProp< NextViews... >;

    using first_specialize = typename FirstView::traits::specialize;
    using first_value_type = typename FirstView::traits::value_type;

    enum : bool { first_is_view = is_view< FirstView >::value };

    using next_specialize = typename NextTraits::specialize;
    using next_value_type = typename NextTraits::value_type;

    enum : bool { next_is_view = NextTraits::is_view };

    // common types

    // determine specialize type
    // if first and next specialize differ, but are not the same specialize, error out
    static_assert( !(!std::is_same< first_specialize, next_specialize >::value &&
        !std::is_same< first_specialize, void>::value &&
        !std::is_same< void, next_specialize >::value)  ,
                   "Kokkos DeduceCommonViewAllocProp ERROR: Only one non-void specialize trait allowed" );

    // otherwise choose non-void specialize if either/both are non-void
    using specialize = typename std::conditional< std::is_same< first_specialize, next_specialize >::value
        , first_specialize
        , typename std::conditional< ( std::is_same< first_specialize, void >::value
            && !std::is_same< next_specialize, void >::value)
            , next_specialize
            , first_specialize
            >::type
            >::type;

    using value_type = typename CommonViewValueType< specialize, first_value_type, next_value_type >::value_type;

    enum : bool { is_view = (first_is_view && next_is_view) };

    using prop_type = CommonViewAllocProp< specialize, value_type >;
  };

} // end namespace Impl

template < class ... Views >
using DeducedCommonPropsType = typename Impl::DeduceCommonViewAllocProp<Views...>::prop_type ;

// User function
template < class ... Views >
KOKKOS_INLINE_FUNCTION
DeducedCommonPropsType<Views...>
common_view_alloc_prop( Views const & ... views )
{
  return DeducedCommonPropsType<Views...>( views... );
}

} // namespace Kokkos


namespace Kokkos {
  namespace Impl {

    using Kokkos::is_view ;

  } /* namespace Impl */
} /* namespace Kokkos */

#include <impl/Kokkos_Atomic_View.hpp>
#endif

#endif /* KOKKOS_OFFSETVIEW_HPP_ */
