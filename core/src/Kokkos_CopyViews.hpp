#ifndef KOKKOS_COPYVIEWS_HPP_
#define KOKKOS_COPYVIEWS_HPP_
#include <string>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

namespace Impl {

template<class Layout>
struct ViewFillLayoutSelector {
};

template<>
struct ViewFillLayoutSelector<Kokkos::LayoutLeft> {
  static const Kokkos::Experimental::Iterate iterate = Kokkos::Experimental::Iterate::Left;
};

template<>
struct ViewFillLayoutSelector<Kokkos::LayoutRight> {
  static const Kokkos::Experimental::Iterate iterate = Kokkos::Experimental::Iterate::Right;
};

template<class ViewType,class Layout, typename iType>
struct ViewFill<ViewType,Layout,0,iType> {

  typedef typename ViewType::non_const_value_type ST;

  ViewFill(const ViewType& a, const ST& val) {
    Kokkos::Impl::DeepCopy< typename ViewType::memory_space, Kokkos::HostSpace >( a.data() , &val, sizeof(ST) );
  }
};


template<class ViewType,class Layout, typename iType>
struct ViewFill<ViewType,Layout,1,iType> {
  ViewType a;
  typename ViewType::const_value_type val;

  ViewFill(const ViewType& a_, typename ViewType::const_value_type& val_):a(a_),val(val_) {
    Kokkos::parallel_for("Kokkos::ViewFill-1D",a.extent(0),*this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const iType& i) const {
    a(i) = val;
  };
};

template<class ViewType,class Layout, typename iType>
struct ViewFill<ViewType,Layout,2,iType> {
  ViewType a;
  typename ViewType::const_value_type val;

  typedef typename ViewType::execution_space execution_space;
  typedef Kokkos::Experimental::Rank<2,ViewFillLayoutSelector<Layout>::iterate,ViewFillLayoutSelector<Layout>::iterate> iterate_type;
  typedef Kokkos::Experimental::MDRangePolicy<execution_space,iterate_type,Kokkos::IndexType<iType>> policy_type;

  ViewFill(const ViewType& a_, typename ViewType::const_value_type& val_):a(a_),val(val_) {
    Kokkos::parallel_for("Kokkos::ViewFill-2D",
       policy_type({0,0},{a.extent(0),a.extent(1)}),*this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const iType& i0, const iType& i1) const {
    a(i0,i1) = val;
  };
};

template<class ViewType,class Layout, typename iType>
struct ViewFill<ViewType,Layout,3,iType> {
  ViewType a;
  typename ViewType::const_value_type val;

  typedef typename ViewType::execution_space execution_space;
  typedef Kokkos::Experimental::Rank<3,ViewFillLayoutSelector<Layout>::iterate,ViewFillLayoutSelector<Layout>::iterate> iterate_type;
  typedef Kokkos::Experimental::MDRangePolicy<execution_space,iterate_type,Kokkos::IndexType<iType>> policy_type;

  ViewFill(const ViewType& a_, typename ViewType::const_value_type& val_):a(a_),val(val_) {
    Kokkos::parallel_for("Kokkos::ViewFill-3D",
       policy_type({0,0,0},{a.extent(0),a.extent(1),a.extent(2)}),*this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const iType& i0, const iType& i1, const iType& i2) const {
    a(i0,i1,i2) = val;
  };
};

template<class ViewType,class Layout, typename iType>
struct ViewFill<ViewType,Layout,4,iType> {
  ViewType a;
  typename ViewType::const_value_type val;

  typedef typename ViewType::execution_space execution_space;
  typedef Kokkos::Experimental::Rank<4,ViewFillLayoutSelector<Layout>::iterate,ViewFillLayoutSelector<Layout>::iterate> iterate_type;
  typedef Kokkos::Experimental::MDRangePolicy<execution_space,iterate_type,Kokkos::IndexType<iType>> policy_type;

  ViewFill(const ViewType& a_, typename ViewType::const_value_type& val_):a(a_),val(val_) {
    Kokkos::parallel_for("Kokkos::ViewFill-4D",
       policy_type({0,0,0,0},{a.extent(0),a.extent(1),a.extent(2),a.extent(3)}),*this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const iType& i0, const iType& i1, const iType& i2, const iType& i3) const {
    a(i0,i1,i2,i3) = val;
  };
};

template<class ViewType,class Layout, typename iType>
struct ViewFill<ViewType,Layout,5,iType> {
  ViewType a;
  typename ViewType::const_value_type val;

  typedef typename ViewType::execution_space execution_space;
  typedef Kokkos::Experimental::Rank<5,ViewFillLayoutSelector<Layout>::iterate,ViewFillLayoutSelector<Layout>::iterate> iterate_type;
  typedef Kokkos::Experimental::MDRangePolicy<execution_space,iterate_type,Kokkos::IndexType<iType>> policy_type;

  ViewFill(const ViewType& a_, typename ViewType::const_value_type& val_):a(a_),val(val_) {
    Kokkos::parallel_for("Kokkos::ViewFill-5D",
       policy_type({0,0,0,0,0},{a.extent(0),a.extent(1),a.extent(2),a.extent(3),a.extent(4)}),*this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const iType& i0, const iType& i1, const iType& i2, const iType& i3, const iType& i4) const {
    a(i0,i1,i2,i3,i4) = val;
  };
};

template<class ViewType,class Layout, typename iType>
struct ViewFill<ViewType,Layout,6,iType> {
  ViewType a;
  typename ViewType::const_value_type val;

  typedef typename ViewType::execution_space execution_space;
  typedef Kokkos::Experimental::Rank<6,ViewFillLayoutSelector<Layout>::iterate,ViewFillLayoutSelector<Layout>::iterate> iterate_type;
  typedef Kokkos::Experimental::MDRangePolicy<execution_space,iterate_type,Kokkos::IndexType<iType>> policy_type;

  ViewFill(const ViewType& a_, typename ViewType::const_value_type& val_):a(a_),val(val_) {
    Kokkos::parallel_for("Kokkos::ViewFill-6D",
       policy_type({0,0,0,0,0,0},{a.extent(0),a.extent(1),a.extent(2),a.extent(3),a.extent(4),a.extent(5)}),*this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const iType& i0, const iType& i1, const iType& i2, const iType& i3, const iType& i4, const iType& i5) const {
    a(i0,i1,i2,i3,i4,i5) = val;
  };
};

template<class ViewType,class Layout, typename iType>
struct ViewFill<ViewType,Layout,7,iType> {
  ViewType a;
  typename ViewType::const_value_type val;

  typedef typename ViewType::execution_space execution_space;
  typedef Kokkos::Experimental::Rank<6,ViewFillLayoutSelector<Layout>::iterate,ViewFillLayoutSelector<Layout>::iterate> iterate_type;
  typedef Kokkos::Experimental::MDRangePolicy<execution_space,iterate_type,Kokkos::IndexType<iType>> policy_type;

  ViewFill(const ViewType& a_, typename ViewType::const_value_type& val_):a(a_),val(val_) {
    Kokkos::parallel_for("Kokkos::ViewFill-7D",
       policy_type({0,0,0,0,0,0},{a.extent(0),a.extent(1),a.extent(2),a.extent(3),
                                  a.extent(5),a.extent(6)}),*this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const iType& i0, const iType& i1, const iType& i3,
                   const iType& i4, const iType& i5, const iType& i6) const {
    for(iType i2=0; i2<a.extent(2);i2++)
      a(i0,i1,i2,i3,i4,i5,i6) = val;
  };
};

template<class ViewType,class Layout, typename iType>
struct ViewFill<ViewType,Layout,8,iType> {
  ViewType a;
  typename ViewType::const_value_type val;

  typedef typename ViewType::execution_space execution_space;
  typedef Kokkos::Experimental::Rank<6,ViewFillLayoutSelector<Layout>::iterate,ViewFillLayoutSelector<Layout>::iterate> iterate_type;
  typedef Kokkos::Experimental::MDRangePolicy<execution_space,iterate_type,Kokkos::IndexType<iType>> policy_type;

  ViewFill(const ViewType& a_, typename ViewType::const_value_type& val_):a(a_),val(val_) {
    Kokkos::parallel_for("Kokkos::ViewFill-8D",
       policy_type({0,0,0,0,0,0},{a.extent(0),a.extent(1),a.extent(3),
                                  a.extent(5),a.extent(6),a.extent(7)}),*this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const iType& i0, const iType& i1, const iType& i3,
                   const iType& i5, const iType& i6, const iType& i7) const {
    for(iType i2=0; i2<a.extent(2);i2++)
    for(iType i4=0; i4<a.extent(4);i4++)
      a(i0,i1,i2,i3,i4,i5,i6,i7) = val;
  };
};

}

/** \brief  Deep copy a value from Host memory into a view.  */
template< class DT , class ... DP >
inline
void deep_copy
  ( const View<DT,DP...> & dst
  , typename ViewTraits<DT,DP...>::const_value_type & value
  , typename std::enable_if<
    std::is_same< typename ViewTraits<DT,DP...>::specialize , void >::value
    >::type * = 0 )
{
  typedef View<DT,DP...> ViewType;

  static_assert(
    std::is_same< typename ViewType::non_const_value_type ,
                  typename ViewType::value_type >::value
    , "deep_copy requires non-const type" );

  // If contigous we can simply do a 1D flat loop
  if(dst.span_is_contiguous()) {
    typedef Kokkos::View<typename ViewType::value_type*,Kokkos::LayoutRight,
        typename ViewType::device_type,Kokkos::MemoryTraits<Kokkos::Unmanaged> >
     ViewTypeFlat;

    ViewTypeFlat dst_flat(dst.data(),dst.size());
    Kokkos::Impl::ViewFill< ViewTypeFlat , Kokkos::LayoutLeft, ViewTypeFlat::Rank, int >( dst_flat , value );
    return;
  }

  // Figure out iteration order to do the ViewFill
  int64_t strides[ViewType::Rank+1];
  dst.stride(strides);
  Kokkos::Iterate iterate;
  if        ( std::is_same<typename ViewType::array_layout,Kokkos::LayoutRight>::value ) {
    iterate = Kokkos::Iterate::Right;
  } else if ( std::is_same<typename ViewType::array_layout,Kokkos::LayoutRight>::value ) {
    iterate = Kokkos::Iterate::Left;
  } else if ( std::is_same<typename ViewType::array_layout,Kokkos::LayoutStride>::value ) {
    if( strides[0] > strides[ViewType::Rank-1] )
      iterate = Kokkos::Iterate::Right;
    else
      iterate = Kokkos::Iterate::Left;
  } else {
    if( std::is_same<typename ViewType::execution_space::array_layout, Kokkos::LayoutRight>::value )
      iterate = Kokkos::Iterate::Right;
    else
      iterate = Kokkos::Iterate::Left;
  }

  // Lets call the right ViewFill functor based on integer space needed and iteration type
  if(dst.span() > std::numeric_limits<int>::max()) {
    if(iterate == Kokkos::Iterate::Right)
      Kokkos::Impl::ViewFill< ViewType, Kokkos::LayoutRight, ViewType::Rank, int64_t >( dst , value );
    else
      Kokkos::Impl::ViewFill< ViewType, Kokkos::LayoutLeft, ViewType::Rank, int64_t >( dst , value );
  } else {
    if(iterate == Kokkos::Iterate::Right)
      Kokkos::Impl::ViewFill< ViewType, Kokkos::LayoutRight, ViewType::Rank, int >( dst , value );
    else
      Kokkos::Impl::ViewFill< ViewType, Kokkos::LayoutLeft, ViewType::Rank, int >( dst , value );
  }
}

/** \brief  Deep copy into a value in Host memory from a view.  */
template< class ST , class ... SP >
inline
void deep_copy
  ( typename ViewTraits<ST,SP...>::non_const_value_type & dst
  , const View<ST,SP...> & src
  , typename std::enable_if<
    std::is_same< typename ViewTraits<ST,SP...>::specialize , void >::value
    >::type * = 0 )
{
  static_assert( ViewTraits<ST,SP...>::rank == 0
               , "ERROR: Non-rank-zero view in deep_copy( value , View )" );

  typedef ViewTraits<ST,SP...>               src_traits ;
  typedef typename src_traits::memory_space  src_memory_space ;
  Kokkos::Impl::DeepCopy< HostSpace , src_memory_space >( & dst , src.data() , sizeof(ST) );
}

//----------------------------------------------------------------------------
/** \brief  A deep copy between views of compatible type, and rank zero.  */
template< class DT , class ... DP , class ST , class ... SP >
inline
void deep_copy
  ( const View<DT,DP...> & dst
  , const View<ST,SP...> & src
  , typename std::enable_if<(
    std::is_same< typename ViewTraits<DT,DP...>::specialize , void >::value &&
    std::is_same< typename ViewTraits<ST,SP...>::specialize , void >::value &&
    ( unsigned(ViewTraits<DT,DP...>::rank) == unsigned(0) &&
      unsigned(ViewTraits<ST,SP...>::rank) == unsigned(0) )
  )>::type * = 0 )
{
  static_assert(
    std::is_same< typename ViewTraits<DT,DP...>::value_type ,
                  typename ViewTraits<ST,SP...>::non_const_value_type >::value
    , "deep_copy requires matching non-const destination type" );

  typedef View<DT,DP...>  dst_type ;
  typedef View<ST,SP...>  src_type ;

  typedef typename dst_type::value_type    value_type ;
  typedef typename dst_type::memory_space  dst_memory_space ;
  typedef typename src_type::memory_space  src_memory_space ;

  if ( dst.data() != src.data() ) {
    Kokkos::Impl::DeepCopy< dst_memory_space , src_memory_space >( dst.data() , src.data() , sizeof(value_type) );
  }
}

//----------------------------------------------------------------------------
/** \brief  A deep copy between views of the default specialization, compatible type,
 *          same non-zero rank, same contiguous layout.
 */
template< class DT , class ... DP , class ST , class ... SP >
inline
void deep_copy
  ( const View<DT,DP...> & dst
  , const View<ST,SP...> & src
  , typename std::enable_if<(
    std::is_same< typename ViewTraits<DT,DP...>::specialize , void >::value &&
    std::is_same< typename ViewTraits<ST,SP...>::specialize , void >::value &&
    ( unsigned(ViewTraits<DT,DP...>::rank) != 0 ||
      unsigned(ViewTraits<ST,SP...>::rank) != 0 )
  )>::type * = 0 )
{
  static_assert(
    std::is_same< typename ViewTraits<DT,DP...>::value_type ,
                  typename ViewTraits<DT,DP...>::non_const_value_type >::value
    , "deep_copy requires non-const destination type" );

  static_assert(
    ( unsigned(ViewTraits<DT,DP...>::rank) ==
      unsigned(ViewTraits<ST,SP...>::rank) )
    , "deep_copy requires Views of equal rank" );

  typedef View<DT,DP...>  dst_type ;
  typedef View<ST,SP...>  src_type ;

  typedef typename dst_type::execution_space  dst_execution_space ;
  typedef typename src_type::execution_space  src_execution_space ;
  typedef typename dst_type::memory_space     dst_memory_space ;
  typedef typename src_type::memory_space     src_memory_space ;
  typedef typename dst_type::value_type       dst_value_type ;
  typedef typename src_type::value_type       src_value_type ;

  enum { DstExecCanAccessSrc =
   Kokkos::Impl::SpaceAccessibility< dst_execution_space , src_memory_space >::accessible };

  enum { SrcExecCanAccessDst =
   Kokkos::Impl::SpaceAccessibility< src_execution_space , dst_memory_space >::accessible };

  // Checking for Overlapping Views.
  dst_value_type* dst_start = dst.data();
  dst_value_type* dst_end   = dst.data() + dst.span();
  src_value_type* src_start = src.data();
  src_value_type* src_end   = src.data() + src.span();
  if( ((std::ptrdiff_t)dst_start == (std::ptrdiff_t)src_start) &&
      ((std::ptrdiff_t)dst_end   == (std::ptrdiff_t)src_end)   &&
       (dst.span_is_contiguous() && src.span_is_contiguous()) )
    return;

  if( ( ( (std::ptrdiff_t)dst_start < (std::ptrdiff_t)src_end ) && ( (std::ptrdiff_t)dst_end > (std::ptrdiff_t)src_start ) ) &&
      ( ( dst.span_is_contiguous() && src.span_is_contiguous() ))) {
    std::string message("Error: Kokkos::deep_copy of overlapping views: ");
    message += dst.label(); message += "(";
    message += std::to_string((std::ptrdiff_t)dst_start); message += ",";
    message += std::to_string((std::ptrdiff_t)dst_end); message += ") ";
    message += src.label(); message += "(";
    message += std::to_string((std::ptrdiff_t)src_start); message += ",";
    message += std::to_string((std::ptrdiff_t)src_end); message += ") ";
    Kokkos::Impl::throw_runtime_exception(message);
  }

  // Check for same extents
  if ( (src.extent(0) != dst.extent(0)) ||
       (src.extent(1) != dst.extent(1)) ||
       (src.extent(2) != dst.extent(2)) ||
       (src.extent(3) != dst.extent(3)) ||
       (src.extent(4) != dst.extent(4)) ||
       (src.extent(5) != dst.extent(5)) ||
       (src.extent(6) != dst.extent(6)) ||
       (src.extent(7) != dst.extent(7))
     ) {
    #ifndef KOKKOS_ENABLE_DEPRECATED_CODE_REMOVAL
      if ( DstExecCanAccessSrc ) {
        // Copying data between views in accessible memory spaces and either non-contiguous or incompatible shape.
        Kokkos::Impl::ViewRemap< dst_type , src_type >( dst , src );
      }
      else if ( SrcExecCanAccessDst ) {
        // Copying data between views in accessible memory spaces and either non-contiguous or incompatible shape.
        Kokkos::Impl::ViewRemap< dst_type , src_type , src_execution_space >( dst , src );
      }
      else {
        Kokkos::Impl::throw_runtime_exception("deep_copy given views that would require a temporary allocation");
      }
      return;
    #else
    std::string message("Deprecation Error: Kokkos::deep_copy extents of views don't match: ");
    message += dst.label(); message += "(";
    for(int r = 0; r<dst_type::Rank-1; r++)
      { message+= std::to_string(dst.extent(r)); message += ","; }
    message+= std::to_string(dst.extent(dst_type::Rank-1)); message += ") ";
    message += src.label(); message += "(";
    for(int r = 0; r<src_type::Rank-1; r++)
      { message+= src::to_string(src.extent(r)); message += ","; }
    message+= std::to_string(src.extent(src_type::Rank-1)); message += ") ";

    Kokkos::throw_runtime_exception(message);
    #endif
  }

  // If same type, equal layout, equal dimensions, equal span, and contiguous memory then can byte-wise copy

  if ( std::is_same< typename ViewTraits<DT,DP...>::value_type ,
                     typename ViewTraits<ST,SP...>::non_const_value_type >::value &&
       (
         std::is_same< typename ViewTraits<DT,DP...>::array_layout ,
                       typename ViewTraits<ST,SP...>::array_layout >::value
         ||
         ( ViewTraits<DT,DP...>::rank == 1 &&
           ViewTraits<ST,SP...>::rank == 1 )
       ) &&
       dst.span_is_contiguous() &&
       src.span_is_contiguous() &&
       dst.stride_0() == src.stride_0() &&
       dst.stride_1() == src.stride_1() &&
       dst.stride_2() == src.stride_2() &&
       dst.stride_3() == src.stride_3() &&
       dst.stride_4() == src.stride_4() &&
       dst.stride_5() == src.stride_5() &&
       dst.stride_6() == src.stride_6() &&
       dst.stride_7() == src.stride_7()
    ) {

    const size_t nbytes = sizeof(typename dst_type::value_type) * dst.span();

    Kokkos::Impl::DeepCopy< dst_memory_space , src_memory_space >
      ( dst.data() , src.data() , nbytes );
  }
  else if ( DstExecCanAccessSrc ) {
    // Copying data between views in accessible memory spaces and either non-contiguous or incompatible shape.
    Kokkos::fence();
    Kokkos::Impl::ViewRemap< dst_type , src_type >( dst , src );
    Kokkos::fence();
  }
  else if ( SrcExecCanAccessDst ) {
    // Copying data between views in accessible memory spaces and either non-contiguous or incompatible shape.
    Kokkos::fence();
    Kokkos::Impl::ViewRemap< dst_type , src_type , src_execution_space >( dst , src );
    Kokkos::fence();
  }
  else {
    Kokkos::Impl::throw_runtime_exception("deep_copy given views that would require a temporary allocation");
  }
}

} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

/** \brief  Deep copy a value from Host memory into a view.  */
template< class ExecSpace ,class DT , class ... DP >
inline
void deep_copy
  ( const ExecSpace &
  , const View<DT,DP...> & dst
  , typename ViewTraits<DT,DP...>::const_value_type & value
  , typename std::enable_if<
    Kokkos::Impl::is_execution_space< ExecSpace >::value &&
    std::is_same< typename ViewTraits<DT,DP...>::specialize , void >::value
    >::type * = 0 )
{
  static_assert(
    std::is_same< typename ViewTraits<DT,DP...>::non_const_value_type ,
                  typename ViewTraits<DT,DP...>::value_type >::value
    , "deep_copy requires non-const type" );

  Kokkos::Impl::ViewFill< View<DT,DP...> >( dst , value );
}

/** \brief  Deep copy into a value in Host memory from a view.  */
template< class ExecSpace , class ST , class ... SP >
inline
void deep_copy
  ( const ExecSpace & exec_space
  , typename ViewTraits<ST,SP...>::non_const_value_type & dst
  , const View<ST,SP...> & src
  , typename std::enable_if<
    Kokkos::Impl::is_execution_space< ExecSpace >::value &&
    std::is_same< typename ViewTraits<ST,SP...>::specialize , void >::value
    >::type * = 0 )
{
  static_assert( ViewTraits<ST,SP...>::rank == 0
               , "ERROR: Non-rank-zero view in deep_copy( value , View )" );

  typedef ViewTraits<ST,SP...>               src_traits ;
  typedef typename src_traits::memory_space  src_memory_space ;
  Kokkos::Impl::DeepCopy< HostSpace , src_memory_space , ExecSpace >
    ( exec_space , & dst , src.data() , sizeof(ST) );
}

//----------------------------------------------------------------------------
/** \brief  A deep copy between views of compatible type, and rank zero.  */
template< class ExecSpace , class DT , class ... DP , class ST , class ... SP >
inline
void deep_copy
  ( const ExecSpace & exec_space
  , const View<DT,DP...> & dst
  , const View<ST,SP...> & src
  , typename std::enable_if<(
    Kokkos::Impl::is_execution_space< ExecSpace >::value &&
    std::is_same< typename ViewTraits<DT,DP...>::specialize , void >::value &&
    std::is_same< typename ViewTraits<ST,SP...>::specialize , void >::value &&
    ( unsigned(ViewTraits<DT,DP...>::rank) == unsigned(0) &&
      unsigned(ViewTraits<ST,SP...>::rank) == unsigned(0) )
  )>::type * = 0 )
{
  static_assert(
    std::is_same< typename ViewTraits<DT,DP...>::value_type ,
                  typename ViewTraits<ST,SP...>::non_const_value_type >::value
    , "deep_copy requires matching non-const destination type" );

  typedef View<DT,DP...>  dst_type ;
  typedef View<ST,SP...>  src_type ;

  typedef typename dst_type::value_type    value_type ;
  typedef typename dst_type::memory_space  dst_memory_space ;
  typedef typename src_type::memory_space  src_memory_space ;

  if ( dst.data() != src.data() ) {
    Kokkos::Impl::DeepCopy< dst_memory_space , src_memory_space , ExecSpace >
      ( exec_space , dst.data() , src.data() , sizeof(value_type) );
  }
}

//----------------------------------------------------------------------------
/** \brief  A deep copy between views of the default specialization, compatible type,
 *          same non-zero rank
 */
template< class ExecSpace , class DT, class ... DP, class ST, class ... SP >
inline
void deep_copy
  ( const ExecSpace & exec_space
  , const View<DT,DP...> & dst
  , const View<ST,SP...> & src
  , typename std::enable_if<(
    Kokkos::Impl::is_execution_space< ExecSpace >::value &&
    std::is_same< typename ViewTraits<DT,DP...>::specialize , void >::value &&
    std::is_same< typename ViewTraits<ST,SP...>::specialize , void >::value &&
    ( unsigned(ViewTraits<DT,DP...>::rank) != 0 ||
      unsigned(ViewTraits<ST,SP...>::rank) != 0 )
  )>::type * = 0 )
{
  static_assert(
    std::is_same< typename ViewTraits<DT,DP...>::value_type ,
                  typename ViewTraits<DT,DP...>::non_const_value_type >::value
    , "deep_copy requires non-const destination type" );

  static_assert(
    ( unsigned(ViewTraits<DT,DP...>::rank) ==
      unsigned(ViewTraits<ST,SP...>::rank) )
    , "deep_copy requires Views of equal rank" );

  typedef View<DT,DP...>  dst_type ;
  typedef View<ST,SP...>  src_type ;

  typedef typename dst_type::execution_space  dst_execution_space ;
  typedef typename src_type::execution_space  src_execution_space ;
  typedef typename dst_type::memory_space     dst_memory_space ;
  typedef typename src_type::memory_space     src_memory_space ;
  typedef typename dst_type::value_type       dst_value_type ;
  typedef typename src_type::value_type       src_value_type ;

  enum { ExecCanAccessSrcDst =
      Kokkos::Impl::SpaceAccessibility< ExecSpace , dst_memory_space >::accessible &&
      Kokkos::Impl::SpaceAccessibility< ExecSpace , src_memory_space >::accessible
  };
  enum { DstExecCanAccessSrc =
   Kokkos::Impl::SpaceAccessibility< dst_execution_space , src_memory_space >::accessible };

  enum { SrcExecCanAccessDst =
   Kokkos::Impl::SpaceAccessibility< src_execution_space , dst_memory_space >::accessible };

  // Checking for Overlapping Views.
  dst_value_type* dst_start = dst.data();
  dst_value_type* dst_end   = dst.data() + dst.span();
  src_value_type* src_start = src.data();
  src_value_type* src_end   = src.data() + src.span();
  if( ( ( (std::ptrdiff_t)dst_start < (std::ptrdiff_t)src_end ) && ( (std::ptrdiff_t)dst_end > (std::ptrdiff_t)src_start ) ) &&
      ( ( dst.span_is_contiguous() && src.span_is_contiguous() ))) {
    std::string message("Error: Kokkos::deep_copy of overlapping views: ");
    message += dst.label(); message += "(";
    message += std::to_string((std::ptrdiff_t)dst_start); message += ",";
    message += std::to_string((std::ptrdiff_t)dst_end); message += ") ";
    message += src.label(); message += "(";
    message += std::to_string((std::ptrdiff_t)src_start); message += ",";
    message += std::to_string((std::ptrdiff_t)src_end); message += ") ";
    Kokkos::Impl::throw_runtime_exception(message);
  }

  // Check for same extents
  if ( (src.extent(0) != dst.extent(0)) ||
       (src.extent(1) != dst.extent(1)) ||
       (src.extent(2) != dst.extent(2)) ||
       (src.extent(3) != dst.extent(3)) ||
       (src.extent(4) != dst.extent(4)) ||
       (src.extent(5) != dst.extent(5)) ||
       (src.extent(6) != dst.extent(6)) ||
       (src.extent(7) != dst.extent(7))
     ) {
    #ifndef KOKKOS_ENABLE_DEPRECATED_CODE_REMOVAL
      if ( ExecCanAccessSrcDst ) {
        exec_space.fence();
        Kokkos::Impl::ViewRemap< dst_type , src_type , ExecSpace >( dst , src );
        exec_space.fence();
      }
      else if ( DstExecCanAccessSrc ) {
        // Copying data between views in accessible memory spaces and either non-contiguous or incompatible shape.
        Kokkos::Impl::ViewRemap< dst_type , src_type >( dst , src );
      }
      else if ( SrcExecCanAccessDst ) {
        // Copying data between views in accessible memory spaces and either non-contiguous or incompatible shape.
        Kokkos::Impl::ViewRemap< dst_type , src_type , src_execution_space >( dst , src );
      }
      else {
        Kokkos::Impl::throw_runtime_exception("deep_copy given views that would require a temporary allocation");
      }
      return;
    #else
    std::string message("Deprecation Error: Kokkos::deep_copy extents of views don't match: ");
    message += dst.label(); message += "(";
    for(int r = 0; r<dst_type::Rank-1; r++)
      { message+= std::to_string(dst.extent(r)); message += ","; }
    message+= std::to_string(dst.extent(dst_type::Rank-1)); message += ") ";
    message += src.label(); message += "(";
    for(int r = 0; r<src_type::Rank-1; r++)
      { message+= src::to_string(src.extent(r)); message += ","; }
    message+= std::to_string(src.extent(src_type::Rank-1)); message += ") ";

    Kokkos::throw_runtime_exception(message);
    #endif
  }

  // If same type, equal layout, equal dimensions, equal span, and contiguous memory then can byte-wise copy

  if ( std::is_same< typename ViewTraits<DT,DP...>::value_type ,
                     typename ViewTraits<ST,SP...>::non_const_value_type >::value &&
       (
         std::is_same< typename ViewTraits<DT,DP...>::array_layout ,
                       typename ViewTraits<ST,SP...>::array_layout >::value
         ||
         ( ViewTraits<DT,DP...>::rank == 1 &&
           ViewTraits<ST,SP...>::rank == 1 )
       ) &&
       dst.span_is_contiguous() &&
       src.span_is_contiguous() &&
       dst.stride_0() == src.stride_0() &&
       dst.stride_1() == src.stride_1() &&
       dst.stride_2() == src.stride_2() &&
       dst.stride_3() == src.stride_3() &&
       dst.stride_4() == src.stride_4() &&
       dst.stride_5() == src.stride_5() &&
       dst.stride_6() == src.stride_6() &&
       dst.stride_7() == src.stride_7()
    ) {

    const size_t nbytes = sizeof(typename dst_type::value_type) * dst.span();

    Kokkos::Impl::DeepCopy< dst_memory_space , src_memory_space , ExecSpace >
      ( exec_space , dst.data() , src.data() , nbytes );
  }
  else if ( ExecCanAccessSrcDst ) {
    exec_space.fence();
    Kokkos::Impl::ViewRemap< dst_type , src_type , ExecSpace >( dst , src );
    exec_space.fence();
  }
  else if ( DstExecCanAccessSrc ) {
    // Copying data between views in accessible memory spaces and either non-contiguous or incompatible shape.
    exec_space.fence();
    Kokkos::Impl::ViewRemap< dst_type , src_type >( dst , src );
    exec_space.fence();
  }
  else if ( SrcExecCanAccessDst ) {
    // Copying data between views in accessible memory spaces and either non-contiguous or incompatible shape.
    exec_space.fence();
    Kokkos::Impl::ViewRemap< dst_type , src_type , src_execution_space >( dst , src );
    exec_space.fence();
  }
  else {
    Kokkos::Impl::throw_runtime_exception("deep_copy given views that would require a temporary allocation");
  }
}

} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

/** \brief  Resize a view with copying old data to new data at the corresponding indices. */
template< class T , class ... P >
inline
typename std::enable_if<
  std::is_same<typename Kokkos::View<T,P...>::array_layout,Kokkos::LayoutLeft>::value ||
  std::is_same<typename Kokkos::View<T,P...>::array_layout,Kokkos::LayoutRight>::value
>::type
resize( Kokkos::View<T,P...> & v ,
             const size_t n0 = 0 ,
             const size_t n1 = 0 ,
             const size_t n2 = 0 ,
             const size_t n3 = 0 ,
             const size_t n4 = 0 ,
             const size_t n5 = 0 ,
             const size_t n6 = 0 ,
             const size_t n7 = 0 )
{
  typedef Kokkos::View<T,P...>  view_type ;

  static_assert( Kokkos::ViewTraits<T,P...>::is_managed , "Can only resize managed views" );

  // Fix #904 by checking dimensions before actually resizing.
  //
  // Rank is known at compile time, so hopefully the compiler will
  // remove branches that are compile-time false.  The upcoming "if
  // constexpr" language feature would make this certain.
  if (view_type::Rank == 1 &&
      n0 == static_cast<size_t> (v.extent(0))) {
    return;
  }
  if (view_type::Rank == 2 &&
      n0 == static_cast<size_t> (v.extent(0)) &&
      n1 == static_cast<size_t> (v.extent(1))) {
    return;
  }
  if (view_type::Rank == 3 &&
      n0 == static_cast<size_t> (v.extent(0)) &&
      n1 == static_cast<size_t> (v.extent(1)) &&
      n2 == static_cast<size_t> (v.extent(2))) {
    return;
  }
  if (view_type::Rank == 4 &&
      n0 == static_cast<size_t> (v.extent(0)) &&
      n1 == static_cast<size_t> (v.extent(1)) &&
      n2 == static_cast<size_t> (v.extent(2)) &&
      n3 == static_cast<size_t> (v.extent(3))) {
    return;
  }
  if (view_type::Rank == 5 &&
      n0 == static_cast<size_t> (v.extent(0)) &&
      n1 == static_cast<size_t> (v.extent(1)) &&
      n2 == static_cast<size_t> (v.extent(2)) &&
      n3 == static_cast<size_t> (v.extent(3)) &&
      n4 == static_cast<size_t> (v.extent(4))) {
    return;
  }
  if (view_type::Rank == 6 &&
      n0 == static_cast<size_t> (v.extent(0)) &&
      n1 == static_cast<size_t> (v.extent(1)) &&
      n2 == static_cast<size_t> (v.extent(2)) &&
      n3 == static_cast<size_t> (v.extent(3)) &&
      n4 == static_cast<size_t> (v.extent(4)) &&
      n5 == static_cast<size_t> (v.extent(5))) {
    return;
  }
  if (view_type::Rank == 7 &&
      n0 == static_cast<size_t> (v.extent(0)) &&
      n1 == static_cast<size_t> (v.extent(1)) &&
      n2 == static_cast<size_t> (v.extent(2)) &&
      n3 == static_cast<size_t> (v.extent(3)) &&
      n4 == static_cast<size_t> (v.extent(4)) &&
      n5 == static_cast<size_t> (v.extent(5)) &&
      n6 == static_cast<size_t> (v.extent(6))) {
    return;
  }
  if (view_type::Rank == 8 &&
      n0 == static_cast<size_t> (v.extent(0)) &&
      n1 == static_cast<size_t> (v.extent(1)) &&
      n2 == static_cast<size_t> (v.extent(2)) &&
      n3 == static_cast<size_t> (v.extent(3)) &&
      n4 == static_cast<size_t> (v.extent(4)) &&
      n5 == static_cast<size_t> (v.extent(5)) &&
      n6 == static_cast<size_t> (v.extent(6)) &&
      n7 == static_cast<size_t> (v.extent(7))) {
    return;
  }
  // If Kokkos ever supports Views of rank > 8, the above code won't
  // be incorrect, because avoiding reallocation in resize() is just
  // an optimization.

  // TODO (mfh 27 Jun 2017) If the old View has enough space but just
  // different dimensions (e.g., if the product of the dimensions,
  // including extra space for alignment, will not change), then
  // consider just reusing storage.  For now, Kokkos always
  // reallocates if any of the dimensions change, even if the old View
  // has enough space.

  view_type v_resized( v.label(), n0, n1, n2, n3, n4, n5, n6, n7 );

  Kokkos::Impl::ViewRemap< view_type , view_type >( v_resized , v );

  v = v_resized ;
}

/** \brief  Resize a view with copying old data to new data at the corresponding indices. */
template< class T , class ... P >
inline
void resize(       Kokkos::View<T,P...> & v ,
    const typename Kokkos::View<T,P...>::array_layout & layout)
{
  typedef Kokkos::View<T,P...>  view_type ;

  static_assert( Kokkos::ViewTraits<T,P...>::is_managed , "Can only resize managed views" );

  view_type v_resized( v.label(), layout );

  Kokkos::Impl::ViewRemap< view_type , view_type >( v_resized , v );

  v = v_resized ;
}

/** \brief  Resize a view with discarding old data. */
template< class T , class ... P >
inline
typename std::enable_if<
  std::is_same<typename Kokkos::View<T,P...>::array_layout,Kokkos::LayoutLeft>::value ||
  std::is_same<typename Kokkos::View<T,P...>::array_layout,Kokkos::LayoutRight>::value
>::type
realloc( Kokkos::View<T,P...> & v ,
              const size_t n0 = 0 ,
              const size_t n1 = 0 ,
              const size_t n2 = 0 ,
              const size_t n3 = 0 ,
              const size_t n4 = 0 ,
              const size_t n5 = 0 ,
              const size_t n6 = 0 ,
              const size_t n7 = 0 )
{
  typedef Kokkos::View<T,P...>  view_type ;

  static_assert( Kokkos::ViewTraits<T,P...>::is_managed , "Can only realloc managed views" );

  const std::string label = v.label();

  v = view_type(); // Deallocate first, if the only view to allocation
  v = view_type( label, n0, n1, n2, n3, n4, n5, n6, n7 );
}

/** \brief  Resize a view with discarding old data. */
template< class T , class ... P >
inline
void realloc(      Kokkos::View<T,P...> & v ,
    const typename Kokkos::View<T,P...>::array_layout & layout)
{
  typedef Kokkos::View<T,P...>  view_type ;

  static_assert( Kokkos::ViewTraits<T,P...>::is_managed , "Can only realloc managed views" );

  const std::string label = v.label();

  v = view_type(); // Deallocate first, if the only view to allocation
  v = view_type( label, layout );
}
} /* namespace Kokkos */

#endif
