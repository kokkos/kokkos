
#ifndef KOKKOS_CILK_REDUCER_H_
#define KOKKOS_CILK_REDUCER_H_

#include <cilk/cilk.h>
#include <cilk/reducer.h>
#include <memoryweb/intrinsics.h>

namespace Kokkos {
namespace Impl {

static void * global_reducer = NULL;

template <class ReducerType>
ReducerType * get_reducer() {
   ReducerType * pRet = (ReducerType *)global_reducer;
   return &(pRet[NODE_ID()]);
}

template <class ReduceWrapper, class ReducerType, class WorkTagFwd, class T = void>
struct CilkReduceContainer;

template <class ReduceWrapper, class ReducerType, class WorkTagFwd, class T = void>
struct CilkEmuReduceView;

template <class ReduceWrapper, class ReducerType, class WorkTagFwd>
struct CilkEmuReduceView<ReduceWrapper, ReducerType, WorkTagFwd, typename std::enable_if< Kokkos::is_reducer_type<ReducerType>::value ||
                                                                 Kokkos::is_view<ReducerType>::value>::type >
{

  typedef typename std::remove_reference<typename ReducerType::value_type>::type nr_value_type;
  typedef typename std::remove_pointer<nr_value_type>::type np_value_type;
  typedef typename std::remove_const<np_value_type>::type rd_value_type; 
  typedef Kokkos::Impl::FunctorValueJoin< ReducerType, WorkTagFwd >   ValueJoin;

  rd_value_type & val;


public:
  inline
  CilkEmuReduceView( rd_value_type & val_ ) : val(val_) {
  }

  inline
  void join( rd_value_type right ) {
     ReduceWrapper* ptr = get_reducer<ReduceWrapper>();
     if (ptr)
     {
//        printf("reducer view join (B): %ld, %ld \n", val, right);
        ValueJoin::join( ptr->r, &val, &right );
//        printf("reducer view join (A): %ld, %ld \n", val, right);
     }  
  }

  inline
  static rd_value_type create( rd_value_type val_ ) {
     return val_;
  }

  inline 
  static void destroy( rd_value_type val_ ) {
  }
};


// Reducer/View access via value
template <class ReduceWrapper, class ReducerType, class WorkTagFwd>
struct CilkReduceContainer<ReduceWrapper, ReducerType, WorkTagFwd, typename std::enable_if< Kokkos::is_reducer_type<ReducerType>::value ||
                                                                 Kokkos::is_view<ReducerType>::value>::type >
{
  Kokkos::HostSpace space;
  enum { isPointer = 1 };
  typedef typename std::remove_reference<typename ReducerType::value_type>::type nr_value_type;
  typedef typename std::remove_pointer<nr_value_type>::type np_value_type;
  typedef typename std::remove_const<np_value_type>::type rd_value_type; 
  typedef Kokkos::Impl::FunctorValueJoin< ReducerType, WorkTagFwd >   ValueJoin;
  typedef Kokkos::Impl::FunctorValueInit< ReducerType, WorkTagFwd >   ValueInit;

  rd_value_type val;
  size_t alloc_bytes;

  typedef CilkEmuReduceView< ReduceWrapper, ReducerType, WorkTagFwd > ViewType;
  typedef rd_value_type ElementType;

  inline
  void identity( rd_value_type * val )
  {
     ReduceWrapper* ptr = get_reducer<ReduceWrapper>();
     if (ptr)
     {
        ValueInit::init( ptr->r, val );
//        printf("reducer init: %ld \n", *val);
     }
  }


  inline
  void reduce( rd_value_type * left, rd_value_type const * right )
  {
     ReduceWrapper* ptr = get_reducer<ReduceWrapper>();
     if (ptr)
     {
//        printf("reducer reduce: %ld, %ld \n", *left, *right);
//        ValueJoin::join( ptr->r, left, right );
     }
  }

};

template <class ReduceWrapper, class ReducerType, class WorkTagFwd>
struct CilkEmuReduceView<ReduceWrapper, ReducerType, WorkTagFwd, typename std::enable_if< !Kokkos::is_reducer_type<ReducerType>::value &&
                                             !Kokkos::is_view<ReducerType>::value && 
                                           ( std::is_array< typename ReducerType::value_type >::value || 
                                             std::is_pointer< typename ReducerType::value_type >::value ) >::type >   {

  typedef typename std::remove_reference<typename ReducerType::value_type>::type nr_value_type;
  typedef typename std::remove_pointer<nr_value_type>::type np_value_type;
  typedef typename std::remove_extent<np_value_type>::type ne_value_type;
  typedef typename std::remove_const<ne_value_type>::type rd_value_type; 

  typedef Kokkos::Impl::FunctorValueJoin< ReducerType, WorkTagFwd >   ValueJoin;

  rd_value_type * val;

public:
  // need to allocate space for val as it is just a pointer, then 
  // copy *val_ 
  inline
  CilkEmuReduceView( rd_value_type * val_ ) {
     printf("array view constructor: %08x \n", val_);
     ReduceWrapper* ptr = get_reducer<ReduceWrapper>();
     if (ptr)
     {        
        Kokkos::HostSpace space;
        val = (rd_value_type *)space.allocate( ptr->alloc_bytes );
        *val = *val_;
     } else {
        val = val_;
     }
  }

  inline
  void join( rd_value_type * right ) {
     ReduceWrapper* ptr = get_reducer<ReduceWrapper>();
     if (ptr)
     {
        printf("(B) array view join (P): %ld, %ld: %08x, %08x \n", val[0], right[0], val, right);
        ValueJoin::join( ptr->r, val, right );
        printf("(A) array view join (P): %ld, %ld: %08x, %08x \n", val[0], right[0], val, right);
     }  
  }

  inline
  static rd_value_type * create( rd_value_type * val_ ) {     
     rd_value_type * lVal = 0;
     ReduceWrapper* ptr = get_reducer<ReduceWrapper>();
     if (ptr)
     {        
        Kokkos::HostSpace space;
        lVal = (rd_value_type *)space.allocate( ptr->alloc_bytes );
        *lVal = *val_;
     } else {
        lVal = val_;
     }
     printf("create array view memory: %08x, %08x \n", val_, lVal);
     return lVal;
  }

  inline 
  static void destroy( rd_value_type * val_ ) {
     printf("array view destroy: %08x \n", val_);
     ReduceWrapper* ptr = get_reducer<ReduceWrapper>();
     if (ptr)
     {        
        Kokkos::HostSpace space;
        space.deallocate( val_, ptr->alloc_bytes );
     }
  }

  inline ~CilkEmuReduceView() {
     if (val) CilkEmuReduceView::destroy(val);
     val = NULL;
  }

};

// Functor with array/pointer
template <class ReduceWrapper, class ReducerType, class WorkTagFwd>
struct CilkReduceContainer<ReduceWrapper, ReducerType, WorkTagFwd, typename std::enable_if< !Kokkos::is_reducer_type<ReducerType>::value &&
                                             !Kokkos::is_view<ReducerType>::value && 
                                           ( std::is_array< typename ReducerType::value_type >::value || 
                                             std::is_pointer< typename ReducerType::value_type >::value ) >::type >   {

  typedef typename std::remove_reference<typename ReducerType::value_type>::type nr_value_type;
  typedef typename std::remove_pointer<nr_value_type>::type np_value_type;
  typedef typename std::remove_extent<np_value_type>::type ne_value_type;
  typedef typename std::remove_const<ne_value_type>::type rd_value_type; 
  typedef Kokkos::Impl::FunctorValueJoin< ReducerType, WorkTagFwd >   ValueJoin;
  typedef Kokkos::Impl::FunctorValueInit< ReducerType, WorkTagFwd >   ValueInit;

  typedef CilkEmuReduceView< ReduceWrapper, ReducerType, WorkTagFwd > ViewType;
  typedef rd_value_type * ElementType;

  rd_value_type * val;
  size_t alloc_bytes;

  inline
  void identity( rd_value_type * val )
  {
     ReduceWrapper* ptr = get_reducer<ReduceWrapper>();
     if (ptr)
     {
        ValueInit::init( ptr->r, val );
     }
  }

  inline
  void reduce( rd_value_type * left, rd_value_type const * right )
  {
     printf("array reduce : %ld, %ld, %08x %08x \n", left[0], right[0], left, right);
     ReduceWrapper* ptr = get_reducer<ReduceWrapper>();
     if (ptr)
     {
        ValueJoin::join( ptr->r, left, right );
     }
  }

};

// non-pointer, non-array view
template <class ReduceWrapper, class ReducerType, class WorkTagFwd>
struct CilkEmuReduceView<ReduceWrapper, ReducerType, WorkTagFwd, typename std::enable_if< !Kokkos::is_reducer_type<ReducerType>::value &&
                                             !Kokkos::is_view<ReducerType>::value && 
                                           ( !std::is_array< typename ReducerType::value_type >::value && 
                                             !std::is_pointer< typename ReducerType::value_type >::value ) >::type >   {

  typedef typename std::remove_reference<typename ReducerType::value_type>::type nr_value_type;
  typedef typename std::remove_pointer<nr_value_type>::type np_value_type;
  typedef typename std::remove_const<np_value_type>::type rd_value_type; 
  typedef Kokkos::Impl::FunctorValueJoin< ReducerType, WorkTagFwd >   ValueJoin;

  rd_value_type & val;

public:
  inline
  CilkEmuReduceView( rd_value_type & val_ ) : val(val_) {
  }

  inline
  void join( rd_value_type right ) {
     ReduceWrapper* ptr = get_reducer<ReduceWrapper>();
     if (ptr)
     {
//        printf("reducer view join (S): %ld, %ld \n", val.value[0], right.value[0]);
        ValueJoin::join( ptr->r, &val, &right );
     }  
  }

  inline
  static rd_value_type create( rd_value_type val_ ) {
     return val_;
  }

  inline 
  static void destroy( rd_value_type val_ ) {
  }

};


// non-pointer, non-array moniod 
template <class ReduceWrapper, class ReducerType, class WorkTagFwd>
struct CilkReduceContainer<ReduceWrapper, ReducerType, WorkTagFwd, typename std::enable_if< !Kokkos::is_reducer_type<ReducerType>::value &&
                                             !Kokkos::is_view<ReducerType>::value && 
                                           ( !std::is_array< typename ReducerType::value_type >::value && 
                                             !std::is_pointer< typename ReducerType::value_type >::value ) >::type >   {

  typedef typename std::remove_reference<typename ReducerType::value_type>::type nr_value_type;
  typedef typename std::remove_pointer<nr_value_type>::type np_value_type;
  typedef typename std::remove_const<np_value_type>::type rd_value_type; 

  typedef CilkEmuReduceView< ReduceWrapper, ReducerType, WorkTagFwd > ViewType;
  typedef rd_value_type ElementType;

  typedef Kokkos::Impl::FunctorValueJoin< ReducerType, WorkTagFwd >   ValueJoin;
  typedef Kokkos::Impl::FunctorValueInit< ReducerType, WorkTagFwd >  ValueInit;

  inline
  void identity( rd_value_type * val )
  {
     ReduceWrapper* ptr = get_reducer<ReduceWrapper>();
     if (ptr)
     {
        ValueInit::init( ptr->r, val );
     }
  }

  inline
  void reduce( rd_value_type * left, rd_value_type const * right )
  {
//     printf("reducer - reduce (S): %ld, %ld \n", (*left).value[0], (*right).value[0]);
     ReduceWrapper* ptr = get_reducer<ReduceWrapper>();
     if (ptr)
     {
//        ValueJoin::join( ptr->r, left, right );
     }
  }

};

template< typename F , typename = std::false_type >
struct lambda_only { using type = void ; };

template< typename F >
struct lambda_only
< F , typename std::is_same< typename F::value_type , void >::type >
{
  using type = typename F::value_type ; 
};

template <typename ReducerType, class Functor, class defaultType, class WorkTagFwd , class T = void>
struct kokkos_cilk_reducer;

template <typename ReducerType, class Functor, class defaultType, class WorkTagFwd >
struct kokkos_cilk_reducer< ReducerType, Functor, defaultType, WorkTagFwd , typename std::enable_if< 
                                                     Kokkos::is_reducer_type<ReducerType>::value ||
                                                     Kokkos::is_view<ReducerType>::value>::type > {

    typedef Kokkos::Impl::if_c< std::is_same< typename lambda_only< Functor >::type, void >::value, ReducerType, Functor> ReducerTypeCond;

    typedef typename ReducerTypeCond::type ReducerTypeFwd;

    typedef CilkReduceContainer< kokkos_cilk_reducer, typename Kokkos::Impl::if_c< 
                                           Kokkos::is_view<ReducerType>::value, Functor, ReducerType>::type, 
                                                           WorkTagFwd >    reduce_container;
    typedef Kokkos::Impl::FunctorValueJoin< typename Kokkos::Impl::if_c< 
                                           Kokkos::is_view<ReducerType>::value, Functor, ReducerType>::type, 
                                                           WorkTagFwd >    ValueJoin;

    typedef Kokkos::Impl::FunctorValueInit< typename Kokkos::Impl::if_c< 
                                           Kokkos::is_view<ReducerType>::value, Functor, ReducerType>::type, 
                                                           WorkTagFwd >    ValueInit;

    cilk::reducer < reduce_container > * local_reducer = NULL;

    const ReducerType f;
    const ReducerType r;
    const size_t alloc_bytes;

    kokkos_cilk_reducer (const ReducerType & f_, const size_t l_alloc_bytes) : f(f_), alloc_bytes(l_alloc_bytes) {        
        typename ReducerTypeFwd::value_type lVal;
        ValueInit::init( f, &lVal );
        local_reducer = new cilk::PerNodeletReducer< reduce_container >(lVal);
    }

    void join(typename ReducerTypeFwd::value_type & val_) {
        typename reduce_container::ViewType * cont = local_reducer->view();
        cont->join( val_ );
    }

    void update_value(typename ReducerTypeFwd::value_type & ret) {
       ret = local_reducer->get_value();
    }

    void release_resources() {
    }

};

// case where the functor is actually a lambda -- implicit Add operation.
template <typename ReducerType, class Functor, class defaultType, class WorkTagFwd >
struct kokkos_cilk_reducer< ReducerType, Functor, defaultType, WorkTagFwd , typename std::enable_if< !Kokkos::is_reducer_type<ReducerType>::value &&
                                                                                        !Kokkos::is_view<ReducerType>::value &&
                                                                         std::is_same< typename lambda_only< Functor >::type, void >::value  >::type > {

    typedef CilkReduceContainer< kokkos_cilk_reducer, Kokkos::Experimental::Sum< defaultType >, void >   reduce_container;

    typedef Kokkos::Impl::FunctorValueJoin< Kokkos::Experimental::Sum< defaultType >, WorkTagFwd >  ValueJoin;
    typedef Kokkos::Impl::FunctorValueInit< Kokkos::Experimental::Sum< defaultType >, WorkTagFwd >  ValueInit;

    defaultType local_value;
    const Functor f;
    const Kokkos::Experimental::Sum< defaultType > r;
    const size_t alloc_bytes;
    
    cilk::reducer < reduce_container > * local_reducer = NULL;

    kokkos_cilk_reducer (const Functor & f_, const size_t l_alloc_bytes) : local_value(0), f(f_), 
                                                                           r(local_value), alloc_bytes(l_alloc_bytes) {
        defaultType lVal = 0;
        local_reducer = new cilk::PerNodeletReducer< reduce_container >(lVal);
    }

    void join(defaultType & val_) {
        typename reduce_container::ViewType * cont = local_reducer->view();
        cont->join( val_ );
    }

    void update_value(defaultType & ret) {
       ret = local_reducer->get_value();
    }

    void release_resources() {
    }


};


template <typename ReducerType, class Functor, class defaultType, class WorkTagFwd >
struct kokkos_cilk_reducer< ReducerType, Functor, defaultType, WorkTagFwd , typename std::enable_if< !Kokkos::is_reducer_type<ReducerType>::value &&
                                                                           !Kokkos::is_view<ReducerType>::value &&
                                           ( std::is_array< typename Functor::value_type >::value || 
                                             std::is_pointer< typename Functor::value_type >::value ) >::type > {

    typedef CilkReduceContainer< kokkos_cilk_reducer, Functor, WorkTagFwd > reduce_container;
    typedef Kokkos::Impl::FunctorValueJoin< Functor, WorkTagFwd >   ValueJoin;
    typedef Kokkos::Impl::FunctorValueInit< Functor, WorkTagFwd >  ValueInit;
  
    const Functor f;
    const Functor r;
    size_t alloc_bytes;

    typename reduce_container::rd_value_type * lVal = NULL;
    cilk::reducer < reduce_container > * local_reducer = NULL;

//    inline
//    kokkos_cilk_reducer & operator = ( const kokkos_cilk_reducer & rhs ) { 
//       f = rhs.f ; alloc_bytes = rhs.alloc_bytes ; lVal = rhs.lVal ; local_reducer = rhs.local_reducer ; return *this ; }

    kokkos_cilk_reducer (const Functor & f_, const size_t l_alloc_bytes) : f(f_), r(f_), alloc_bytes(l_alloc_bytes) {
        uint64_t myRed = (long) mw_ptr0to1(this);
        this->lVal = (typename reduce_container::rd_value_type *)mw_localmalloc( l_alloc_bytes, (void*)myRed );

        // construct functor on nodelet
        new ((void *)&(this->f)) Functor(f_);
        ValueInit::init( this->f, this->lVal );

        local_reducer = new cilk::PerNodeletReducer< reduce_container >(this->lVal);

    }

    void join(typename reduce_container::rd_value_type * val_) {
        printf("reducer join: %ld\n", (long)NODE_ID());
        fflush(stdout);
        
        typename reduce_container::ViewType * cont = local_reducer->view();
        cont->join( val_ );
    }

    void update_value(typename reduce_container::rd_value_type * ret) {
       typename reduce_container::rd_value_type * lRet = local_reducer->get_value();
       *ret = *lRet;
    }

    void release_resources() {
    }
    
};


// non-pointer non-array -- could be struct or scalar
template <typename ReducerType, class Functor, class defaultType, class WorkTagFwd >
struct kokkos_cilk_reducer< ReducerType, Functor, defaultType, WorkTagFwd , typename std::enable_if< !Kokkos::is_reducer_type<ReducerType>::value &&
                                                                           !Kokkos::is_view<ReducerType>::value &&
                                           ( !std::is_array< typename Functor::value_type >::value &&
                                             !std::is_pointer< typename Functor::value_type >::value ) >::type >   {

    typedef CilkReduceContainer< kokkos_cilk_reducer, ReducerType, WorkTagFwd > reduce_container;
    typedef Kokkos::Impl::FunctorValueJoin< Functor, WorkTagFwd >  ValueJoin;
    typedef Kokkos::Impl::FunctorValueInit< Functor, WorkTagFwd >  ValueInit;

    cilk::reducer < reduce_container > * local_reducer = NULL;

    const Functor f;
    const Functor r;
    const size_t alloc_bytes;

    kokkos_cilk_reducer (const Functor & f_, const size_t l_alloc_bytes) : f(f_), r(f_), alloc_bytes(l_alloc_bytes) {        
        typename Functor::value_type lVal;
        ValueInit::init( f, &lVal );
        local_reducer = new cilk::PerNodeletReducer< reduce_container >(lVal);
    }

   void join(typename Functor::value_type & val_) {
        typename reduce_container::ViewType * cont = local_reducer->view();
        cont->join( val_ );
    }
    
    void update_value(typename Functor::value_type & ret) {
        ret = local_reducer->get_value();
    }

    void release_resources() {
    }


};

} 
}

#define INITIALIZE_CILK_REDUCER( wrapper, reducer ) {}


#endif
