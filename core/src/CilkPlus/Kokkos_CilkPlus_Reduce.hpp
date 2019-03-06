
#ifndef KOKKOS_CILK_REDUCER_H_
#define KOKKOS_CILK_REDUCER_H_

#include <cilk/cilk.h>
#include <cilk/reducer.h>


namespace Kokkos {
namespace Impl {

template <class ReducerType, class WorkTagFwd, class T = void>
struct CilkReduceContainer;


// Value only
template <class ReducerType, class WorkTagFwd>
struct CilkReduceContainer<ReducerType, WorkTagFwd, typename std::enable_if< Kokkos::is_reducer_type<ReducerType>::value ||
                                                                 Kokkos::is_view<ReducerType>::value>::type >
{
  Kokkos::HostSpace space;
  enum { isPointer = 1 };
  typedef typename std::remove_reference<typename ReducerType::value_type>::type nr_value_type;
  typedef typename std::remove_pointer<nr_value_type>::type np_value_type;
  typedef typename std::remove_const<np_value_type>::type rd_value_type; 
  typedef Kokkos::Impl::FunctorValueJoin< ReducerType, WorkTagFwd >   ValueJoin;

  rd_value_type val;
  size_t alloc_bytes;

/*  CilkReduceContainer(  ) : val(NULL), alloc_bytes(0) {}
  CilkReduceContainer( rd_value_type & val_ ) : val(&val_), alloc_bytes(0) {}
  CilkReduceContainer( rd_value_type * val_ ) : val(val_), alloc_bytes(0) {}*/

  void operator=( CilkReduceContainer & crc ) const {
     val = crc.val;
     alloc_bytes = alloc_bytes;
  }

  void initializeContents( const size_t l_alloc_bytes) {
      alloc_bytes = l_alloc_bytes;
      // printf("calling alloc(initialize Contents): %d \n", l_alloc_bytes);
      //val = (rd_value_type*) space.allocate( alloc_bytes );
      //val = (rd_value_type*) malloc( alloc_bytes );
      //printf("view contents created: %d, %X \n", alloc_bytes, (unsigned long)val);
  }

  void initializeData( const ReducerType & f ) {
    f.init(val);
  }

  void getValue( rd_value_type & val_ ) {
     //printf("copy value %X, %X \n", (unsigned long)val_, (unsigned long)val);
     val_ = val;
  }

  rd_value_type * getReference() {
     return &val;
  }

  void join( const ReducerType & f, const typename ReducerType::value_type val_ )
  {
     ValueJoin::join( f, &val, &val_ );
  }

  void freeMemory(  ) {
 //    if (val && alloc_bytes > 0) {
         // printf("freeing memory: %d \n", alloc_bytes);
         //space.deallocate(val,alloc_bytes);
//         free(val);
//         val = NULL;
//     }
  }

  ~CilkReduceContainer() {
     // printf("in destructor \n");
  }

};

template <class ReducerType, class WorkTagFwd>
struct CilkReduceContainer<ReducerType, WorkTagFwd, typename std::enable_if< !Kokkos::is_reducer_type<ReducerType>::value &&
                                             !Kokkos::is_view<ReducerType>::value && 
                                           ( std::is_array< typename ReducerType::value_type >::value || 
                                             std::is_pointer< typename ReducerType::value_type >::value ) >::type >   {

  enum { isPointer = 1 };

  Kokkos::HostSpace space;
  typedef typename std::remove_reference<typename ReducerType::value_type>::type nr_value_type;
  typedef typename std::remove_pointer<nr_value_type>::type np_value_type;
  typedef typename std::remove_extent<np_value_type>::type ne_value_type;
  typedef typename std::remove_const<ne_value_type>::type rd_value_type; 
  typedef Kokkos::Impl::FunctorValueJoin< ReducerType, WorkTagFwd >   ValueJoin;

  rd_value_type * val;
  size_t alloc_bytes;

/*  CilkReduceContainer(  ) : val(NULL), alloc_bytes(0) {}
  CilkReduceContainer( rd_value_type & val_ ) : val(&val_), alloc_bytes(0) {}
  CilkReduceContainer( rd_value_type * val_ ) : val(val_), alloc_bytes(0) {}*/
  
  void operator=( CilkReduceContainer & crc ) const {
     val = crc.val;
     alloc_bytes = alloc_bytes;
  }


  void initializeContents( const size_t l_alloc_bytes) {
      alloc_bytes = l_alloc_bytes;
      // printf("calling alloc(initialize Contents): %d \n", l_alloc_bytes);
      //val = (rd_value_type*) space.allocate( alloc_bytes );
      val = (rd_value_type*) malloc( alloc_bytes );
      //printf("view contents created: %d, %X \n", alloc_bytes, (unsigned long)val);
  }

  void getValue( rd_value_type * val_ ) {
     //printf("copy value %X, %X \n", (unsigned long)val_, (unsigned long)val);
     memcpy( val_, val, alloc_bytes);
  }

  rd_value_type * getReference() {
     return val;
  }
  
  void freeMemory(  ) {
     if (val && alloc_bytes > 0) { 
         // printf("freeing memory: %d \n", alloc_bytes);
         //space.deallocate(val,alloc_bytes);
         free(val);
         val = NULL;
     }
  }

  ~CilkReduceContainer() {
     // printf("in destructor \n");
  }

};

template <class ReducerType, class WorkTagFwd>
struct CilkReduceContainer<ReducerType, WorkTagFwd, typename std::enable_if< !Kokkos::is_reducer_type<ReducerType>::value &&
                                             !Kokkos::is_view<ReducerType>::value && 
                                           ( !std::is_array< typename ReducerType::value_type >::value && 
                                             !std::is_pointer< typename ReducerType::value_type >::value ) >::type >   {

  enum { isPointer = 0 };
  typedef typename std::remove_reference<typename ReducerType::value_type>::type nr_value_type;
  typedef typename std::remove_pointer<nr_value_type>::type np_value_type;
  typedef typename std::remove_const<np_value_type>::type rd_value_type; 
  typedef Kokkos::Impl::FunctorValueJoin< ReducerType, WorkTagFwd >   ValueJoin;

  rd_value_type val;

/*  CilkReduceContainer(  )  {}
  CilkReduceContainer( rd_value_type & val_ ) : val(&val_) {}
  CilkReduceContainer( rd_value_type * val_ ) : val(val_) {}*/
  
  void operator=( CilkReduceContainer & crc ) const {
     val = crc.val;
  }


  // This does nothing for the reference (non-pointer) value_type 
  void initializeContents( const size_t l_alloc_bytes) {
     //printf("initializing view memory: %d\n", l_alloc_bytes);
     memset(&val, 0, l_alloc_bytes);
  }

  void getValue( rd_value_type & val_ ) {
     val_ = val;
  }

  rd_value_type & getReference() {
     return val;
  }

  void join( const ReducerType & f, const typename ReducerType::value_type val_ )
  {
     ValueJoin::join( f, &val, &val_ );
  }

  void freeMemory(  ) {
  } 

  ~CilkReduceContainer() {
     // printf("in destructor \n");
  }
};


static void * global_reducer = NULL;

template<class Reducer>
void value_reduce ( void * reducer, void * left, void * right )
{
   // printf("--> value_reduce \n");
   if (global_reducer != NULL)
      ((Reducer*)global_reducer)->reduce((typename Reducer::reduce_container*)left, (typename Reducer::reduce_container*)right);
}

template<class Reducer>
void value_init ( void * reducer, void * data )
{ 
   // printf("--> value_init \n");
   Reducer* ptr = ((Reducer*)global_reducer);
   if (ptr)
   {
      ptr->initializeView(data);
   }
}

template<class Reducer>
void value_dealloc ( void * reducer, void * data )
{ 
   // printf("--> value_dealloc \n");
   Reducer* ptr = ((Reducer*)global_reducer);
   if (ptr)
   {
      ptr->cleanupView(data);
   }
}

template <typename ReducerType, class Functor, class WorkTagFwd , class T = void>
struct kokkos_cilk_reducer;

template <typename ReducerType, class Functor, class WorkTagFwd >
struct kokkos_cilk_reducer< ReducerType, Functor, WorkTagFwd , typename std::enable_if< 
                                                     Kokkos::is_reducer_type<ReducerType>::value ||
                                                     Kokkos::is_view<ReducerType>::value>::type > {

    typedef CilkReduceContainer< typename Kokkos::Impl::if_c< Kokkos::is_view<ReducerType>::value, Functor, ReducerType>::type, WorkTagFwd > reduce_container;
    typedef Kokkos::Impl::FunctorValueJoin< typename Kokkos::Impl::if_c< Kokkos::is_view<ReducerType>::value, Functor, ReducerType>::type, WorkTagFwd >   ValueJoin;
    CILK_C_DECLARE_REDUCER( reduce_container ) kr;

    const ReducerType f;
    const size_t alloc_bytes;

    kokkos_cilk_reducer (const ReducerType & f_, const size_t l_alloc_bytes) : f(f_), alloc_bytes(l_alloc_bytes) {
    }

    void initializeView(void * data) {
      ((reduce_container*)data)->initializeContents(alloc_bytes);
      //printf("calling functor init \n");

      ((reduce_container*)data)->initializeData(f);
      //int* trythis = (int*)((reduce_container*)data)->getReference();
      //printf("after init: %d  \n", trythis[0]); //, trythis[1], trythis[2]);
    }

    void cleanupView(void * data) {
      ((reduce_container*)data)->freeMemory();
    }

    void init() {
         initializeView(&REDUCER_VIEW(kr));
         CILK_C_REGISTER_REDUCER(kr);
    }

    void join(typename ReducerType::value_type val_) {
        reduce_container * cont = &REDUCER_VIEW(kr);

        //int* trythis = (int*)cont->getReference();
        //printf("before join: %d  \n", trythis[0]); //, trythis[1], trythis[2]);

        cont->join( f, val_ );

        //trythis = (int*)cont->getReference();
        //printf("after join(%d): %d  \n", val_, trythis[0]); //, trythis[1], trythis[2]);

    }

    void update_value(typename ReducerType::value_type & ret) {
       reduce_container * cont = &REDUCER_VIEW(kr);
       cont->getValue(ret);
       //long* trythis = (long*)(ret);
       //printf("update value: %d, %d, %d  \n", trythis[0], trythis[1], trythis[2]);
    }

    void reduce(reduce_container* left, reduce_container* right ) {
      typename ReducerType::value_type * rRef = right->getReference();

      left->join(f, *rRef);

    }

    void release_resources()
    {
//       REDUCER_VIEW(kr).freeMemory();
       CILK_C_UNREGISTER_REDUCER(kr);
    }

};


template <typename ReducerType, class Functor, class WorkTagFwd >
struct kokkos_cilk_reducer< ReducerType, Functor, WorkTagFwd , typename std::enable_if< !Kokkos::is_reducer_type<ReducerType>::value &&
                                                                           !Kokkos::is_view<ReducerType>::value &&
                                           ( std::is_array< typename Functor::value_type >::value || 
                                             std::is_pointer< typename Functor::value_type >::value ) >::type > {

    typedef CilkReduceContainer< ReducerType, WorkTagFwd > reduce_container;
    typedef Kokkos::Impl::FunctorValueJoin< Functor, WorkTagFwd >   ValueJoin;
    CILK_C_DECLARE_REDUCER( reduce_container ) kr;

    const Functor f;
    const size_t alloc_bytes;

    kokkos_cilk_reducer (const Functor & f_, const size_t l_alloc_bytes) : f(f_), alloc_bytes(l_alloc_bytes) {
    }

    void initializeView(void * data) {
      ((reduce_container*)data)->initializeContents(alloc_bytes);
      //printf("calling functor init \n");
      f.init(((reduce_container*)data)->getReference());
      //long* trythis = (long*)((reduce_container*)data)->getReference();
      //printf("after init: %d, %d, %d  \n", trythis[0], trythis[1], trythis[2]);
    }

    void cleanupView(void * data) {
      ((reduce_container*)data)->freeMemory();
    }

    void init() {
         initializeView(&REDUCER_VIEW(kr));
         CILK_C_REGISTER_REDUCER(kr);
    }

    void join(typename Functor::value_type val_) {
        reduce_container * cont = &REDUCER_VIEW(kr);
        ValueJoin::join( f, cont->getReference(), val_ );
    }

    void reduce(reduce_container* left, reduce_container* right ) {

      ValueJoin::join( f, left->getReference(), right->getReference() );

    }

    void update_value(typename Functor::value_type ret) {
       reduce_container * cont = &REDUCER_VIEW(kr);
       cont->getValue(ret);
       //long* trythis = (long*)(ret);
       //printf("update value: %d, %d, %d  \n", trythis[0], trythis[1], trythis[2]);
    }

    void release_resources()
    {
//       REDUCER_VIEW(kr).freeMemory();
       CILK_C_UNREGISTER_REDUCER(kr);
    }

};

template <typename ReducerType, class Functor, class WorkTagFwd >
struct kokkos_cilk_reducer< ReducerType, Functor, WorkTagFwd , typename std::enable_if< !Kokkos::is_reducer_type<ReducerType>::value &&
                                                                           !Kokkos::is_view<ReducerType>::value &&
                                           ( !std::is_array< typename Functor::value_type >::value &&
                                             !std::is_pointer< typename Functor::value_type >::value ) >::type >   {

    typedef CilkReduceContainer< ReducerType, WorkTagFwd > reduce_container;
    typedef Kokkos::Impl::FunctorValueJoin< Functor, WorkTagFwd >   ValueJoin;
    CILK_C_DECLARE_REDUCER( reduce_container ) kr;
    /*
     *   {hyperObject, ReducerType}
     */

    const Functor f;
    const size_t alloc_bytes;


    kokkos_cilk_reducer (const Functor & f_, const size_t l_alloc_bytes) : f(f_), alloc_bytes(l_alloc_bytes) {
    }

    void resetView(void* data) {
    }

    void initializeView(void * data) {
      ((reduce_container*)data)->initializeContents(alloc_bytes);
    }

    void cleanupView(void * data) {
      ((reduce_container*)data)->freeMemory();
    }

    void init() {
         initializeView(&REDUCER_VIEW(kr));
         CILK_C_REGISTER_REDUCER(kr);
    }

   void join(typename Functor::value_type & val_) {
        reduce_container * cont = &REDUCER_VIEW(kr);
        cont->join( f, val_ );
    }

    void reduce(reduce_container* left, reduce_container* right ) {
      typename Functor::value_type val_;
      right->getValue(val_);
      left->join( f, val_ );

    }
    
    void update_value(typename Functor::value_type & ret) {
        reduce_container * cont = &REDUCER_VIEW(kr);
        cont->getValue(ret);
    }

    void release_resources()
    {
//       REDUCER_VIEW(kr).freeMemory();
       CILK_C_UNREGISTER_REDUCER(kr);
    }

};

} 
}

#define INITIALIZE_CILK_REDUCER( wrapper, reducer ) \
      reducer.kr.__cilkrts_hyperbase.__c_monoid.reduce_fn = &value_reduce<wrapper>; \
      reducer.kr.__cilkrts_hyperbase.__c_monoid.identity_fn = &value_init<wrapper>; \
      reducer.kr.__cilkrts_hyperbase.__c_monoid.destroy_fn = &value_dealloc<wrapper>; \
      reducer.kr.__cilkrts_hyperbase.__c_monoid.allocate_fn = __cilkrts_hyperobject_alloc; \
      reducer.kr.__cilkrts_hyperbase.__c_monoid.deallocate_fn = __cilkrts_hyperobject_dealloc; \
      reducer.kr.__cilkrts_hyperbase.__flags = 0; \
      reducer.kr.__cilkrts_hyperbase.__view_offset = __CILKRTS_CACHE_LINE__; \
      reducer.kr.__cilkrts_hyperbase.__view_size = sizeof(typename wrapper::reduce_container); \
      reducer.init();


#endif
