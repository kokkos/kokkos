/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
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

#ifndef KOKKOS_RESCUDASPACE_HPP
#define KOKKOS_RESCUDASPACE_HPP

#include <Kokkos_Macros.hpp>
#if defined( KOKKOS_ENABLE_CUDA )

#include <Kokkos_CudaSpace.hpp>
#include <cmath>
#include <map>

/*--------------------------------------------------------------------------*/

namespace Kokkos {

namespace Experimental {

enum : int {
  ScalarInt,
  ScalarLong,
  ScalarLongLong,
  ScalarFloat,
  ScalarDouble
};

template<class Type, class T = void>
struct get_type_enum;

template<class Type>
struct get_type_enum<Type, typename std::enable_if< std::is_same< Type, int>::value, void >::type >{
   enum { type_id = ScalarInt };
};

template<class Type>
struct get_type_enum<Type, typename std::enable_if< std::is_same< Type, long>::value, void >::type > {
   enum { type_id = ScalarLong };
};

template<class Type>
struct get_type_enum<Type, typename std::enable_if< std::is_same< Type, long long>::value, void >::type > {
   enum { type_id = ScalarLongLong };
};

template<class Type>
struct get_type_enum<Type, typename std::enable_if< std::is_same< Type, float>::value, void >::type > {
   enum { type_id = ScalarFloat };
};

template<class Type>
struct get_type_enum<Type, typename std::enable_if< std::is_same< Type, double>::value, void >::type > {
   enum { type_id = ScalarDouble };
};

template<class Type, class Enabled = void>
struct MergeFunctor;

template<class Type>
struct MergeFunctor<Type, typename std::enable_if< std::is_same< Type, float >::value ||
                                          std::is_same< Type, double >::value, void >::type > {
  
   KOKKOS_INLINE_FUNCTION
   MergeFunctor() {}

   KOKKOS_INLINE_FUNCTION
   MergeFunctor(const MergeFunctor & mf) {}

   KOKKOS_INLINE_FUNCTION
   bool compare( Type a, Type b ) const {
      return (abs(a-b)<0.00000001);
   }
};

template<class Type>
struct MergeFunctor<Type, typename std::enable_if< !std::is_same< Type, float >::value &&
                                          !std::is_same< Type, double >::value, void >::type > {

   KOKKOS_INLINE_FUNCTION
   MergeFunctor() {}

   KOKKOS_INLINE_FUNCTION
   MergeFunctor(const MergeFunctor & mf) {}

   KOKKOS_INLINE_FUNCTION
   bool compare( Type a, Type b ) const {
      return (a == b);
   }
};

struct DupLstType {

   Kokkos::Impl::SharedAllocationRecord<void,void> * rec;
   DupLstType * next;

   KOKKOS_INLINE_FUNCTION
   DupLstType () : next (nullptr), rec (nullptr) {}

};

class DuplicateTracker {
public:
   Kokkos::Impl::SharedAllocationRecord<void,void> * original;
   DupLstType * dup_list;
   int data_type;

   KOKKOS_INLINE_FUNCTION
   DuplicateTracker() : original(nullptr), dup_list(new DupLstType()) { data_type = get_type_enum<int>::type_id; }
   
   KOKKOS_INLINE_FUNCTION
   DuplicateTracker(int sType) : original(nullptr), dup_list(new DupLstType()) {data_type = sType;}
   
   KOKKOS_INLINE_FUNCTION
   DuplicateTracker(const DuplicateTracker & dt) : original( dt.original ), dup_list(dt.dup_list) {data_type = dt.data_type;}

   inline 
   void add_dup( Kokkos::Impl::SharedAllocationRecord<void,void>* dup ) {
      DupLstType * pWork = dup_list;
      while (pWork != nullptr) {
         if (pWork->rec == nullptr) {
            pWork->rec = dup;
            break;
         } else if (pWork->next == nullptr) {
            pWork->next = new DupLstType();
            pWork->next->rec = dup;
            break;
         } else {
            pWork = pWork->next;
         }
      }
   }
};

template<class Type, class ExecutionSpace>
class SpecDuplicateTracker : public DuplicateTracker  {
public:
   typedef typename std::remove_reference<Type>::type nr_type;
   typedef typename std::remove_pointer<nr_type>::type np_type;
   typedef typename std::remove_extent<np_type>::type ne_type;
   typedef typename std::remove_const<ne_type>::type rd_type;
   typedef MergeFunctor<rd_type> functor_type;

   functor_type cf;

   KOKKOS_INLINE_FUNCTION
   SpecDuplicateTracker() : DuplicateTracker( get_type_enum<rd_type>::type_id ) { 
   }
   KOKKOS_INLINE_FUNCTION
   SpecDuplicateTracker(const SpecDuplicateTracker & rhs) : DuplicateTracker( rhs ), cf(rhs.cf) { 
   }
   
   void combine_dups(); 

   KOKKOS_INLINE_FUNCTION
   void operator ()(const int i) const {
      rd_type * ptr = (Type*)original->data();      
      DupLstType * dupOuter = dup_list;
      for (int j = 0; j < 3 && dupOuter != nullptr; j++) {
         ptr[i]  =  ((rd_type*)dupOuter->rec->data())[i];
         DupLstType * dupInner = dupOuter->next;
         for ( int k = 0; k < 2 && dupInner != nullptr; k++) {
            rd_type * dup = (rd_type*)dupInner->rec->data();
            if ( cf.compare( dup[i], ptr[i] ) )  // just need 2 that are the same
               return;
            dupInner = dupInner->next;
         }
         dupOuter = dupOuter->next;
      }
   }

};

} // Experimental


/** \brief  Cuda on-device memory management */

class ResCudaSpace : public CudaSpace {
public:
  //! Tag this class as a kokkos memory space
  typedef ResCudaSpace             memory_space ;
  typedef ResCudaSpace          resilient_space ;
  typedef Kokkos::Cuda          execution_space ;
  typedef Kokkos::Device<execution_space,memory_space> device_type;

  typedef unsigned int          size_type ;

  /*--------------------------------*/

  ResCudaSpace();
  ResCudaSpace( ResCudaSpace && rhs ) = default ;
  ResCudaSpace( const ResCudaSpace & rhs ) = default ;
  ResCudaSpace & operator = ( ResCudaSpace && rhs ) = default ;
  ResCudaSpace & operator = ( const ResCudaSpace & rhs ) = default ;
  ~ResCudaSpace() = default ;

  template< class Type >
  static void track_duplicate( Kokkos::Impl::SharedAllocationRecord<void,void> * orig, Kokkos::Impl::SharedAllocationRecord<void,void> * dup );
  
  static void combine_duplicates();
  static void clear_duplicates_list();

  static std::map<std::string, Kokkos::Experimental::DuplicateTracker * > duplicate_map;

private:

  friend class Kokkos::Impl::SharedAllocationRecord< Kokkos::ResCudaSpace , void > ;
};

} // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

static_assert( Kokkos::Impl::MemorySpaceAccess< Kokkos::ResCudaSpace , Kokkos::ResCudaSpace >::assignable , "" );

//----------------------------------------

template<>
struct MemorySpaceAccess< Kokkos::HostSpace , Kokkos::ResCudaSpace > {
  enum { assignable = false };
  enum { accessible = false };
  enum { deepcopy   = true };
};

//----------------------------------------

template<>
struct MemorySpaceAccess< Kokkos::ResCudaSpace , Kokkos::HostSpace > {
  enum { assignable = false };
  enum { accessible = false };
  enum { deepcopy   = true };
};

template<>
struct MemorySpaceAccess< Kokkos::CudaSpace , Kokkos::ResCudaSpace > {
  enum { assignable = true };
  enum { accessible = true };
  enum { deepcopy   = true };
};

//----------------------------------------

template<>
struct MemorySpaceAccess< Kokkos::ResCudaSpace , Kokkos::CudaSpace > {
  enum { assignable = true };
  enum { accessible = true };
  enum { deepcopy   = true };
};

template<>
struct MemorySpaceAccess< Kokkos::ResCudaSpace , Kokkos::CudaUVMSpace > {
  // CudaSpace::execution_space == CudaUVMSpace::execution_space
  enum { assignable = true };
  enum { accessible = true };
  enum { deepcopy   = true };
};

template<>
struct MemorySpaceAccess< Kokkos::ResCudaSpace , Kokkos::CudaHostPinnedSpace > {
  // CudaSpace::execution_space != CudaHostPinnedSpace::execution_space
  enum { assignable = false };
  enum { accessible = true }; // ResCudaSpace::execution_space
  enum { deepcopy   = true };
};

//----------------------------------------
// CudaUVMSpace::execution_space == Cuda
// CudaUVMSpace accessible to both Cuda and Host

template<>
struct MemorySpaceAccess< Kokkos::CudaUVMSpace , Kokkos::ResCudaSpace > {
  // CudaUVMSpace::execution_space == CudaSpace::execution_space
  // Can access CudaUVMSpace from Host but cannot access ResCudaSpace from Host
  enum { assignable = false };

  // CudaUVMSpace::execution_space can access CudaSpace
  enum { accessible = true };
  enum { deepcopy   = true };
};

template<>
struct MemorySpaceAccess< Kokkos::CudaHostPinnedSpace , Kokkos::ResCudaSpace > {
  enum { assignable = false }; // Cannot access from Host
  enum { accessible = false };
  enum { deepcopy   = true };
};

//----------------------------------------

}} // namespace Kokkos::Impl

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

template<> struct DeepCopy< ResCudaSpace , ResCudaSpace , Cuda>
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const Cuda & , void * dst , const void * src , size_t );
};

template<> struct DeepCopy< ResCudaSpace , HostSpace , Cuda >
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const Cuda & , void * dst , const void * src , size_t );
};

template<> struct DeepCopy< HostSpace , ResCudaSpace , Cuda >
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const Cuda & , void * dst , const void * src , size_t );
};

template<class ExecutionSpace> struct DeepCopy< ResCudaSpace , ResCudaSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< ResCudaSpace , ResCudaSpace , Cuda >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncCuda (dst,src,n);
  }
};

template<class ExecutionSpace> struct DeepCopy< ResCudaSpace , HostSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< ResCudaSpace , HostSpace , Cuda>( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncCuda (dst,src,n);
  }
};

template<class ExecutionSpace>
struct DeepCopy< HostSpace , ResCudaSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< HostSpace , ResCudaSpace , Cuda >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncCuda (dst,src,n);
  }
};

template<class ExecutionSpace>
struct DeepCopy< ResCudaSpace , CudaSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< CudaSpace , CudaSpace , Cuda >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncCuda (dst,src,n);
  }
};

template<class ExecutionSpace>
struct DeepCopy< CudaSpace , ResCudaSpace , ExecutionSpace>
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< CudaSpace , CudaSpace , Cuda >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncCuda (dst,src,n);
  }
};

} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

/** Running in ResCudaSpace attempting to access HostSpace: error */
template<>
struct VerifyExecutionCanAccessMemorySpace< Kokkos::ResCudaSpace , Kokkos::HostSpace >
{
  enum { value = false };
  KOKKOS_INLINE_FUNCTION static void verify( void )
    { Kokkos::abort("Cuda code attempted to access HostSpace memory"); }

  KOKKOS_INLINE_FUNCTION static void verify( const void * )
    { Kokkos::abort("Cuda code attempted to access HostSpace memory"); }
};

/** Running in ResCudaSpace accessing CudaUVMSpace: ok */
template<>
struct VerifyExecutionCanAccessMemorySpace< Kokkos::ResCudaSpace , Kokkos::CudaUVMSpace >
{
  enum { value = true };
  KOKKOS_INLINE_FUNCTION static void verify( void ) { }
  KOKKOS_INLINE_FUNCTION static void verify( const void * ) { }
};

/** Running in CudaSpace accessing CudaHostPinnedSpace: ok */
template<>
struct VerifyExecutionCanAccessMemorySpace< Kokkos::ResCudaSpace , Kokkos::CudaHostPinnedSpace >
{
  enum { value = true };
  KOKKOS_INLINE_FUNCTION static void verify( void ) { }
  KOKKOS_INLINE_FUNCTION static void verify( const void * ) { }
};

/** Running in CudaSpace attempting to access an unknown space: error */
template< class OtherSpace >
struct VerifyExecutionCanAccessMemorySpace<
  typename enable_if< ! is_same<Kokkos::ResCudaSpace,OtherSpace>::value , Kokkos::ResCudaSpace >::type ,
  OtherSpace >
{
  enum { value = false };
  KOKKOS_INLINE_FUNCTION static void verify( void )
    { Kokkos::abort("Cuda code attempted to access unknown Space memory"); }

  KOKKOS_INLINE_FUNCTION static void verify( const void * )
    { Kokkos::abort("Cuda code attempted to access unknown Space memory"); }
};

//----------------------------------------------------------------------------
/** Running in HostSpace attempting to access CudaSpace */
template<>
struct VerifyExecutionCanAccessMemorySpace< Kokkos::HostSpace , Kokkos::ResCudaSpace >
{
  enum { value = false };
  inline static void verify( void ) { ResCudaSpace::access_error(); }
  inline static void verify( const void * p ) { ResCudaSpace::access_error(p); }
};

} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template<>
class SharedAllocationRecord< Kokkos::ResCudaSpace , void >
  : public SharedAllocationRecord< void , void >
{
private:

  friend class SharedAllocationRecord< Kokkos::CudaUVMSpace , void > ;

  typedef SharedAllocationRecord< void , void >  RecordBase ;

  SharedAllocationRecord( const SharedAllocationRecord & ) = delete ;
  SharedAllocationRecord & operator = ( const SharedAllocationRecord & ) = delete ;

  static void deallocate( RecordBase * );

  static ::cudaTextureObject_t
  attach_texture_object( const unsigned sizeof_alias
                       , void * const   alloc_ptr
                       , const size_t   alloc_size );

#ifdef KOKKOS_DEBUG
  static RecordBase s_root_record ;
#endif

  ::cudaTextureObject_t   m_tex_obj ;
  const Kokkos::ResCudaSpace m_space ;

protected:

  ~SharedAllocationRecord();
  SharedAllocationRecord() : RecordBase(), m_tex_obj(0), m_space() {}

  SharedAllocationRecord( const Kokkos::ResCudaSpace        & arg_space
                        , const std::string              & arg_label
                        , const size_t                     arg_alloc_size
                        , const RecordBase::function_type  arg_dealloc = & deallocate
                        );

public:
  Kokkos::ResCudaSpace get_space() const { return m_space; }
  std::string get_label() const ;

  static SharedAllocationRecord * allocate( const Kokkos::ResCudaSpace &  arg_space
                                          , const std::string       &  arg_label
                                          , const size_t               arg_alloc_size );

  /**\brief  Allocate tracked memory in the space */
  static
  void * allocate_tracked( const Kokkos::ResCudaSpace & arg_space
                         , const std::string & arg_label
                         , const size_t arg_alloc_size );

  /**\brief  Reallocate tracked memory in the space */
  static
  void * reallocate_tracked( void * const arg_alloc_ptr
                           , const size_t arg_alloc_size );

  /**\brief  Deallocate tracked memory in the space */
  static
  void deallocate_tracked( void * const arg_alloc_ptr );

  static SharedAllocationRecord * get_record( void * arg_alloc_ptr );

  template< typename AliasType >
  inline
  ::cudaTextureObject_t attach_texture_object()
    {
      static_assert( ( std::is_same< AliasType , int >::value ||
                       std::is_same< AliasType , ::int2 >::value ||
                       std::is_same< AliasType , ::int4 >::value )
                   , "Cuda texture fetch only supported for alias types of int, ::int2, or ::int4" );

      if ( m_tex_obj == 0 ) {
        m_tex_obj = attach_texture_object( sizeof(AliasType)
                                         , (void*) RecordBase::m_alloc_ptr
                                         , RecordBase::m_alloc_size );
      }

      return m_tex_obj ;
    }

  template< typename AliasType >
  inline
  int attach_texture_object_offset( const AliasType * const ptr )
    {
      // Texture object is attached to the entire allocation range
      return ptr - reinterpret_cast<AliasType*>( RecordBase::m_alloc_ptr );
    }

  static void print_records( std::ostream & , const Kokkos::ResCudaSpace & , bool detail = false );
};



} // namespace Impl


template< class Type >
void ResCudaSpace::track_duplicate( Kokkos::Impl::SharedAllocationRecord<void,void> * orig, Kokkos::Impl::SharedAllocationRecord<void,void> * dup ) {
     Kokkos::Impl::SharedAllocationRecord<ResCudaSpace,void> * SP = (Kokkos::Impl::SharedAllocationRecord<ResCudaSpace,void> *)dup;
     typedef Kokkos::Experimental::SpecDuplicateTracker<Type, Kokkos::ResCuda> dt_type;
     dt_type * dt = nullptr;
     auto loc = duplicate_map.find(SP->get_label());
     if ( loc != duplicate_map.end() ) {
        dt_type* dt = (dt_type*)loc->second;
     } else {
        dt_type* dt = new dt_type();
        dt->original = orig;
        duplicate_map[SP->get_label()] = (Kokkos::Experimental::DuplicateTracker*)dt;
     }
     dt->add_dup(dup);
  }

} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #if defined( KOKKOS_ENABLE_CUDA ) */
#endif /* #define KOKKOS_RESCUDASPACE_HPP */

