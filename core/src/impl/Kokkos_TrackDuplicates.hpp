
#ifndef __DUPLICATE_TRACKER__
#define __DUPLICATE_TRACKER__

#include <Kokkos_Macros.hpp>
#include <cmath>
#include <map>
#include <typeinfo>

namespace Kokkos {

namespace Experimental {

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

class DuplicateTracker {
public:
   void * original_data;
   int dup_cnt;
   int data_len;
   void * dup_list[3];
   void * func_ptr;

   static std::map<std::string, void*> kernel_func_list;
   static void add_kernel_func ( std::string name, void * func_ptr );
   static void * get_kernel_func ( std::string name );

   inline virtual ~DuplicateTracker() {}

   inline 
   DuplicateTracker() : original_data(nullptr) { 
      dup_cnt = 0;
      data_len = 0;
      for (int i = 0; i < 3; i++) { dup_list[i] = nullptr; }
   }
   
   inline
   DuplicateTracker(const DuplicateTracker & dt) : original_data( dt.original_data ) {
      dup_cnt = dt.dup_cnt;
      data_len = dt.data_len;
      for (int i = 0; i < dup_cnt; i++) { dup_list[i] = dt.dup_list[i]; }
   }

   inline 
   void add_dup( Kokkos::Impl::SharedAllocationRecord<void,void>* dup ) {
      if (dup_cnt < 3) {
         dup_list[dup_cnt] = (void*)dup->data();
         dup_cnt++;
         printf("duplicate added to list: %d\n", dup_cnt);
      }
   }

   inline 
   virtual void combine_dups() {
   }
};

template<class DType, class ExecSpace> 
class CombineFunctor {
public:
   typedef MergeFunctor<DType> functor_type;
   functor_type cf;
 
   DType* orig_view;
   DType* dup_view[3];
   size_t m_len;

   static void * s_dup_kernel;

   inline  
   CombineFunctor() : orig_view(nullptr), dup_view{}, m_len (0) {}

   inline void load_ptrs( DType * orig, DType * d1, DType * d2, DType * d3, size_t len) {
       orig_view = orig;
       dup_view[0] = d1;
       dup_view[1] = d2;
       dup_view[2] = d3;
       m_len = len;
   }

   KOKKOS_INLINE_FUNCTION
   int get_len() const {
      return (int)m_len;
   }

   KOKKOS_INLINE_FUNCTION 
   CombineFunctor( const CombineFunctor & rhs ) : orig_view(rhs.orig_view), dup_view{} {
      for (int i = 0; i < 3; i++)
          dup_view[i] = rhs.dup_view[i];
      m_len = rhs.m_len;
   }
   
   KOKKOS_INLINE_FUNCTION
   //void operator ()(const int i) const {
   void exec(const int i) const {
      printf("combine dups: %d\n", i);
      for (int j = 0; j < 3; j++) {
         printf("iterating outer: %d - %d \n", i, j);
         orig_view[i]  =  dup_view[j][i];
         printf("first entry: %d, %d\n",j, orig_view[i]);
         int k = j < 2 ? j+1 : 0;
         for ( int r = 0; r < 2 ; r++) {
            printf("iterate inner %d, %d, %d \n", i, j, k);
            if ( cf.compare( dup_view[k][i], orig_view[i] ) )  // just need 2 that are the same
            {
               printf("match found: %d - %d\n", i, j);
               return;
            }
            k = k < 2 ? k+1 : 0;
         }
      }
      printf("no match found: %i\n", i);
   }

};

template<class Type, class ExecutionSpace>
class SpecDuplicateTracker : public DuplicateTracker  {
public:
   typedef typename std::remove_reference<Type>::type nr_type;
   typedef typename std::remove_pointer<nr_type>::type np_type;
   typedef typename std::remove_extent<np_type>::type ne_type;
   typedef typename std::remove_const<ne_type>::type rd_type;
   typedef CombineFunctor<rd_type, ExecutionSpace> comb_type;

   comb_type m_cf;

   inline 
   SpecDuplicateTracker() : DuplicateTracker( ), m_cf() { 
   }

   inline 
   SpecDuplicateTracker(const SpecDuplicateTracker & rhs) : DuplicateTracker( rhs ), m_cf(rhs.m_cf)  { 
   }
   
   virtual void combine_dups();

};

  template< class Type, class MemorySpace >
  static void track_duplicate ( Kokkos::Impl::SharedAllocationRecord<void,void> * orig, Kokkos::Impl::SharedAllocationRecord<void,void> * dup ) {
    Kokkos::Impl::SharedAllocationRecord<MemorySpace,void> * SP = static_cast<Kokkos::Impl::SharedAllocationRecord<MemorySpace,void> *>(dup);
    typedef Kokkos::Experimental::SpecDuplicateTracker<Type, typename MemorySpace::execution_space> dt_type;
    //typedef CombineFunctor<typename dt_type::rd_type, typename MemorySpace::execution_space> comb_type;

    dt_type * dt = nullptr;
    auto loc = MemorySpace::duplicate_map.find(SP->get_label());
    if ( loc != MemorySpace::duplicate_map.end() ) {
       dt = (dt_type*)loc->second;
       printf("retrieved existing tracking entry from map: %s\n", SP->get_label().c_str());
    } else {
       dt = new dt_type();
       //printf("dup_kernel ptr = %08x \n", comb_type::s_dup_kernel); 
       //dt->func_ptr = (void*)pLaunch; // comb_type::s_dup_kernel;
       dt->func_ptr = DuplicateTracker::get_kernel_func( typeid(Type).name() );
       dt->data_len = orig->size();
       dt->original_data = orig->data();
       MemorySpace::duplicate_map[SP->get_label()] = static_cast<Kokkos::Experimental::DuplicateTracker*>(dt);
       printf("creating new tracking entry in hash map: %s, %08x \n", SP->get_label().c_str(), (unsigned long)dt->func_ptr);
    }
    dt->add_dup(dup);
 }


} // Experimental

} // Kokkos

#endif
