

#include <Kokkos_Core.hpp>
#include <map>

namespace Test {

class theBaseClass {

public:
   void * data;
   theBaseClass(void * d_ ) : data(d_)  {}

   virtual void test_data() = 0;
};

template< class ExecSpace, class T >
class derivedClass : public theBaseClass {

public:
   int N;
   typedef Kokkos::View<T*, ExecSpace, Kokkos::MemoryUnmanaged> um_view_type;

   inline
   derivedClass() : theBaseClass(nullptr) {}
   
   inline
   derivedClass(int n_, T* d_) : theBaseClass((void*)d_), N(n_) {}

   KOKKOS_INLINE_FUNCTION
   derivedClass(const derivedClass & dc_) : theBaseClass(dc_.data), N(dc_.N) {}

   KOKKOS_INLINE_FUNCTION
   derivedClass(derivedClass && dc_) : theBaseClass(std::move(dc_.data)), N(std::move(dc_.N)) {}

   KOKKOS_INLINE_FUNCTION
   derivedClass * operator = (const derivedClass & dc_) {
      data = dc_.data;
      N = dc_.N;
      return *this;
   }

   virtual void test_data() {
      um_view_type uv((T*)data, N);

      Kokkos::parallel_for( N, KOKKOS_LAMBDA (const int i) {
         uv(i) = i;
      });

   }
   
};

template< class ExecSpace >
struct TestVirtualClass {

   typedef Kokkos::View<int*, typename ExecSpace::memory_space > view_type;
   typedef typename view_type::HostMirror host_view_type;
   int N;
   TestVirtualClass(int n_) : N(n_) {}

   void test_for() {
      view_type v("root",N);
      host_view_type hv = Kokkos::create_mirror(v);
      
      std::map< std::string,theBaseClass*> mp; 
      theBaseClass * tbc = new derivedClass<ExecSpace, int>(v.extent(0), v.data());
      mp[v.label()] = tbc;
  
      {
         theBaseClass * tbt = mp[(std::string)"root"];
         tbc->test_data();

         Kokkos::deep_copy( hv, v );
         for (int i = 0; i < N; i++) {
            printf("result: %d, %d \n", i, hv(i) );
         }
      }
   }
};

TEST_F( TEST_CATEGORY, test_virtual )
{
  { TestVirtualClass< TEST_EXECSPACE >f(10); f.test_for(); }
}

} //namespace Test
