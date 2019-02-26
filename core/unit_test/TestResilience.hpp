
#include <Kokkos_Core.hpp>

namespace Test {

struct ResSurrogate {
   typedef Kokkos::View< int*, Kokkos::CudaSpace > ViewType;
//   typedef Kokkos::View< int*, Kokkos::HostSpace > ViewType;
   ViewType vt;

   KOKKOS_INLINE_FUNCTION
   ResSurrogate(const ViewType & vt_) : vt(vt_) {
      printf("inside functor constructor\n");	
   }

   KOKKOS_INLINE_FUNCTION
   void operator()(const int i) const {
      vt(i) = i;
   } 

};

struct testCopy {

   testCopy() {
      printf("test copy original\n");
   }

   testCopy(const testCopy &) {
      printf("test copy const\n");
   }

   void callme() const {
   }

};



template< class ExecSpace, class ScheduleType >
struct TestResilientRange {
  typedef int value_type; ///< typedef required for the parallel_reduce

  typedef Kokkos::View< int*, Kokkos::ResCudaSpace > view_type;
//  typedef Kokkos::View< int*, Kokkos::HostSpace > test_type;
//  typedef Kokkos::View< int*, Kokkos::HostSpace > view_type;

  int N;

  TestResilientRange( const size_t N_ )
    :  N(N_)
    {}

  void test_for()
  {
//     view_type m_data ( Kokkos::ViewAllocateWithoutInitializing( "data" ), N );
     view_type m_data ( "data", N );
//     test_type t_data ( "test", N );
     typename view_type::HostMirror v = Kokkos::create_mirror_view(m_data); 
//      for (int i = 0; i < N; i++) {
//         v(i) = i;
//      }
//      Kokkos::deep_copy( m_data, v );

//      ResSurrogate f(m_data);
      printf("calling parallel_for\n");
//      Kokkos::parallel_for(N, f);
      Kokkos::RangePolicy<Kokkos::ResCuda> rp (0,N);
/*      auto ml = KOKKOS_LAMBDA(const int i){ 
   #if defined(__CUDA_ARCH__)
         printf("insided lambda[%d]\n", i);
   #endif
         m_data(i)=i;
      };
*/
      Kokkos::parallel_for(rp,KOKKOS_LAMBDA(const int i){ 
         m_data(i)=i;
      });
      Kokkos::fence();
        
      Kokkos::deep_copy(v, m_data);

      for (int i = 0; i < N; i++) {
         ASSERT_EQ(v(i), i );
      }
  }

};

TEST_F( TEST_CATEGORY, range_for_resilience )
{
  { TestResilientRange< TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Static> >f(10); f.test_for(); }
}

} //namespace Test
