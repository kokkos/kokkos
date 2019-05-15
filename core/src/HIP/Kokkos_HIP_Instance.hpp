/*--------------------------------------------------------------------------*/

#ifndef KOKKOS_HIP_INSTANCE_HPP_
#define KOKKOS_HIP_INSTANCE_HPP_
#include<Kokkos_HIP.hpp>

namespace Kokkos {
namespace Experimental {
namespace Impl {


//----------------------------------------------------------------------------

class HIPInternal {
private:

  HIPInternal( const HIPInternal & );
  HIPInternal & operator = ( const HIPInternal & );


public:

  typedef Kokkos::Experimental::HIP::size_type size_type ;

  int         m_hipDev ;
  int         m_hipArch ;
  unsigned    m_multiProcCount ;
  unsigned    m_maxWorkgroup ;
  unsigned    m_maxSharedWords ;
  size_type   m_scratchSpaceCount ;
  size_type   m_scratchFlagsCount ;
  size_type * m_scratchSpace ;
  size_type * m_scratchFlags ;

  hipStream_t m_stream;

  static int was_finalized;

  static HIPInternal & singleton();

  int verify_is_initialized( const char * const label ) const ;

  int is_initialized() const
    { return m_hipDev>=0; }//0 != m_scratchSpace && 0 != m_scratchFlags ; }

  void initialize( int hip_device_id );
  void finalize();

  void print_configuration( std::ostream & ) const ;


  ~HIPInternal();

  HIPInternal()
    : m_hipDev( -1 )
    , m_hipArch( -1 )
    , m_multiProcCount( 0 )
    , m_maxWorkgroup( 0 )
    , m_maxSharedWords( 0 )
    , m_scratchSpaceCount( 0 )
    , m_scratchFlagsCount( 0 )
    , m_scratchSpace( 0 )
    , m_scratchFlags( 0 )
    {}

  size_type * scratch_space( const size_type size );
  size_type * scratch_flags( const size_type size );
};

}
}
}

#endif
