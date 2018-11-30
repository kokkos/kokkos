//----------------------------------------------------------------------------
namespace Kokkos {
namespace Impl {

class CudaInternal {
private:

  CudaInternal( const CudaInternal & );
  CudaInternal & operator = ( const CudaInternal & );


public:

  typedef Cuda::size_type size_type ;

  int         m_cudaDev ;
  int         m_cudaArch ;
  unsigned    m_multiProcCount ;
  unsigned    m_maxWarpCount ;
  unsigned    m_maxBlock ;
  unsigned    m_maxSharedWords ;
  uint32_t    m_maxConcurrency ;
  size_type   m_scratchSpaceCount ;
  size_type   m_scratchFlagsCount ;
  size_type   m_scratchUnifiedCount ;
  size_type   m_scratchUnifiedSupported ;
  size_type   m_streamCount ;
  size_type * m_scratchSpace ;
  size_type * m_scratchFlags ;
  size_type * m_scratchUnified ;
  uint32_t  * m_scratchConcurrentBitset ;
  cudaStream_t m_stream ;

  static int was_initialized;
  static int was_finalized;

  static CudaInternal & singleton();

  int verify_is_initialized( const char * const label ) const ;

  int is_initialized() const
    { return 0 != m_scratchSpace && 0 != m_scratchFlags ; }

  void initialize( int cuda_device_id , int stream_count );
  void finalize();

  void print_configuration( std::ostream & ) const ;

  ~CudaInternal();

  CudaInternal()
    : m_cudaDev( -1 )
    , m_cudaArch( -1 )
    , m_multiProcCount( 0 )
    , m_maxWarpCount( 0 )
    , m_maxBlock( 0 )
    , m_maxSharedWords( 0 )
    , m_maxConcurrency( 0 )
    , m_scratchSpaceCount( 0 )
    , m_scratchFlagsCount( 0 )
    , m_scratchUnifiedCount( 0 )
    , m_scratchUnifiedSupported( 0 )
    , m_streamCount( 0 )
    , m_scratchSpace( 0 )
    , m_scratchFlags( 0 )
    , m_scratchUnified( 0 )
    , m_scratchConcurrentBitset( 0 )
    , m_stream( 0 )
    {}

  size_type * scratch_space( const size_type size );
  size_type * scratch_flags( const size_type size );
  size_type * scratch_unified( const size_type size );
};

} // Namespace Impl
} // Namespace Kokkos
