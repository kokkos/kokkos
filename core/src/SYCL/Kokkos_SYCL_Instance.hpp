/*--------------------------------------------------------------------------*/

#ifndef KOKKOS_SYCL_INSTANCE_HPP_
#define KOKKOS_SYCL_INSTANCE_HPP_
#include <Kokkos_SYCL.hpp>
#include <CL/sycl.hpp>
#include <iosfwd>
namespace Kokkos {
namespace Experimental {
namespace Impl {

//----------------------------------------------------------------------------

class SYCLInternal {
 private:
  SYCLInternal(const SYCLInternal&);
  SYCLInternal& operator=(const SYCLInternal&);

 public:
  typedef Kokkos::Experimental::SYCL::size_type size_type;

  int m_syclDev;
  int m_syclArch;
  unsigned m_multiProcCount;
  unsigned m_maxWorkgroup;
  unsigned m_maxSharedWords;
  size_type m_scratchSpaceCount;
  size_type m_scratchFlagsCount;
  size_type* m_scratchSpace;
  size_type* m_scratchFlags;

  cl::sycl::queue* m_queue;

  static int was_finalized;

  static SYCLInternal& singleton();

  int verify_is_initialized(const char* const label) const;

  int is_initialized() const {
    return m_syclDev >= 0;
  }  // 0 != m_scratchSpace && 0 != m_scratchFlags ; }

  void initialize(int sycl_device_id);
  void initialize();
  void finalize();

  void print_configuration(std::ostream&) const;
  void print_configuration(std::ostream&, const bool) const;

  void listDevices(std::ostream&) const;
  void listDevices() const;

  ~SYCLInternal();

  SYCLInternal()
      : m_syclDev(-1),
        m_syclArch(-1),
        m_multiProcCount(0),
        m_maxWorkgroup(0),
        m_maxSharedWords(0),
        m_scratchSpaceCount(0),
        m_scratchFlagsCount(0),
        m_scratchSpace(0),
        m_scratchFlags(0) {}

  size_type* scratch_space(const size_type size);
  size_type* scratch_flags(const size_type size);
};

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#endif
