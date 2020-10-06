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
 public:
  typedef Kokkos::Experimental::SYCL::size_type size_type;

  SYCLInternal();
  ~SYCLInternal();

  SYCLInternal(const SYCLInternal&) = delete;
  SYCLInternal& operator=(const SYCLInternal&) = delete;
  SYCLInternal& operator=(SYCLInternal&&) = delete;
  SYCLInternal(SYCLInternal&&)            = delete;

  int m_syclDev                 = -1;
  int m_syclArch                = -1;
  unsigned m_multiProcCount     = 0;
  unsigned m_maxWorkgroup       = 0;
  unsigned m_maxSharedWords     = 0;
  size_type m_scratchSpaceCount = 0;
  size_type m_scratchFlagsCount = 0;
  size_type* m_scratchSpace     = 0;
  size_type* m_scratchFlags     = 0;

  std::unique_ptr<cl::sycl::queue> m_queue;

  // An indirect kernel is one where the functor to be executed is explicitly
  // created in USM shared memory before being executed, to get around the
  // trivially copyable limitation of SYCL.
  //
  // m_indirectKernel just manages the memory as a reuseable buffer.  It is
  // stored in an optional because the allocator contains a queue
  using IndirectKernelAllocator =
      sycl::usm_allocator<std::byte, sycl::usm::alloc::shared>;
  using IndirectKernelMemory =
      std::vector<IndirectKernelAllocator::value_type, IndirectKernelAllocator>;
  using IndirectKernel = std::optional<IndirectKernelMemory>;
  IndirectKernel m_indirectKernel;

  static int was_finalized;

  static SYCLInternal& singleton();

  int verify_is_initialized(const char* const label) const;

  int is_initialized() const {
    return m_queue != nullptr;
  }

  void initialize(const cl::sycl::device& d);
  void initialize(const cl::sycl::device_selector& s);
  void initialize(int sycl_device_id);
  void initialize();
  void finalize();

  void print_configuration(std::ostream&) const;
  void print_configuration(std::ostream&, const bool) const;

  void listDevices(std::ostream&) const;
  void listDevices() const;

  size_type* scratch_space(const size_type size);
  size_type* scratch_flags(const size_type size);
};

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#endif

