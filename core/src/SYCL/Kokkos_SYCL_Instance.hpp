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

  // USMObject is a reusable buffer for a single object
  // in USM memory
  template <sycl::usm::alloc Kind>
  class USMObject {
   public:
    static constexpr sycl::usm::alloc kind = Kind;

    USMObject(USMObject const&) = delete;
    USMObject(USMObject&&)      = delete;
    USMObject& operator=(USMObject&&) = delete;
    USMObject& operator=(USMObject const&) = delete;

    explicit USMObject(sycl::queue q) noexcept : m_q(std::move(q)) {}
    ~USMObject() { sycl::free(m_data, m_q); }

    sycl::queue queue() const noexcept { return m_q; }

    void* data() noexcept { return m_data; }
    const void* data() const noexcept { return m_data; }

    size_t capacity() const noexcept { return m_capacity; }

    // reserve() allocates space for at least n bytes
    // returns the new capacity
    size_t reserve(size_t n) {
      if (m_capacity < n) {
        void* malloced = sycl::malloc(n, m_q, kind);
        if (!malloced) throw std::bad_alloc();
        sycl::free(m_data, m_q);
        m_data     = malloced;
        m_capacity = n;
      }

      return m_capacity;
    }

    // This will memcpy an object T into memory held by this object
    // returns: a T* to that object
    //
    // Note:  it is UB to dereference this pointer with an object that is
    // not an implicit-lifetime nor trivially-copyable type, but presumably much
    // faster because we can use USM device memory
    template <typename T>
    T* memcpy_from(const T& t) {
      reserve(sizeof(T));
      sycl::event memcopied = m_q.memcpy(m_data, std::addressof(t), sizeof(T));
      memcopied.wait();

      return reinterpret_cast<T*>(m_data);
    }

    // This will copy-constuct an object T into memory held by this object
    // returns: a unique_ptr<T, destruct_delete> that will call the
    // destructor on the type when it goes out of scope.
    //
    // Note:  This will not work with USM device memory
    template <typename T>
    std::unique_ptr<T, Kokkos::Impl::destruct_delete> copy_construct_from(
        const T& t) {
      static_assert(kind != sycl::usm::alloc::device,
                    "Cannot copy construct into USM device memory");

      reserve(sizeof(T));
      return std::unique_ptr<T, Kokkos::Impl::destruct_delete>(new (m_data)
                                                                   T(t));
    }

    // Performs either memcpy (for USM device memory) and returns a T*
    // (but is technically UB when dereferenced on an object that is not
    // an implicit-lifetime nor trivially-copyable type
    //
    // or
    //
    // performs copy construction (for other USM memory types) and returns a
    // unique_ptr<T, ...>
    template <typename T>
    auto copy_from(const T& t) {
      if constexpr (sycl::usm::alloc::device == kind)
        return memcpy_from(t);
      else
        return copy_construct_from(t);
    }

   private:
    sycl::queue m_q;
    void* m_data      = nullptr;
    size_t m_capacity = 0;
  };

  // An indirect kernel is one where the functor to be executed is explicitly
  // copied to USM device memory before being executed, to get around the
  // trivially copyable limitation of SYCL.
  using IndirectKernelMemory = USMObject<sycl::usm::alloc::device>;
  using IndirectKernel       = std::optional<IndirectKernelMemory>;
  IndirectKernel m_indirectKernel;

  static int was_finalized;

  static SYCLInternal& singleton();

  int verify_is_initialized(const char* const label) const;

  int is_initialized() const { return m_queue != nullptr; }

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

