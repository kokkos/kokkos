/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
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
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
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

/// \file Kokkos_Serial.hpp
/// \brief Declaration and definition of Kokkos::Serial device.

#ifndef KOKKOS_SERIAL_HPP
#define KOKKOS_SERIAL_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_SERIAL)

#include <cstddef>
#include <iosfwd>
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_TaskScheduler.hpp>
#include <Kokkos_Layout.hpp>
#include <Kokkos_HostSpace.hpp>
#include <Kokkos_ScratchSpace.hpp>
#include <Kokkos_MemoryTraits.hpp>
#include <impl/Kokkos_Tags.hpp>
#include <impl/Kokkos_HostThreadTeam.hpp>
#include <impl/Kokkos_FunctorAnalysis.hpp>
#include <impl/Kokkos_FunctorAdapter.hpp>
#include <impl/Kokkos_Tools.hpp>
#include <impl/Kokkos_ExecSpaceInitializer.hpp>

#include <KokkosExp_MDRangePolicy.hpp>

#include <Kokkos_UniqueToken.hpp>

namespace Kokkos {

/// \class Serial
/// \brief Kokkos device for non-parallel execution
///
/// A "device" represents a parallel execution model.  It tells Kokkos
/// how to parallelize the execution of kernels in a parallel_for or
/// parallel_reduce.  For example, the Threads device uses Pthreads or
/// C++11 threads on a CPU, the OpenMP device uses the OpenMP language
/// extensions, and the Cuda device uses NVIDIA's CUDA programming
/// model.  The Serial device executes "parallel" kernels
/// sequentially.  This is useful if you really do not want to use
/// threads, or if you want to explore different combinations of MPI
/// and shared-memory parallel programming models.
class Serial {
 public:
  //! \name Type declarations that all Kokkos devices must provide.
  //@{

  //! Tag this class as an execution space:
  using execution_space = Serial;
  //! This device's preferred memory space.
  using memory_space = Kokkos::HostSpace;
  //! The size_type alias best suited for this device.
  using size_type = memory_space::size_type;
  //! This execution space preferred device_type
  using device_type = Kokkos::Device<execution_space, memory_space>;

  //! This device's preferred array layout.
  using array_layout = LayoutRight;

  /// \brief  Scratch memory space
  using scratch_memory_space = ScratchMemorySpace<Kokkos::Serial>;

  //@}

  /// \brief True if and only if this method is being called in a
  ///   thread-parallel function.
  ///
  /// For the Serial device, this method <i>always</i> returns false,
  /// because parallel_for or parallel_reduce with the Serial device
  /// always execute sequentially.
  inline static int in_parallel() { return false; }

  /// \brief Wait until all dispatched functors complete.
  ///
  /// The parallel_for or parallel_reduce dispatch of a functor may
  /// return asynchronously, before the functor completes.  This
  /// method does not return until all dispatched functors on this
  /// device have completed.
  static void impl_static_fence() {}

  void fence() const {}

  /** \brief  Return the maximum amount of concurrency.  */
  static int concurrency() { return 1; }

  //! Print configuration information to the given output stream.
  static void print_configuration(std::ostream&,
                                  const bool /* detail */ = false) {}

  static void impl_initialize();

  static bool impl_is_initialized();

  //! Free any resources being consumed by the device.
  static void impl_finalize();

  //--------------------------------------------------------------------------

  inline static int impl_thread_pool_size(int = 0) { return 1; }
  KOKKOS_INLINE_FUNCTION static int impl_thread_pool_rank() { return 0; }

  //--------------------------------------------------------------------------

  KOKKOS_INLINE_FUNCTION static unsigned impl_hardware_thread_id() {
    return impl_thread_pool_rank();
  }
  inline static unsigned impl_max_hardware_threads() {
    return impl_thread_pool_size(0);
  }

  uint32_t impl_instance_id() const noexcept { return 0; }

  static const char* name();
  //--------------------------------------------------------------------------
};

namespace Tools {
namespace Experimental {
template <>
struct DeviceTypeTraits<Serial> {
  static constexpr DeviceType id = DeviceType::Serial;
};
}  // namespace Experimental
}  // namespace Tools

namespace Impl {

class SerialSpaceInitializer : public ExecSpaceInitializerBase {
 public:
  SerialSpaceInitializer()  = default;
  ~SerialSpaceInitializer() = default;
  void initialize(const InitArguments& args) final;
  void finalize(const bool) final;
  void fence() final;
  void print_configuration(std::ostream& msg, const bool detail) final;
};

}  // namespace Impl
}  // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

template <>
struct MemorySpaceAccess<Kokkos::Serial::memory_space,
                         Kokkos::Serial::scratch_memory_space> {
  enum : bool { assignable = false };
  enum : bool { accessible = true };
  enum : bool { deepcopy = false };
};

template <>
struct VerifyExecutionCanAccessMemorySpace<
    Kokkos::Serial::memory_space, Kokkos::Serial::scratch_memory_space> {
  enum : bool { value = true };
  inline static void verify(void) {}
  inline static void verify(const void*) {}
};

}  // namespace Impl
}  // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

// Resize thread team data scratch memory
void serial_resize_thread_team_data(size_t pool_reduce_bytes,
                                    size_t team_reduce_bytes,
                                    size_t team_shared_bytes,
                                    size_t thread_local_bytes);

HostThreadTeamData* serial_get_thread_team_data();

} /* namespace Impl */
} /* namespace Kokkos */

namespace Kokkos {
namespace Impl {

/*
 * < Kokkos::Serial , WorkArgTag >
 * < WorkArgTag , Impl::enable_if< std::is_same< Kokkos::Serial ,
 * Kokkos::DefaultExecutionSpace >::value >::type >
 *
 */
template <class... Properties>
class TeamPolicyInternal<Kokkos::Serial, Properties...>
    : public PolicyTraits<Properties...> {
 private:
  size_t m_team_scratch_size[2];
  size_t m_thread_scratch_size[2];
  int m_league_size;
  int m_chunk_size;

 public:
  //! Tag this class as a kokkos execution policy
  using execution_policy = TeamPolicyInternal;

  using traits = PolicyTraits<Properties...>;

  //! Execution space of this execution policy:
  using execution_space = Kokkos::Serial;

  const typename traits::execution_space& space() const {
    static typename traits::execution_space m_space;
    return m_space;
  }

  template <class ExecSpace, class... OtherProperties>
  friend class TeamPolicyInternal;

  template <class... OtherProperties>
  TeamPolicyInternal(
      const TeamPolicyInternal<Kokkos::Serial, OtherProperties...>& p) {
    m_league_size            = p.m_league_size;
    m_team_scratch_size[0]   = p.m_team_scratch_size[0];
    m_thread_scratch_size[0] = p.m_thread_scratch_size[0];
    m_team_scratch_size[1]   = p.m_team_scratch_size[1];
    m_thread_scratch_size[1] = p.m_thread_scratch_size[1];
    m_chunk_size             = p.m_chunk_size;
  }

  //----------------------------------------

  template <class FunctorType>
  int team_size_max(const FunctorType&, const ParallelForTag&) const {
    return 1;
  }
  template <class FunctorType>
  int team_size_max(const FunctorType&, const ParallelReduceTag&) const {
    return 1;
  }
  template <class FunctorType, class ReducerType>
  int team_size_max(const FunctorType&, const ReducerType&,
                    const ParallelReduceTag&) const {
    return 1;
  }
  template <class FunctorType>
  int team_size_recommended(const FunctorType&, const ParallelForTag&) const {
    return 1;
  }
  template <class FunctorType>
  int team_size_recommended(const FunctorType&,
                            const ParallelReduceTag&) const {
    return 1;
  }
  template <class FunctorType, class ReducerType>
  int team_size_recommended(const FunctorType&, const ReducerType&,
                            const ParallelReduceTag&) const {
    return 1;
  }

  //----------------------------------------

  inline int team_size() const { return 1; }
  inline bool impl_auto_team_size() const { return false; }
  inline bool impl_auto_vector_length() const { return false; }
  inline void impl_set_team_size(size_t) {}
  inline void impl_set_vector_length(size_t) {}
  inline int league_size() const { return m_league_size; }
  inline size_t scratch_size(const int& level, int = 0) const {
    return m_team_scratch_size[level] + m_thread_scratch_size[level];
  }

  inline int impl_vector_length() const { return 1; }
  inline static int vector_length_max() {
    return 1024;
  }  // Use arbitrary large number, is meant as a vectorizable length

  inline static int scratch_size_max(int level) {
    return (level == 0 ? 1024 * 32 : 20 * 1024 * 1024);
  }
  /** \brief  Specify league size, request team size */
  TeamPolicyInternal(const execution_space&, int league_size_request,
                     int team_size_request, int /* vector_length_request */ = 1)
      : m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_league_size(league_size_request),
        m_chunk_size(32) {
    if (team_size_request > 1)
      Kokkos::abort("Kokkos::abort: Requested Team Size is too large!");
  }

  TeamPolicyInternal(const execution_space& space, int league_size_request,
                     const Kokkos::AUTO_t& /**team_size_request*/,
                     int vector_length_request = 1)
      : TeamPolicyInternal(space, league_size_request, -1,
                           vector_length_request) {}

  TeamPolicyInternal(const execution_space& space, int league_size_request,
                     const Kokkos::AUTO_t& /* team_size_request */
                     ,
                     const Kokkos::AUTO_t& /* vector_length_request */
                     )
      : TeamPolicyInternal(space, league_size_request, -1, -1) {}

  TeamPolicyInternal(const execution_space& space, int league_size_request,
                     int team_size_request,
                     const Kokkos::AUTO_t& /* vector_length_request */
                     )
      : TeamPolicyInternal(space, league_size_request, team_size_request, -1) {}

  TeamPolicyInternal(int league_size_request,
                     const Kokkos::AUTO_t& team_size_request,
                     int vector_length_request = 1)
      : TeamPolicyInternal(typename traits::execution_space(),
                           league_size_request, team_size_request,
                           vector_length_request) {}

  TeamPolicyInternal(int league_size_request,
                     const Kokkos::AUTO_t& team_size_request,
                     const Kokkos::AUTO_t& vector_length_request)
      : TeamPolicyInternal(typename traits::execution_space(),
                           league_size_request, team_size_request,
                           vector_length_request) {}
  TeamPolicyInternal(int league_size_request, int team_size_request,
                     const Kokkos::AUTO_t& vector_length_request)
      : TeamPolicyInternal(typename traits::execution_space(),
                           league_size_request, team_size_request,
                           vector_length_request) {}

  TeamPolicyInternal(int league_size_request, int team_size_request,
                     int vector_length_request = 1)
      : TeamPolicyInternal(typename traits::execution_space(),
                           league_size_request, team_size_request,
                           vector_length_request) {}

  inline int chunk_size() const { return m_chunk_size; }

  /** \brief set chunk_size to a discrete value*/
  inline TeamPolicyInternal& set_chunk_size(
      typename traits::index_type chunk_size_) {
    m_chunk_size = chunk_size_;
    return *this;
  }

  /** \brief set per team scratch size for a specific level of the scratch
   * hierarchy */
  inline TeamPolicyInternal& set_scratch_size(const int& level,
                                              const PerTeamValue& per_team) {
    m_team_scratch_size[level] = per_team.value;
    return *this;
  }

  /** \brief set per thread scratch size for a specific level of the scratch
   * hierarchy */
  inline TeamPolicyInternal& set_scratch_size(
      const int& level, const PerThreadValue& per_thread) {
    m_thread_scratch_size[level] = per_thread.value;
    return *this;
  }

  /** \brief set per thread and per team scratch size for a specific level of
   * the scratch hierarchy */
  inline TeamPolicyInternal& set_scratch_size(
      const int& level, const PerTeamValue& per_team,
      const PerThreadValue& per_thread) {
    m_team_scratch_size[level]   = per_team.value;
    m_thread_scratch_size[level] = per_thread.value;
    return *this;
  }

  using member_type = Impl::HostThreadTeamMember<Kokkos::Serial>;
};
} /* namespace Impl */
} /* namespace Kokkos */

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/
/* Parallel patterns for Kokkos::Serial with RangePolicy */

namespace Kokkos {
namespace Impl {

template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::RangePolicy<Traits...>, Kokkos::Serial> {
 private:
  using Policy = Kokkos::RangePolicy<Traits...>;

  const FunctorType m_functor;
  const Policy m_policy;

  template <class TagType>
  typename std::enable_if<std::is_same<TagType, void>::value>::type exec()
      const {
    const typename Policy::member_type e = m_policy.end();
    for (typename Policy::member_type i = m_policy.begin(); i < e; ++i) {
      m_functor(i);
    }
  }

  template <class TagType>
  typename std::enable_if<!std::is_same<TagType, void>::value>::type exec()
      const {
    const TagType t{};
    const typename Policy::member_type e = m_policy.end();
    for (typename Policy::member_type i = m_policy.begin(); i < e; ++i) {
      m_functor(t, i);
    }
  }

 public:
  inline void execute() const {
    this->template exec<typename Policy::work_tag>();
  }

  inline ParallelFor(const FunctorType& arg_functor, const Policy& arg_policy)
      : m_functor(arg_functor), m_policy(arg_policy) {}
};

/*--------------------------------------------------------------------------*/

template <class FunctorType, class ReducerType, class... Traits>
class ParallelReduce<FunctorType, Kokkos::RangePolicy<Traits...>, ReducerType,
                     Kokkos::Serial> {
 private:
  using Policy  = Kokkos::RangePolicy<Traits...>;
  using WorkTag = typename Policy::work_tag;

  using ReducerConditional =
      Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                         FunctorType, ReducerType>;

  using ReducerTypeFwd = typename ReducerConditional::type;
  using WorkTagFwd =
      typename Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                                  WorkTag, void>::type;

  using Analysis =
      FunctorAnalysis<FunctorPatternInterface::REDUCE, Policy, FunctorType>;

  using ValueInit = Kokkos::Impl::FunctorValueInit<ReducerTypeFwd, WorkTagFwd>;

  using pointer_type   = typename Analysis::pointer_type;
  using reference_type = typename Analysis::reference_type;

  const FunctorType m_functor;
  const Policy m_policy;
  const ReducerType m_reducer;
  const pointer_type m_result_ptr;

  template <class TagType>
  inline typename std::enable_if<std::is_same<TagType, void>::value>::type exec(
      reference_type update) const {
    const typename Policy::member_type e = m_policy.end();
    for (typename Policy::member_type i = m_policy.begin(); i < e; ++i) {
      m_functor(i, update);
    }
  }

  template <class TagType>
  inline typename std::enable_if<!std::is_same<TagType, void>::value>::type
  exec(reference_type update) const {
    const TagType t{};

    const typename Policy::member_type e = m_policy.end();
    for (typename Policy::member_type i = m_policy.begin(); i < e; ++i) {
      m_functor(t, i, update);
    }
  }

 public:
  inline void execute() const {
    const size_t pool_reduce_size =
        Analysis::value_size(ReducerConditional::select(m_functor, m_reducer));
    const size_t team_reduce_size  = 0;  // Never shrinks
    const size_t team_shared_size  = 0;  // Never shrinks
    const size_t thread_local_size = 0;  // Never shrinks

    serial_resize_thread_team_data(pool_reduce_size, team_reduce_size,
                                   team_shared_size, thread_local_size);

    HostThreadTeamData& data = *serial_get_thread_team_data();

    pointer_type ptr =
        m_result_ptr ? m_result_ptr : pointer_type(data.pool_reduce_local());

    reference_type update =
        ValueInit::init(ReducerConditional::select(m_functor, m_reducer), ptr);

    this->template exec<WorkTag>(update);

    Kokkos::Impl::FunctorFinal<ReducerTypeFwd, WorkTagFwd>::final(
        ReducerConditional::select(m_functor, m_reducer), ptr);
  }

  template <class HostViewType>
  ParallelReduce(
      const FunctorType& arg_functor, const Policy& arg_policy,
      const HostViewType& arg_result_view,
      typename std::enable_if<Kokkos::is_view<HostViewType>::value &&
                                  !Kokkos::is_reducer_type<ReducerType>::value,
                              void*>::type = nullptr)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(InvalidType()),
        m_result_ptr(arg_result_view.data()) {
    static_assert(Kokkos::is_view<HostViewType>::value,
                  "Kokkos::Serial reduce result must be a View");

    static_assert(
        Kokkos::Impl::MemorySpaceAccess<typename HostViewType::memory_space,
                                        Kokkos::HostSpace>::accessible,
        "Kokkos::Serial reduce result must be a View in HostSpace");
  }

  inline ParallelReduce(const FunctorType& arg_functor, Policy arg_policy,
                        const ReducerType& reducer)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(reducer),
        m_result_ptr(reducer.view().data()) {
    /*static_assert( std::is_same< typename ViewType::memory_space
                                    , Kokkos::HostSpace >::value
      , "Reduction result on Kokkos::OpenMP must be a Kokkos::View in HostSpace"
      );*/
  }
};

/*--------------------------------------------------------------------------*/

template <class FunctorType, class... Traits>
class ParallelScan<FunctorType, Kokkos::RangePolicy<Traits...>,
                   Kokkos::Serial> {
 private:
  using Policy  = Kokkos::RangePolicy<Traits...>;
  using WorkTag = typename Policy::work_tag;

  using Analysis =
      FunctorAnalysis<FunctorPatternInterface::SCAN, Policy, FunctorType>;

  using ValueInit = Kokkos::Impl::FunctorValueInit<FunctorType, WorkTag>;

  using pointer_type   = typename Analysis::pointer_type;
  using reference_type = typename Analysis::reference_type;

  const FunctorType m_functor;
  const Policy m_policy;

  template <class TagType>
  inline typename std::enable_if<std::is_same<TagType, void>::value>::type exec(
      reference_type update) const {
    const typename Policy::member_type e = m_policy.end();
    for (typename Policy::member_type i = m_policy.begin(); i < e; ++i) {
      m_functor(i, update, true);
    }
  }

  template <class TagType>
  inline typename std::enable_if<!std::is_same<TagType, void>::value>::type
  exec(reference_type update) const {
    const TagType t{};
    const typename Policy::member_type e = m_policy.end();
    for (typename Policy::member_type i = m_policy.begin(); i < e; ++i) {
      m_functor(t, i, update, true);
    }
  }

 public:
  inline void execute() const {
    const size_t pool_reduce_size  = Analysis::value_size(m_functor);
    const size_t team_reduce_size  = 0;  // Never shrinks
    const size_t team_shared_size  = 0;  // Never shrinks
    const size_t thread_local_size = 0;  // Never shrinks

    serial_resize_thread_team_data(pool_reduce_size, team_reduce_size,
                                   team_shared_size, thread_local_size);

    HostThreadTeamData& data = *serial_get_thread_team_data();

    reference_type update =
        ValueInit::init(m_functor, pointer_type(data.pool_reduce_local()));

    this->template exec<WorkTag>(update);
  }

  inline ParallelScan(const FunctorType& arg_functor, const Policy& arg_policy)
      : m_functor(arg_functor), m_policy(arg_policy) {}
};

/*--------------------------------------------------------------------------*/
template <class FunctorType, class ReturnType, class... Traits>
class ParallelScanWithTotal<FunctorType, Kokkos::RangePolicy<Traits...>,
                            ReturnType, Kokkos::Serial> {
 private:
  using Policy  = Kokkos::RangePolicy<Traits...>;
  using WorkTag = typename Policy::work_tag;

  using Analysis =
      FunctorAnalysis<FunctorPatternInterface::SCAN, Policy, FunctorType>;

  using ValueInit = Kokkos::Impl::FunctorValueInit<FunctorType, WorkTag>;

  using pointer_type   = typename Analysis::pointer_type;
  using reference_type = typename Analysis::reference_type;

  const FunctorType m_functor;
  const Policy m_policy;
  ReturnType& m_returnvalue;

  template <class TagType>
  inline typename std::enable_if<std::is_same<TagType, void>::value>::type exec(
      reference_type update) const {
    const typename Policy::member_type e = m_policy.end();
    for (typename Policy::member_type i = m_policy.begin(); i < e; ++i) {
      m_functor(i, update, true);
    }
  }

  template <class TagType>
  inline typename std::enable_if<!std::is_same<TagType, void>::value>::type
  exec(reference_type update) const {
    const TagType t{};
    const typename Policy::member_type e = m_policy.end();
    for (typename Policy::member_type i = m_policy.begin(); i < e; ++i) {
      m_functor(t, i, update, true);
    }
  }

 public:
  inline void execute() {
    const size_t pool_reduce_size  = Analysis::value_size(m_functor);
    const size_t team_reduce_size  = 0;  // Never shrinks
    const size_t team_shared_size  = 0;  // Never shrinks
    const size_t thread_local_size = 0;  // Never shrinks

    serial_resize_thread_team_data(pool_reduce_size, team_reduce_size,
                                   team_shared_size, thread_local_size);

    HostThreadTeamData& data = *serial_get_thread_team_data();

    reference_type update =
        ValueInit::init(m_functor, pointer_type(data.pool_reduce_local()));

    this->template exec<WorkTag>(update);

    m_returnvalue = update;
  }

  inline ParallelScanWithTotal(const FunctorType& arg_functor,
                               const Policy& arg_policy,
                               ReturnType& arg_returnvalue)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_returnvalue(arg_returnvalue) {}
};

}  // namespace Impl
}  // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/
/* Parallel patterns for Kokkos::Serial with MDRangePolicy */

namespace Kokkos {
namespace Impl {

template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::MDRangePolicy<Traits...>,
                  Kokkos::Serial> {
 private:
  using MDRangePolicy = Kokkos::MDRangePolicy<Traits...>;
  using Policy        = typename MDRangePolicy::impl_range_policy;

  using iterate_type = typename Kokkos::Impl::HostIterateTile<
      MDRangePolicy, FunctorType, typename MDRangePolicy::work_tag, void>;

  const FunctorType m_functor;
  const MDRangePolicy m_mdr_policy;
  const Policy m_policy;

  void exec() const {
    const typename Policy::member_type e = m_policy.end();
    for (typename Policy::member_type i = m_policy.begin(); i < e; ++i) {
      iterate_type(m_mdr_policy, m_functor)(i);
    }
  }

 public:
  inline void execute() const { this->exec(); }
  template <typename Policy, typename Functor>
  static int max_tile_size_product(const Policy&, const Functor&) {
    /**
     * 1024 here is just our guess for a reasonable max tile size,
     * it isn't a hardware constraint. If people see a use for larger
     * tile size products, we're happy to change this.
     */
    return 1024;
  }
  inline ParallelFor(const FunctorType& arg_functor,
                     const MDRangePolicy& arg_policy)
      : m_functor(arg_functor),
        m_mdr_policy(arg_policy),
        m_policy(Policy(0, m_mdr_policy.m_num_tiles).set_chunk_size(1)) {}
};

template <class FunctorType, class ReducerType, class... Traits>
class ParallelReduce<FunctorType, Kokkos::MDRangePolicy<Traits...>, ReducerType,
                     Kokkos::Serial> {
 private:
  using MDRangePolicy = Kokkos::MDRangePolicy<Traits...>;
  using Policy        = typename MDRangePolicy::impl_range_policy;

  using WorkTag = typename MDRangePolicy::work_tag;

  using ReducerConditional =
      Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                         FunctorType, ReducerType>;
  using ReducerTypeFwd = typename ReducerConditional::type;
  using WorkTagFwd =
      typename Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                                  WorkTag, void>::type;

  using Analysis = FunctorAnalysis<FunctorPatternInterface::REDUCE,
                                   MDRangePolicy, FunctorType>;

  using ValueInit = Kokkos::Impl::FunctorValueInit<ReducerTypeFwd, WorkTagFwd>;

  using pointer_type   = typename Analysis::pointer_type;
  using value_type     = typename Analysis::value_type;
  using reference_type = typename Analysis::reference_type;

  using iterate_type =
      typename Kokkos::Impl::HostIterateTile<MDRangePolicy, FunctorType,
                                             WorkTag, reference_type>;

  const FunctorType m_functor;
  const MDRangePolicy m_mdr_policy;
  const Policy m_policy;
  const ReducerType m_reducer;
  const pointer_type m_result_ptr;

  inline void exec(reference_type update) const {
    const typename Policy::member_type e = m_policy.end();
    for (typename Policy::member_type i = m_policy.begin(); i < e; ++i) {
      iterate_type(m_mdr_policy, m_functor, update)(i);
    }
  }

 public:
  template <typename Policy, typename Functor>
  static int max_tile_size_product(const Policy&, const Functor&) {
    /**
     * 1024 here is just our guess for a reasonable max tile size,
     * it isn't a hardware constraint. If people see a use for larger
     * tile size products, we're happy to change this.
     */
    return 1024;
  }
  inline void execute() const {
    const size_t pool_reduce_size =
        Analysis::value_size(ReducerConditional::select(m_functor, m_reducer));
    const size_t team_reduce_size  = 0;  // Never shrinks
    const size_t team_shared_size  = 0;  // Never shrinks
    const size_t thread_local_size = 0;  // Never shrinks

    serial_resize_thread_team_data(pool_reduce_size, team_reduce_size,
                                   team_shared_size, thread_local_size);

    HostThreadTeamData& data = *serial_get_thread_team_data();

    pointer_type ptr =
        m_result_ptr ? m_result_ptr : pointer_type(data.pool_reduce_local());

    reference_type update =
        ValueInit::init(ReducerConditional::select(m_functor, m_reducer), ptr);

    this->exec(update);

    Kokkos::Impl::FunctorFinal<ReducerTypeFwd, WorkTagFwd>::final(
        ReducerConditional::select(m_functor, m_reducer), ptr);
  }

  template <class HostViewType>
  ParallelReduce(
      const FunctorType& arg_functor, const MDRangePolicy& arg_policy,
      const HostViewType& arg_result_view,
      typename std::enable_if<Kokkos::is_view<HostViewType>::value &&
                                  !Kokkos::is_reducer_type<ReducerType>::value,
                              void*>::type = nullptr)
      : m_functor(arg_functor),
        m_mdr_policy(arg_policy),
        m_policy(Policy(0, m_mdr_policy.m_num_tiles).set_chunk_size(1)),
        m_reducer(InvalidType()),
        m_result_ptr(arg_result_view.data()) {
    static_assert(Kokkos::is_view<HostViewType>::value,
                  "Kokkos::Serial reduce result must be a View");

    static_assert(
        Kokkos::Impl::MemorySpaceAccess<typename HostViewType::memory_space,
                                        Kokkos::HostSpace>::accessible,
        "Kokkos::Serial reduce result must be a View in HostSpace");
  }

  inline ParallelReduce(const FunctorType& arg_functor,
                        MDRangePolicy arg_policy, const ReducerType& reducer)
      : m_functor(arg_functor),
        m_mdr_policy(arg_policy),
        m_policy(Policy(0, m_mdr_policy.m_num_tiles).set_chunk_size(1)),
        m_reducer(reducer),
        m_result_ptr(reducer.view().data()) {
    /*static_assert( std::is_same< typename ViewType::memory_space
                                    , Kokkos::HostSpace >::value
      , "Reduction result on Kokkos::OpenMP must be a Kokkos::View in HostSpace"
      );*/
  }
};

}  // namespace Impl
}  // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/
/* Parallel patterns for Kokkos::Serial with TeamPolicy */

namespace Kokkos {
namespace Impl {

template <class FunctorType, class... Properties>
class ParallelFor<FunctorType, Kokkos::TeamPolicy<Properties...>,
                  Kokkos::Serial> {
 private:
  enum { TEAM_REDUCE_SIZE = 512 };

  using Policy = TeamPolicyInternal<Kokkos::Serial, Properties...>;
  using Member = typename Policy::member_type;

  const FunctorType m_functor;
  const int m_league;
  const int m_shared;

  template <class TagType>
  inline typename std::enable_if<std::is_same<TagType, void>::value>::type exec(
      HostThreadTeamData& data) const {
    for (int ileague = 0; ileague < m_league; ++ileague) {
      m_functor(Member(data, ileague, m_league));
    }
  }

  template <class TagType>
  inline typename std::enable_if<!std::is_same<TagType, void>::value>::type
  exec(HostThreadTeamData& data) const {
    const TagType t{};
    for (int ileague = 0; ileague < m_league; ++ileague) {
      m_functor(t, Member(data, ileague, m_league));
    }
  }

 public:
  inline void execute() const {
    const size_t pool_reduce_size  = 0;  // Never shrinks
    const size_t team_reduce_size  = TEAM_REDUCE_SIZE;
    const size_t team_shared_size  = m_shared;
    const size_t thread_local_size = 0;  // Never shrinks

    serial_resize_thread_team_data(pool_reduce_size, team_reduce_size,
                                   team_shared_size, thread_local_size);

    HostThreadTeamData& data = *serial_get_thread_team_data();

    this->template exec<typename Policy::work_tag>(data);
  }

  ParallelFor(const FunctorType& arg_functor, const Policy& arg_policy)
      : m_functor(arg_functor),
        m_league(arg_policy.league_size()),
        m_shared(arg_policy.scratch_size(0) + arg_policy.scratch_size(1) +
                 FunctorTeamShmemSize<FunctorType>::value(arg_functor, 1)) {}
};

/*--------------------------------------------------------------------------*/

template <class FunctorType, class ReducerType, class... Properties>
class ParallelReduce<FunctorType, Kokkos::TeamPolicy<Properties...>,
                     ReducerType, Kokkos::Serial> {
 private:
  enum { TEAM_REDUCE_SIZE = 512 };

  using Policy = TeamPolicyInternal<Kokkos::Serial, Properties...>;

  using Analysis =
      FunctorAnalysis<FunctorPatternInterface::REDUCE, Policy, FunctorType>;

  using Member  = typename Policy::member_type;
  using WorkTag = typename Policy::work_tag;

  using ReducerConditional =
      Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                         FunctorType, ReducerType>;
  using ReducerTypeFwd = typename ReducerConditional::type;
  using WorkTagFwd =
      typename Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                                  WorkTag, void>::type;

  using ValueInit = Kokkos::Impl::FunctorValueInit<ReducerTypeFwd, WorkTagFwd>;

  using pointer_type   = typename Analysis::pointer_type;
  using reference_type = typename Analysis::reference_type;

  const FunctorType m_functor;
  const int m_league;
  const ReducerType m_reducer;
  pointer_type m_result_ptr;
  const int m_shared;

  template <class TagType>
  inline typename std::enable_if<std::is_same<TagType, void>::value>::type exec(
      HostThreadTeamData& data, reference_type update) const {
    for (int ileague = 0; ileague < m_league; ++ileague) {
      m_functor(Member(data, ileague, m_league), update);
    }
  }

  template <class TagType>
  inline typename std::enable_if<!std::is_same<TagType, void>::value>::type
  exec(HostThreadTeamData& data, reference_type update) const {
    const TagType t{};

    for (int ileague = 0; ileague < m_league; ++ileague) {
      m_functor(t, Member(data, ileague, m_league), update);
    }
  }

 public:
  inline void execute() const {
    const size_t pool_reduce_size =
        Analysis::value_size(ReducerConditional::select(m_functor, m_reducer));

    const size_t team_reduce_size  = TEAM_REDUCE_SIZE;
    const size_t team_shared_size  = m_shared;
    const size_t thread_local_size = 0;  // Never shrinks

    serial_resize_thread_team_data(pool_reduce_size, team_reduce_size,
                                   team_shared_size, thread_local_size);

    HostThreadTeamData& data = *serial_get_thread_team_data();

    pointer_type ptr =
        m_result_ptr ? m_result_ptr : pointer_type(data.pool_reduce_local());

    reference_type update =
        ValueInit::init(ReducerConditional::select(m_functor, m_reducer), ptr);

    this->template exec<WorkTag>(data, update);

    Kokkos::Impl::FunctorFinal<ReducerTypeFwd, WorkTagFwd>::final(
        ReducerConditional::select(m_functor, m_reducer), ptr);
  }

  template <class ViewType>
  ParallelReduce(
      const FunctorType& arg_functor, const Policy& arg_policy,
      const ViewType& arg_result,
      typename std::enable_if<Kokkos::is_view<ViewType>::value &&
                                  !Kokkos::is_reducer_type<ReducerType>::value,
                              void*>::type = nullptr)
      : m_functor(arg_functor),
        m_league(arg_policy.league_size()),
        m_reducer(InvalidType()),
        m_result_ptr(arg_result.data()),
        m_shared(arg_policy.scratch_size(0) + arg_policy.scratch_size(1) +
                 FunctorTeamShmemSize<FunctorType>::value(m_functor, 1)) {
    static_assert(Kokkos::is_view<ViewType>::value,
                  "Reduction result on Kokkos::Serial must be a Kokkos::View");

    static_assert(
        Kokkos::Impl::MemorySpaceAccess<typename ViewType::memory_space,
                                        Kokkos::HostSpace>::accessible,
        "Reduction result on Kokkos::Serial must be a Kokkos::View in "
        "HostSpace");
  }

  inline ParallelReduce(const FunctorType& arg_functor, Policy arg_policy,
                        const ReducerType& reducer)
      : m_functor(arg_functor),
        m_league(arg_policy.league_size()),
        m_reducer(reducer),
        m_result_ptr(reducer.view().data()),
        m_shared(arg_policy.scratch_size(0) + arg_policy.scratch_size(1) +
                 FunctorTeamShmemSize<FunctorType>::value(arg_functor, 1)) {
    /*static_assert( std::is_same< typename ViewType::memory_space
                            , Kokkos::HostSpace >::value
    , "Reduction result on Kokkos::OpenMP must be a Kokkos::View in HostSpace"
    );*/
  }
};

}  // namespace Impl
}  // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Experimental {

template <>
class UniqueToken<Serial, UniqueTokenScope::Instance> {
 public:
  using execution_space = Serial;
  using size_type       = int;

  /// \brief create object size for concurrency on the given instance
  ///
  /// This object should not be shared between instances
  UniqueToken(execution_space const& = execution_space()) noexcept {}

  /// \brief create object size for requested size on given instance
  ///
  /// It is the users responsibility to only acquire size tokens concurrently
  UniqueToken(size_type, execution_space const& = execution_space()) {}

  /// \brief upper bound for acquired values, i.e. 0 <= value < size()
  KOKKOS_INLINE_FUNCTION
  int size() const noexcept { return 1; }

  /// \brief acquire value such that 0 <= value < size()
  KOKKOS_INLINE_FUNCTION
  int acquire() const noexcept { return 0; }

  /// \brief release a value acquired by generate
  KOKKOS_INLINE_FUNCTION
  void release(int) const noexcept {}
};

template <>
class UniqueToken<Serial, UniqueTokenScope::Global> {
 public:
  using execution_space = Serial;
  using size_type       = int;

  /// \brief create object size for concurrency on the given instance
  ///
  /// This object should not be shared between instances
  UniqueToken(execution_space const& = execution_space()) noexcept {}

  /// \brief upper bound for acquired values, i.e. 0 <= value < size()
  KOKKOS_INLINE_FUNCTION
  int size() const noexcept { return 1; }

  /// \brief acquire value such that 0 <= value < size()
  KOKKOS_INLINE_FUNCTION
  int acquire() const noexcept { return 0; }

  /// \brief release a value acquired by generate
  KOKKOS_INLINE_FUNCTION
  void release(int) const noexcept {}
};

}  // namespace Experimental
}  // namespace Kokkos

#include <impl/Kokkos_Serial_Task.hpp>

#endif  // defined( KOKKOS_ENABLE_SERIAL )
#endif  /* #define KOKKOS_SERIAL_HPP */
