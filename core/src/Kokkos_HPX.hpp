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

#ifndef KOKKOS_HPX_HPP
#define KOKKOS_HPX_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_HPX)

#include <Kokkos_Core_fwd.hpp>

#include <Kokkos_HostSpace.hpp>
#include <cstddef>
#include <iosfwd>

#ifdef KOKKOS_ENABLE_HBWSPACE
#include <Kokkos_HBWSpace.hpp>
#endif

#include <Kokkos_HostSpace.hpp>
#include <Kokkos_Layout.hpp>
#include <Kokkos_MemoryTraits.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_ScratchSpace.hpp>
#include <Kokkos_TaskScheduler.hpp>
#include <impl/Kokkos_FunctorAdapter.hpp>
#include <impl/Kokkos_FunctorAnalysis.hpp>
#include <impl/Kokkos_Profiling_Interface.hpp>
#include <impl/Kokkos_Tags.hpp>
#include <impl/Kokkos_TaskQueue.hpp>

#include <KokkosExp_MDRangePolicy.hpp>

#include <hpx/hpx_start.hpp>
#include <hpx/include/apply.hpp>
#include <hpx/include/local_lcos.hpp>
#include <hpx/include/parallel_executor_parameters.hpp>
#include <hpx/include/parallel_for_loop.hpp>
#include <hpx/include/parallel_reduce.hpp>
#include <hpx/include/run_as.hpp>
#include <hpx/include/runtime.hpp>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

#define KOKKOS_HPX_ENABLE_LOG 0

#if KOKKOS_HPX_ENABLE_LOG == 1
#define LOG(x)                                                                 \
  {                                                                            \
    std::stringstream s__;                                                     \
    s__ << x << std::endl;                                                     \
    std::cerr << s__.str();                                                    \
  }
#else
#define LOG(...)
#endif

namespace hpx {
template <typename F> inline void run_hpx_function(F &&f) {
  if (hpx::threads::get_self_ptr()) {
    f();
  } else {
      hpx::threads::run_as_hpx_thread(std::move(f));
  }
}
} // namespace hpx

namespace Kokkos {

static bool kokkos_hpx_initialized = false;

// This represents the HPX runtime instance. It can be stateful and keep track
// of an instance of its own, but in this case it should probably just be a way
// to access properties of the HPX runtime through a common API (as defined by
// Kokkos). Can in principle create as many of these as we want and all can
// access the same HPX runtime (there can only be one in any case). Most methods
// are static.
class HPX {
public:
  using execution_space = HPX;
  using memory_space = HostSpace;
  using device_type = Kokkos::Device<execution_space, memory_space>;
  using array_layout = LayoutRight;
  using size_type = memory_space::size_type;
  using scratch_memory_space = ScratchMemorySpace<HPX>;

  inline HPX() noexcept {}
  static void print_configuration(std::ostream &, const bool verbose = false) {
    std::cout << "HPX backend" << std::endl;
  }

  inline static bool in_parallel(HPX const & = HPX()) noexcept { return false; }
  inline static void fence(HPX const & = HPX()) noexcept {
    // TODO: This could keep a list of futures of ongoing tasks and wait for
    // all to be ready.

    // Can be no-op currently as long as all parallel calls are blocking.
  }

  inline static bool is_asynchronous(HPX const & = HPX()) noexcept {
    return false;
  }

  static std::vector<HPX> partition(...) {
    Kokkos::abort("HPX::partition_master: can't partition an HPX instance\n");
  }

  template <typename F>
  static void partition_master(F const &f, int requested_num_partitions = 0,
                               int requested_partition_size = 0) {
    if (requested_num_partitions > 1) {
      Kokkos::abort("HPX::partition_master: can't partition an HPX instance\n");
    }
  }

  static int concurrency() { return hpx::get_num_worker_threads(); }

  // NOTE: impl_ is optional
  static void impl_initialize(int thread_count) {
    LOG("HPX::initialize");

    hpx::runtime *rt = hpx::get_runtime_ptr();
    if (rt != nullptr) {
      LOG("The HPX backend has already been initialized, skipping");
    } else {
      std::vector<std::string> config = {
          "hpx.os_threads=" + std::to_string(thread_count),
#ifdef KOKKOS_DEBUG
          "--hpx:attach-debugger=exception",
#endif
      };
      int argc_hpx = 1;
      char name[] = "kokkos_hpx";
      char *argv_hpx[] = {name, nullptr};
      hpx::start(nullptr, argc_hpx, argv_hpx, config);
      kokkos_hpx_initialized = true;
    }
  }

  static void impl_initialize() {
    LOG("HPX::initialize");

    hpx::runtime *rt = hpx::get_runtime_ptr();
    if (rt != nullptr) {
      LOG("The HPX backend has already been initialized, skipping");
    } else {
      std::vector<std::string> config = {
#ifdef KOKKOS_DEBUG
          "--hpx:attach-debugger=exception",
#endif
      };
      int argc_hpx = 1;
      char name[] = "kokkos_hpx";
      char *argv_hpx[] = {name, nullptr};
      hpx::start(nullptr, argc_hpx, argv_hpx, config);
      kokkos_hpx_initialized = true;
    }
  }

  static bool impl_is_initialized() noexcept {
    LOG("HPX::impl_is_initialized");
    hpx::runtime *rt = hpx::get_runtime_ptr();
    return rt != nullptr;
  }

  static void impl_finalize() {
    LOG("HPX::finalize");

    if (kokkos_hpx_initialized) {
      hpx::runtime *rt = hpx::get_runtime_ptr();
      if (rt == nullptr) {
        LOG("HPX::finalize: The backend has been stopped manually");
      } else {
        hpx::apply([]() { hpx::finalize(); });
        hpx::stop();
      }
    } else {
      LOG("HPX::finalize: the runtime was not started through Kokkos, "
          "skipping");
    }
  };

  inline static int impl_thread_pool_size() noexcept {
    LOG("HPX::impl_thread_pool_size");
    hpx::runtime *rt = hpx::get_runtime_ptr();
    if (rt == nullptr) {
      // TODO: Exit with error?
      return 0;
    } else {
      if (hpx::threads::get_self_ptr() == nullptr) {
        return concurrency();
      } else {
        return hpx::this_thread::get_pool()->get_os_thread_count();
      }
    }
  }

  static int impl_thread_pool_rank() noexcept {
    LOG("HPX::impl_thread_pool_rank");
    hpx::runtime *rt = hpx::get_runtime_ptr();
    if (rt == nullptr) {
      // TODO: Exit with error?
      return 0;
    } else {
      if (hpx::threads::get_self_ptr() == nullptr) {
        // TODO: Exit with error?
        return 0;
      } else {
        return hpx::this_thread::get_pool()->get_pool_index();
      }
    }
  }

  inline static int impl_thread_pool_size(int depth) {
    LOG("HPX::impl_thread_pool_size");
    return 0;
  }

  inline static int impl_max_hardware_threads() noexcept {
    LOG("HPX::impl_max_hardware_threads");
    return concurrency();
  }

  KOKKOS_INLINE_FUNCTION static int impl_hardware_thread_id() noexcept {
    LOG("HPX::impl_hardware_thread_id");
    return hpx::get_worker_thread_num();
  }

  static constexpr const char *name() noexcept { return "HPX"; }
};
} // namespace Kokkos

// These specify the properties of the default memory space associate with the
// HPX execution space. Need not worry too much about this one right now.
namespace Kokkos {
namespace Impl {
template <>
struct MemorySpaceAccess<Kokkos::HPX::memory_space,
                         Kokkos::HPX::scratch_memory_space> {
  enum { assignable = false };
  enum { accessible = true };
  enum { deepcopy = false };
};

template <>
struct VerifyExecutionCanAccessMemorySpace<Kokkos::HPX::memory_space,
                                           Kokkos::HPX::scratch_memory_space> {
  enum { value = true };
  inline static void verify(void) {}
  inline static void verify(const void *) {}
};
} // namespace Impl
} // namespace Kokkos

namespace Kokkos {
namespace Experimental {
// Instance specialization is for partitioned instances, Global for the full
// instance.
template <> class UniqueToken<HPX, UniqueTokenScope::Instance> {
public:
  using execution_space = HPX;
  using size_type = int;
  UniqueToken(execution_space const & = execution_space()) noexcept {}

  // NOTE: Threads are not guaranteed to run where you tell them to run.
  int size() const noexcept { return hpx::get_num_worker_threads(); }
  int acquire() const noexcept { return hpx::get_worker_thread_num(); }
  void release(int) const noexcept {}
};

template <> class UniqueToken<HPX, UniqueTokenScope::Global> {
public:
  using execution_space = HPX;
  using size_type = int;
  UniqueToken(execution_space const & = execution_space()) noexcept {}

  // NOTE: Threads are not guaranteed to run where you tell them to run.
  int size() const noexcept { return hpx::get_num_worker_threads(); }
  int acquire() const noexcept { return hpx::get_worker_thread_num(); }
  void release(int) const noexcept {}
};
} // namespace Experimental
} // namespace Kokkos

// TODO: This is not complete yet.
namespace Kokkos {
namespace Impl {

// HPXTeamMember is the member_type that gets passed into user code when calling
// parallel for loops with thread team execution policies. This should provide
// enough information for the user code to determine in which thread, and team
// it is running, i.e. the indices (ranks).
//
// It should also provide access to scratch memory. The scratch memory should be
// allocated at the top level, before the first call to a parallel function, so
// that it is available for allocations in parallel code.
//
// It takes a team policy as an argument in the constructor.
struct HPXTeamMember {
private:
  using execution_space = Kokkos::HPX;
  using scratch_memory_space = Kokkos::ScratchMemorySpace<Kokkos::HPX>;

  // This is the actual shared scratch memory. It has two levels (0, 1). Levels
  // can be allocated from the same place, but have to be separate pieces of
  // memory. Scratch memory is separate from reduction memory.
  scratch_memory_space m_team_shared;
  // Size of the shared scratch memory.
  std::size_t m_team_shared_size;

  // This is the reduction buffer. It contains team_size * 512 bytes. NOTE: This
  // is also "misused" for other purposes. It can be used for the broadcast and
  // scan operations as well. Not needed if team_size = 1.
  char *m_reduce_buffer;
  std::size_t m_reduce_buffer_size;

  int m_league_size;
  int m_league_rank;
  int m_team_size;
  int m_team_rank;

  // This is used to implement the barrier function. This is not needed if team
  // size is 1.
  // std::shared_ptr<hpx::lcos::local::barrier> m_barrier;

public:
  // Returns the team shared scratch memory. Exactly the same as
  // team_scratch(0).
  KOKKOS_INLINE_FUNCTION
  const scratch_memory_space &team_shmem() const {
    return m_team_shared.set_team_thread_mode(0, 1, 0);
  }

  // Returns the team shared scratch memory at the specified level. Level
  // ignored on CPU backends. Exactly the same as team_shmem.
  KOKKOS_INLINE_FUNCTION
  const execution_space::scratch_memory_space &team_scratch(const int) const {
    return m_team_shared.set_team_thread_mode(0, 1, 0);
  }

  // Scratch space specific for the specified thread.
  KOKKOS_INLINE_FUNCTION
  const execution_space::scratch_memory_space &thread_scratch(const int) const {
    return m_team_shared.set_team_thread_mode(0, team_size(), team_rank());
  }

  KOKKOS_INLINE_FUNCTION int league_rank() const { return m_league_rank; }
  KOKKOS_INLINE_FUNCTION int league_size() const { return m_league_size; }
  KOKKOS_INLINE_FUNCTION int team_rank() const { return m_team_rank; }
  KOKKOS_INLINE_FUNCTION int team_size() const { return m_team_size; }

  template <class... Properties>
  KOKKOS_INLINE_FUNCTION
  HPXTeamMember(const TeamPolicyInternal<Kokkos::HPX, Properties...> &policy,
                const int team_rank, const int league_rank, void *scratch,
                int scratch_size, char *reduce_buffer,
                std::size_t reduce_buffer_size
                /*, std::shared_ptr<hpx::lcos::local::barrier> barrier*/)
      : m_team_shared(scratch, scratch_size, scratch, scratch_size),
        m_team_shared_size(scratch_size), m_league_size(policy.league_size()),
        m_league_rank(league_rank), m_team_size(policy.team_size()),
        m_team_rank(team_rank), m_reduce_buffer(reduce_buffer),
        m_reduce_buffer_size(reduce_buffer_size) /*, m_barrier(barrier)*/ {}

  // Waits for all team members to reach the barrier. TODO: This should also
  // issue a memory fence!?

  // NOTE: Assume team_size = 1 for now, this is a no-op.
  KOKKOS_INLINE_FUNCTION
  void team_barrier() const { /*m_barrier->wait();*/
  }

  // TODO: Need to understand how the following team_* functions work in
  // relation to nested parallelism and how memory should be allocated to
  // accommodate for the temporary values.

  // TODO: Want to write value into something shared between all threads. Need
  // to take care of memory barriers here.

  // This is disabled on OpenMP backend if
  // KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST is not defined. What does that
  // do?
  template <class ValueType>
  KOKKOS_INLINE_FUNCTION void team_broadcast(ValueType &value,
                                             const int &thread_id) const {
    static_assert(std::is_trivially_default_constructible<ValueType>(),
                  "Only trivial constructible types can be broadcasted");

    // Here we simply get the beginning of the reduce buffer (same on all
    // threads) as the place to store the broadcast value.

    // NOTE: No-op with team_size = 1.
    /*
    ValueType *const shared_value = (ValueType *)m_reduce_buffer;

    team_barrier();

    if (m_team_rank == thread_id) {
      *shared_value = value;
    }

    team_barrier();

    value = *shared_value;*/
  }

  template <class Closure, class ValueType>
  KOKKOS_INLINE_FUNCTION void team_broadcast(const Closure &f, ValueType &value,
                                             const int &thread_id) const {
    static_assert(std::is_trivially_default_constructible<ValueType>(),
                  "Only trivial constructible types can be broadcasted");

    // Here we simply get the beginning of the reduce buffer (same on all
    // threads) as the place to store the broadcast value.

    // NOTE: (Almost) no-op with team_size = 1.
    /*
    ValueType *const shared_value = (ValueType *)m_reduce_buffer;

    team_barrier();

    if (m_team_rank == thread_id) {
      f(value);
      *shared_value = value;
    }

    team_barrier();

    value = *shared_value;*/

    value = f(value);
  }

  // TODO
  template <class ValueType, class JoinOp>
  KOKKOS_INLINE_FUNCTION ValueType team_reduce(const ValueType &value,
                                               const JoinOp &op_in) const {
    // if (1 < m_team_size) {
    //   if (m_team_rank != 0) {
    //     // TODO: Magic 512.
    //     *((ValueType *)(m_reduce_buffer + m_team_rank * 512)) = value;
    //   }

    //   // Root does not overwrite shared memory until all threads arrive
    //   // and copy to their local buffer.
    //   team_barrier();

    //   if (m_team_rank == 0) {
    //     const Impl::Reducer<ValueType, JoinOp> reducer(join);

    //     ValueType *const dst = (ValueType *)m_reduce_buffer;
    //     *dst = value;

    //     for (int i = 1; i < m_team_size; ++i) {
    //       value_type *const src =
    //           (value_type *)(m_reduce_buffer + m_team_rank * 512);

    //       reducer.join(dst, *src);
    //     }
    //   }

    //   team_barrier();

    //   // TODO: Don't need to do this for team rank 0.
    //   value = *((value_type *)m_reduce_buffer);
    // }

    // NOTE: No-op with team_size = 1.
    return value;
  }

  template <class ReducerType>
  KOKKOS_INLINE_FUNCTION
      typename std::enable_if<is_reducer<ReducerType>::value>::type
      team_reduce(const ReducerType &reducer) const {
    // if (1 < m_team_size) {
    //   using value_type = typename ReducerType::value_type;

    //   if (0 != m_team_rank) {
    //     *((value_type *)(m_reduce_buffer + m_team_rank * 512)) =
    //         reducer.reference();
    //   }

    //   // Root does not overwrite shared memory until all threads arrive
    //   // and copy to their local buffer.
    //   team_barrier();

    //   if (0 == m_team_rank) {
    //     for (int i = 1; i < m_team_size; ++i) {
    //       value_type *const src =
    //           (value_type *)(m_reduce_buffer + m_team_rank * 512);

    //       reducer.join(reducer.reference(), *src);
    //     }

    //     *((value_type *)m_reduce_buffer) = reducer.reference();
    //   }

    //   team_barrier();

    //   if (0 != m_team_rank) {
    //     reducer.reference() = *((value_type *)m_reduce_buffer);
    //   }
    // }

    // NOTE: No-op with team_size = 1.
  }

  template <typename Type>
  KOKKOS_INLINE_FUNCTION Type
  team_scan(const Type &value, Type *const global_accum = nullptr) const {
    // NOTE: Almost no-op with team_size = 1.
    // TODO: Is 0 correct for all Types?
    if (global_accum) {
      Kokkos::atomic_fetch_add(global_accum, value);
    }

    return 0;
  }
};

// TeamPolicyInternal is the data that gets passed into a parallel function as a
// team policy. It should specify how many teams and threads there should be in
// the league, how much scratch space there should be.
//
// This object doesn't store any persistent state. It just holds parameters for
// parallel execution.
template <class... Properties>
class TeamPolicyInternal<Kokkos::HPX, Properties...>
    : public PolicyTraits<Properties...> {
  int m_league_size;
  int m_team_size;

  // TODO: Are these the sizes for the two levels of scratch space? One for
  // team-shared and one for thread-specific scratch space?
  size_t m_team_scratch_size[2];
  size_t m_thread_scratch_size[2];

  // TODO: What is the chunk size? What is getting chunked? Normally this is
  // loop iterations. Can we use the HPX chunkers (static_chunk_size). Doesn't
  // really make sense though for a team policy...? Or does it mean that
  // chunk_size teams will execute immediately after each other without the
  // runtime having to go and get the next team index.

  // NOTE: m_chunk_size contiguous iterations are guaranteed to run on the same
  // thread. I.e. with league_size = n, team_size = 1 the first m_chunk_size
  // ranks from league_size will run in a single task, the next m_chunk_size in
  // another etc. TODO: Need to make sure this is handled correctly in all
  // parallel functions.
  int m_chunk_size;

  using execution_policy = TeamPolicyInternal;
  using traits = PolicyTraits<Properties...>;

public:
  TeamPolicyInternal &operator=(const TeamPolicyInternal &p){};

  template <class FunctorType>
  inline static int team_size_max(const FunctorType &) {
    return 1;

    // NOTE: We return 1 for the team size for now (simpler and good enough in
    // most cases). We could increase team size to the number of hyperthreads at
    // most. Team size should not be the size of the thread pool.

    // return hpx::get_num_worker_threads();
  }

  template <class FunctorType>
  inline static int team_size_recommended(const FunctorType &) {
    return 1;
  }

  template <class FunctorType>
  inline static int team_size_recommended(const FunctorType &, const int &) {
    return 1;
  }

private:
  inline void init(const int league_size_request, const int team_size_request) {
    m_league_size = league_size_request;
    const int max_team_size = 1;
    m_team_size =
        team_size_request > max_team_size ? max_team_size : team_size_request;
    // TODO: Calculate auto chunk size like in OpenMP backend.
    m_chunk_size = 16;
  }

public:
  inline int team_size() const { return m_team_size; }
  inline int league_size() const { return m_league_size; }

  // TODO: Need to handle scratch space correctly. What is this supposed to
  // return? The scratch size of the given team on the given level? -1 means
  // shared scratch size? This is not part of the public API. This is just a
  // helper function.
  inline size_t scratch_size(const int &level, int team_size_ = -1) const {
    if (team_size_ < 0) {
      team_size_ = m_team_size;
    }
    return m_team_scratch_size[level] +
           team_size_ * m_thread_scratch_size[level];
  }

public:
  TeamPolicyInternal(typename traits::execution_space &,
                     int league_size_request, int team_size_request,
                     int /* vector_length_request */ = 1)
      : m_team_scratch_size{0, 0}, m_thread_scratch_size{0, 0},
        m_chunk_size(0) {
    init(league_size_request, team_size_request);
  }

  TeamPolicyInternal(typename traits::execution_space &,
                     int league_size_request,
                     const Kokkos::AUTO_t &team_size_request,
                     int /* vector_length_request */ = 1)
      : m_team_scratch_size{0, 0}, m_thread_scratch_size{0, 0},
        m_chunk_size(0) {
    init(league_size_request, 1);
  }

  TeamPolicyInternal(int league_size_request, int team_size_request,
                     int /* vector_length_request */ = 1)
      : m_team_scratch_size{0, 0}, m_thread_scratch_size{0, 0},
        m_chunk_size(0) {
    init(league_size_request, team_size_request);
  }

  TeamPolicyInternal(int league_size_request,
                     const Kokkos::AUTO_t &team_size_request,
                     int /* vector_length_request */ = 1)
      : m_team_scratch_size{0, 0}, m_thread_scratch_size{0, 0},
        m_chunk_size(0) {
    init(league_size_request, 1);
  }

  inline int chunk_size() const { return m_chunk_size; }

  inline TeamPolicyInternal &
  set_chunk_size(typename traits::index_type chunk_size_) {
    m_chunk_size = chunk_size_;
    return *this;
  }

  inline TeamPolicyInternal &set_scratch_size(const int &level,
                                              const PerTeamValue &per_team) {
    m_team_scratch_size[level] = per_team.value;
    return *this;
  }

  inline TeamPolicyInternal &
  set_scratch_size(const int &level, const PerThreadValue &per_thread) {
    m_thread_scratch_size[level] = per_thread.value;
    return *this;
  }

  inline TeamPolicyInternal &
  set_scratch_size(const int &level, const PerTeamValue &per_team,
                   const PerThreadValue &per_thread) {
    m_team_scratch_size[level] = per_team.value;
    m_thread_scratch_size[level] = per_thread.value;
    return *this;
  }

  typedef HPXTeamMember member_type;
};
} // namespace Impl
} // namespace Kokkos

namespace Kokkos {
namespace Impl {

template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::RangePolicy<Traits...>, Kokkos::HPX> {
private:
  typedef Kokkos::RangePolicy<Traits...> Policy;
  typedef typename Policy::work_tag WorkTag;
  typedef typename Policy::WorkRange WorkRange;
  typedef typename Policy::member_type Member;

  const FunctorType m_functor;
  const Policy m_policy;

  template <class TagType>
  inline static
      typename std::enable_if<std::is_same<TagType, void>::value>::type
      exec_functor(const FunctorType &functor, const Member iwork) {
    functor(iwork);
  }

  template <class TagType>
  inline static
      typename std::enable_if<!std::is_same<TagType, void>::value>::type
      exec_functor(const FunctorType &functor, const Member iwork) {
    const TagType t{};
    functor(t, iwork);
  }

public:
  inline void execute() const {
    auto f = [this]() {
      // TODO: hpx::get_num_worker_threads() is wrong. Should be pool threads.
      // TODO: Should restrict number of threads in for_loop.
      // auto const num_worker_threads = hpx::get_num_worker_threads();

      // This is the easy HPX way to do things.
      hpx::parallel::for_loop(hpx::parallel::execution::par.with(
                                  hpx::parallel::execution::guided_chunk_size(
                                      m_policy.chunk_size())),
                              m_policy.begin(), m_policy.end(),
                              [this](typename Policy::member_type const i) {
                                exec_functor<WorkTag>(m_functor, i);
                              });

      // This is the complicated, manual way to do things.
      // hpx::parallel::for_loop(
      //     hpx::parallel::execution::par, 0, num_worker_threads,
      //     [this, num_worker_threads](std::size_t const t) {
      //       // TODO: Use utilities that already exist.
      //       const typename Policy::member_type b = m_policy.begin();
      //       const typename Policy::member_type e = m_policy.end();
      //       const typename Policy::member_type n = e - b;
      //       const typename Policy::member_type chunk_size =
      //           (n - 1) / num_worker_threads + 1;

      //       const typename Policy::member_type b_local = b + t * chunk_size;
      //       Member e_local = b + (t + 1) * chunk_size;
      //       if (e_local > e) {
      //         e_local = e;
      //       };

      //       LOG("chunk range in thread " << hpx::get_worker_thread_num()
      //                                    << " is " << b_local << " to "
      //                                    << e_local);

      //       for (typename Policy::member_type i = b_local; i < e_local; ++i)
      //       {
      //         exec_functor<WorkTag>(m_functor, i);
      //       }
      //     });
    };

    if (hpx::threads::get_self_ptr()) {
      f();
    } else {
      hpx::threads::run_as_hpx_thread(f);
    }
  }

  inline ParallelFor(const FunctorType &arg_functor, Policy arg_policy)
      : m_functor(arg_functor), m_policy(arg_policy) {}
};

template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::MDRangePolicy<Traits...>, Kokkos::HPX> {
private:
  typedef Kokkos::MDRangePolicy<Traits...> MDRangePolicy;
  typedef typename MDRangePolicy::impl_range_policy Policy;
  typedef typename MDRangePolicy::work_tag WorkTag;
  typedef typename Policy::WorkRange WorkRange;
  typedef typename Policy::member_type Member;
  typedef typename Kokkos::Impl::HostIterateTile<
      MDRangePolicy, FunctorType, typename MDRangePolicy::work_tag, void>
      iterate_type;
  const FunctorType m_functor;
  const MDRangePolicy m_mdr_policy;
  const Policy m_policy; // construct as RangePolicy( 0, num_tiles
                         // ).set_chunk_size(1) in ctor

public:
  inline void execute() const {
    auto f = [this]() {
      // TODO: hpx::get_num_worker_threads() is wrong. Should be pool threads.
      // TODO: Should restrict number of threads in for_loop.
      // auto const num_worker_threads = hpx::get_num_worker_threads();

      // This is the easy HPX way to do things.
      hpx::parallel::for_loop(hpx::parallel::execution::par.with(
                                  hpx::parallel::execution::guided_chunk_size(
                                      m_policy.chunk_size())),
                              m_policy.begin(), m_policy.end(),
                              [this](typename Policy::member_type const i) {
                                iterate_type(m_mdr_policy, m_functor)(i);
                              });

      // This is the complicated, manual way to do things.
      // hpx::parallel::for_loop(
      //     hpx::parallel::execution::par, 0, num_worker_threads,
      //     [this, num_worker_threads](std::size_t const t) {
      //       // TODO: Use utilities that already exist.
      //       const typename Policy::member_type b = m_policy.begin();
      //       const typename Policy::member_type e = m_policy.end();
      //       const typename Policy::member_type n = e - b;
      //       const typename Policy::member_type chunk_size =
      //           (n - 1) / num_worker_threads + 1;

      //       const typename Policy::member_type b_local = b + t * chunk_size;
      //       Member e_local = b + (t + 1) * chunk_size;
      //       if (e_local > e) {
      //         e_local = e;
      //       };

      //       LOG("chunk range in thread " << hpx::get_worker_thread_num()
      //                                    << " is " << b_local << " to "
      //                                    << e_local);

      //       for (typename Policy::member_type i = b_local; i < e_local; ++i)
      //       {
      //         iterate_type(m_mdr_policy, m_functor)(i);
      //       }
      //     });
    };

    if (hpx::threads::get_self_ptr()) {
      f();
    } else {
      hpx::threads::run_as_hpx_thread(f);
    }
  }

  inline ParallelFor(const FunctorType &arg_functor, MDRangePolicy arg_policy)
      : m_functor(arg_functor), m_mdr_policy(arg_policy),
        m_policy(Policy(0, m_mdr_policy.m_num_tiles).set_chunk_size(1)) {}
}; // namespace Impl
} // namespace Impl
} // namespace Kokkos

namespace Kokkos {
namespace Impl {

// TODO: Could this be made to work with hpx::parallel::reduce?
template <class FunctorType, class ReducerType, class... Traits>
class ParallelReduce<FunctorType, Kokkos::RangePolicy<Traits...>, ReducerType,
                     Kokkos::HPX> {
private:
  typedef Kokkos::RangePolicy<Traits...> Policy;
  typedef typename Policy::work_tag WorkTag;
  typedef typename Policy::WorkRange WorkRange;
  typedef typename Policy::member_type Member;
  typedef FunctorAnalysis<FunctorPatternInterface::REDUCE, Policy, FunctorType>
      Analysis;
  typedef Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                             FunctorType, ReducerType>
      ReducerConditional;
  typedef typename ReducerConditional::type ReducerTypeFwd;
  typedef
      typename Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                                  WorkTag, void>::type WorkTagFwd;
  typedef Kokkos::Impl::FunctorValueInit<ReducerTypeFwd, WorkTagFwd> ValueInit;
  typedef Kokkos::Impl::FunctorValueJoin<ReducerTypeFwd, WorkTagFwd> ValueJoin;
  typedef typename Analysis::value_type value_type;
  typedef typename Analysis::pointer_type pointer_type;
  typedef typename Analysis::reference_type reference_type;

  const FunctorType m_functor;
  const Policy m_policy;
  const ReducerType m_reducer;
  const pointer_type m_result_ptr;

  template <class TagType>
  inline static
      typename std::enable_if<std::is_same<TagType, void>::value>::type
      exec_functor(const FunctorType &functor, const Member iwork,
                   reference_type update) {
    functor(iwork, update);
  }

  template <class TagType>
  inline static
      typename std::enable_if<!std::is_same<TagType, void>::value>::type
      exec_functor(const FunctorType &functor, const Member iwork,
                   reference_type update) {
    const TagType t{};
    functor(t, iwork, update);
  }

public:
  inline void execute() const {

    auto f = [this]() {
#if defined(KOKKOS_HPX_NATIVE_REDUCE) && (KOKKOS_HPX_NATIVE_REDUCE == 1)
    // TODO: Could this be made to work?
    // value_type starting_value;
    // reference_type starting_value_ref = ValueInit::init(
    //     ReducerConditional::select(m_functor, m_reducer), &starting_value);
    // hpx::parallel::reduce(
    //     hpx::parallel::execution::par, m_policy.begin(), m_policy.end(),
    //     [this](typename Policy::member_type const &x, typename
    //     Policy::member_type const &y) {
    //       value_type z;
    //       reference_type z_ref = ValueInit::init(
    //           ReducerConditional::select(m_functor, m_reducer), &z);
    //       exec_functor<WorkTag>(m_functor, x, z_ref);
    //       exec_functor<WorkTag>(m_functor, y, z_ref);
    //       return z;
    //     },
    //     starting_value_ref);

    // if (m_result_ptr != nullptr) {
    //   const int n = Analysis::value_count(
    //       ReducerConditional::select(m_functor, m_reducer));

    //   for (int j = 0; j < n; ++j) {
    //     m_result_ptr[j] = starting_value[j];
    //   }
    // }
#else
      // TODO: hpx::get_num_worker_threads() is wrong. Should be pool threads.
      auto const num_worker_threads = hpx::get_num_worker_threads();

      // This gets the size (in bytes) of the value we are reducing.
      const size_t value_size_bytes = Analysis::value_size(
          ReducerConditional::select(m_functor, m_reducer));

      // Need to get or allocate a pointer for the results. This would normally
      // come from an instance-specific scratch pool (only one operation at a
      // time). (HPXExec* m_instance)

      // NOTE: If we want to support multiple HPX backend parallel regions
      // running simultaneously we can't reuse the scratch space (without
      // additional checks).

      // NOTE: We can't reuse this one because there might not be enough space
      // for all threads. This only works on the serial backend.

      // ptr = m_result_ptr;

      std::vector<value_type> intermediate_results(num_worker_threads);

      hpx::parallel::execution::static_chunk_size s(1);
      hpx::parallel::for_loop(
          hpx::parallel::execution::par.with(s), 0, num_worker_threads,
          [this, &intermediate_results,
           num_worker_threads](std::size_t const t) {
            // This initializes the t:th reduction value to the appropriate
            // init value based on the functor.
            reference_type update = ValueInit::init(
                ReducerConditional::select(m_functor, m_reducer),
                (pointer_type)(&intermediate_results[t]));

            // TODO: Use utilities that already exist.
            const typename Policy::member_type b = m_policy.begin();
            const typename Policy::member_type e = m_policy.end();
            const typename Policy::member_type n = e - b;
            const typename Policy::member_type chunk_size =
                (n - 1) / num_worker_threads + 1;

            const typename Policy::member_type b_local = b + t * chunk_size;
            Member e_local = b + (t + 1) * chunk_size;
            if (e_local > e) {
              e_local = e;
            };

            LOG("chunk range in thread " << hpx::get_worker_thread_num()
                                         << " is " << b_local << " to "
                                         << e_local);

            for (typename Policy::member_type i = b_local; i < e_local; ++i) {
              exec_functor<WorkTag>(m_functor, i, update);
            }

            LOG("intermediate result from worker thread "
                << hpx::get_worker_thread_num() << " is " << update);
          });

      LOG("reduced value from worker thread " << 0 << " is "
                                              << intermediate_results[0]);
      for (int i = 1; i < num_worker_threads; ++i) {
        LOG("reduced value from worker thread " << i << " is "
                                                << intermediate_results[i]);
        ValueJoin::join(ReducerConditional::select(m_functor, m_reducer),
                        &intermediate_results[0], &intermediate_results[i]);
      }

      Kokkos::Impl::FunctorFinal<ReducerTypeFwd, WorkTagFwd>::final(
          ReducerConditional::select(m_functor, m_reducer),
          &intermediate_results[0]);

      LOG("final reduced value is " << intermediate_results[0]);

      if (m_result_ptr != nullptr) {
        const int n = Analysis::value_count(
            ReducerConditional::select(m_functor, m_reducer));

        for (int j = 0; j < n; ++j) {
          // Elements could be either scalar values or arrays. So we have to get
          // the pointer to the underlying storage, and dereference it with the
          // subscript operator. When the value is a scalar n == 1 and we will
          // only get the first element from intermediate_results.
          m_result_ptr[j] = intermediate_results.data()[j];
        }

        LOG("final reduced value in result_ptr is " << m_result_ptr[0]);
      }
#endif
    };

    if (hpx::threads::get_self_ptr()) {
      f();
    } else {
      hpx::threads::run_as_hpx_thread(f);
    }
  }

  template <class ViewType>
  inline ParallelReduce(
      const FunctorType &arg_functor, Policy arg_policy,
      const ViewType &arg_view,
      typename std::enable_if<Kokkos::is_view<ViewType>::value &&
                                  !Kokkos::is_reducer_type<ReducerType>::value,
                              void *>::type = NULL)
      : m_functor(arg_functor), m_policy(arg_policy), m_reducer(InvalidType()),
        m_result_ptr(arg_view.data()) {}

  inline ParallelReduce(const FunctorType &arg_functor, Policy arg_policy,
                        const ReducerType &reducer)
      : m_functor(arg_functor), m_policy(arg_policy), m_reducer(reducer),
        m_result_ptr(reducer.view().data()) {}
};

// TODO: Could this be made to work with hpx::parallel::reduce?
template <class FunctorType, class ReducerType, class... Traits>
class ParallelReduce<FunctorType, Kokkos::MDRangePolicy<Traits...>, ReducerType,
                     Kokkos::HPX> {
private:
  typedef Kokkos::MDRangePolicy<Traits...> MDRangePolicy;
  typedef typename MDRangePolicy::impl_range_policy Policy;
  typedef typename MDRangePolicy::work_tag WorkTag;
  typedef typename Policy::WorkRange WorkRange;
  typedef typename Policy::member_type Member;
  typedef FunctorAnalysis<FunctorPatternInterface::REDUCE, MDRangePolicy,
                          FunctorType>
      Analysis;
  typedef Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                             FunctorType, ReducerType>
      ReducerConditional;
  typedef typename ReducerConditional::type ReducerTypeFwd;
  typedef
      typename Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                                  WorkTag, void>::type WorkTagFwd;
  typedef Kokkos::Impl::FunctorValueInit<ReducerTypeFwd, WorkTagFwd> ValueInit;
  typedef Kokkos::Impl::FunctorValueJoin<ReducerTypeFwd, WorkTagFwd> ValueJoin;
  typedef typename Analysis::pointer_type pointer_type;
  typedef typename Analysis::value_type value_type;
  typedef typename Analysis::reference_type reference_type;
  using iterate_type =
      typename Kokkos::Impl::HostIterateTile<MDRangePolicy, FunctorType,
                                             WorkTag, reference_type>;
  HPXExec *m_instance;
  const FunctorType m_functor;
  const MDRangePolicy m_mdr_policy;
  const Policy m_policy; // construct as RangePolicy( 0, num_tiles
                         // ).set_chunk_size(1) in ctor
  const ReducerType m_reducer;
  const pointer_type m_result_ptr;

  template <class TagType>
  inline static
      typename std::enable_if<std::is_same<TagType, void>::value>::type
      exec_functor(const FunctorType &functor, const Member iwork,
                   reference_type update) {
    functor(iwork, update);
  }

  template <class TagType>
  inline static
      typename std::enable_if<!std::is_same<TagType, void>::value>::type
      exec_functor(const FunctorType &functor, const Member iwork,
                   reference_type update) {
    const TagType t{};
    functor(t, iwork, update);
  }

public:
  inline void execute() const {
    auto f = [this]() {
      // TODO: hpx::get_num_worker_threads() is wrong. Should be pool
      // threads.
      auto const num_worker_threads = hpx::get_num_worker_threads();

      // This gets the size (in bytes) of the value we are reducing.
      const size_t value_size_bytes = Analysis::value_size(
          ReducerConditional::select(m_functor, m_reducer));

      // Need to get or allocate a pointer for the results. This would
      // normally come from an instance-specific scratch pool (only one
      // operation at a time). (HPXExec* m_instance)

      // NOTE: If we want to support multiple HPX backend parallel regions
      // running simultaneously we can't reuse the scratch space (without
      // additional checks).

      // NOTE: We can't reuse this one because there might not be enough
      // space for all threads. This only works on the serial backend.

      // ptr = m_result_ptr;

      std::vector<value_type> intermediate_results(num_worker_threads);

      hpx::parallel::execution::static_chunk_size s(1);
      hpx::parallel::for_loop(
          hpx::parallel::execution::par.with(s), 0, num_worker_threads,
          [this, &intermediate_results,
           num_worker_threads](std::size_t const t) {
            // This initializes the t:th reduction value to the appropriate
            // init value based on the functor.
            reference_type update = ValueInit::init(
                ReducerConditional::select(m_functor, m_reducer),
                (pointer_type)(&intermediate_results[t]));

            // TODO: Use utilities that already exist.
            const typename Policy::member_type b = m_policy.begin();
            const typename Policy::member_type e = m_policy.end();
            const typename Policy::member_type n = e - b;
            const typename Policy::member_type chunk_size =
                (n - 1) / num_worker_threads + 1;

            const typename Policy::member_type b_local = b + t * chunk_size;
            Member e_local = b + (t + 1) * chunk_size;
            if (e_local > e) {
              e_local = e;
            };

            LOG("chunk range in thread " << hpx::get_worker_thread_num()
                                         << " is " << b_local << " to "
                                         << e_local);

            for (typename Policy::member_type i = b_local; i < e_local; ++i) {
              exec_functor<WorkTag>(m_functor, i, update);
            }

            LOG("intermediate result from worker thread "
                << hpx::get_worker_thread_num() << " is " << update);
          });

      LOG("reduced value from worker thread " << 0 << " is "
                                              << intermediate_results[0]);
      for (int i = 1; i < num_worker_threads; ++i) {
        LOG("reduced value from worker thread " << i << " is "
                                                << intermediate_results[i]);
        ValueJoin::join(ReducerConditional::select(m_functor, m_reducer),
                        &intermediate_results[0], &intermediate_results[i]);
      }

      Kokkos::Impl::FunctorFinal<ReducerTypeFwd, WorkTagFwd>::final(
          ReducerConditional::select(m_functor, m_reducer),
          &intermediate_results[0]);

      LOG("final reduced value is " << intermediate_results[0]);

      if (m_result_ptr != nullptr) {
        const int n = Analysis::value_count(
            ReducerConditional::select(m_functor, m_reducer));

        for (int j = 0; j < n; ++j) {
          // Elements could be either scalar values or arrays. So we have to
          // get the pointer to the underlying storage, and dereference it
          // with the subscript operator. When the value is a scalar n == 1
          // and we will only get the first element from
          // intermediate_results.
          m_result_ptr[j] = intermediate_results.data()[j];
        }

        LOG("final reduced value in result_ptr is " << m_result_ptr[0]);
      }
    };

    if (hpx::threads::get_self_ptr()) {
      f();
    } else {
      hpx::threads::run_as_hpx_thread(f);
    }
  }

  template <class ViewType>
  inline ParallelReduce(
      const FunctorType &arg_functor, MDRangePolicy arg_policy,
      const ViewType &arg_view,
      typename std::enable_if<Kokkos::is_view<ViewType>::value &&
                                  !Kokkos::is_reducer_type<ReducerType>::value,
                              void *>::type = NULL)
      : m_functor(arg_functor), m_mdr_policy(arg_policy),
        m_policy(Policy(0, m_mdr_policy.m_num_tiles).set_chunk_size(1)),
        m_reducer(InvalidType()), m_result_ptr(arg_view.data()) {}

  inline ParallelReduce(const FunctorType &arg_functor,
                        MDRangePolicy arg_policy, const ReducerType &reducer)
      : m_functor(arg_functor), m_mdr_policy(arg_policy),
        m_policy(Policy(0, m_mdr_policy.m_num_tiles).set_chunk_size(1)),
        m_reducer(reducer), m_result_ptr(reducer.view().data()) {}
};
} // namespace Impl
} // namespace Kokkos

namespace Kokkos {
namespace Impl {

template <class FunctorType, class... Traits>
class ParallelScan<FunctorType, Kokkos::RangePolicy<Traits...>, Kokkos::HPX> {
private:
  typedef Kokkos::RangePolicy<Traits...> Policy;
  typedef typename Policy::work_tag WorkTag;
  typedef typename Policy::member_type Member;

  typedef FunctorAnalysis<FunctorPatternInterface::SCAN, Policy, FunctorType>
      Analysis;

  typedef Kokkos::Impl::FunctorValueInit<FunctorType, WorkTag> ValueInit;
  typedef Kokkos::Impl::FunctorValueJoin<FunctorType, WorkTag> ValueJoin;
  typedef Kokkos::Impl::FunctorValueOps<FunctorType, WorkTag> ValueOps;

  typedef typename Analysis::pointer_type pointer_type;
  typedef typename Analysis::reference_type reference_type;
  typedef typename Analysis::value_type value_type;

  const FunctorType m_functor;
  const Policy m_policy;

  template <class TagType>
  inline static
      typename std::enable_if<std::is_same<TagType, void>::value>::type
      exec_functor(const FunctorType &functor, const Member iwork,
                   reference_type update, const bool final) {
    functor(iwork, update, final);
  }

  template <class TagType>
  inline static
      typename std::enable_if<!std::is_same<TagType, void>::value>::type
      exec_functor(const FunctorType &functor, const Member iwork,
                   reference_type update, const bool final) {
    const TagType t{};
    functor(t, iwork, update, final);
  }

public:
  inline void execute() const {
    auto f = [this]() {
      // TODO: hpx::get_num_worker_threads() is wrong. Should be pool threads.
      // TODO: Should be bigger than pool threads (for uneven chunk sizes).
      auto const num_worker_threads = hpx::get_num_worker_threads();

      // This gets the size (in bytes) of the value we are reducing.
      const int value_count = Analysis::value_count(m_functor);
      const size_t value_size_bytes = 2 * Analysis::value_size(m_functor);

      // Need to get or allocate a pointer for the results. This would
      // normally come from an instance-specific scratch pool (only one
      // operation at a time). (HPXExec* m_instance)

      // NOTE: If we want to support multiple HPX backend parallel regions
      // running simultaneously we can't reuse the scratch space (without
      // additional checks).

      // NOTE: We can't reuse this one because there might not be enough
      // space for all threads. This only works on the serial backend.

      // ptr = m_result_ptr;

      std::vector<value_type> intermediate_results(num_worker_threads);
      std::vector<value_type> intermediate_results2(num_worker_threads); // ??

      //  NOTE: This structure is copied straight from the OpenMP backend. This
      //  is not necessarily the fastest or most elegant way of writing this.
      hpx::lcos::local::barrier barrier(num_worker_threads);
      hpx::parallel::execution::static_chunk_size s(1);

      hpx::parallel::for_loop(
          hpx::parallel::execution::par.with(s), 0, num_worker_threads,
          [this, &intermediate_results, &intermediate_results2, &barrier,
           num_worker_threads, value_count](std::size_t const t) {
            // This initializes the t:th reduction value to the appropriate
            // init value based on the functor.
            reference_type update_sum = ValueInit::init(
                m_functor, (pointer_type)(&intermediate_results[t]));

            // TODO: Use utilities that already exist.
            const typename Policy::member_type b = m_policy.begin();
            const typename Policy::member_type e = m_policy.end();
            const typename Policy::member_type n = e - b;
            const typename Policy::member_type chunk_size =
                (n - 1) / num_worker_threads + 1;

            const typename Policy::member_type b_local = b + t * chunk_size;
            Member e_local = b + (t + 1) * chunk_size;
            if (e_local > e) {
              e_local = e;
            };

            std::cerr << "chunk range in thread "
                      << hpx::get_worker_thread_num() << " is " << b_local
                      << " to " << e_local << std::endl;

            for (typename Policy::member_type i = b_local; i < e_local; ++i) {
              exec_functor<WorkTag>(m_functor, i, update_sum, false);
            }

            barrier.wait();

            // TODO: Use call_once?
            if (t == 0) {
              ValueInit::init(m_functor,
                              (pointer_type)(&intermediate_results2[0]));

              for (int i = 1; i < num_worker_threads; ++i) {
                pointer_type ptr_prev =
                    (pointer_type)(&intermediate_results[i - 1]);
                pointer_type ptr2_prev =
                    (pointer_type)(&intermediate_results2[i - 1]);

                pointer_type ptr = (pointer_type)(&intermediate_results[i]);
                pointer_type ptr2 = (pointer_type)(&intermediate_results2[i]);

                for (int j = 0; j < value_count; ++j) {
                  ptr2[j] = ptr2_prev[j];
                }

                ValueJoin::join(m_functor, ptr2, ptr_prev);
              }
            }

            barrier.wait();

            reference_type update_base =
                ValueOps::reference((pointer_type)(&intermediate_results2[t]));

            for (typename Policy::member_type i = b_local; i < e_local; ++i) {
              exec_functor<WorkTag>(m_functor, i, update_base, true);
            }
          });

      LOG("reduced value from worker thread " << 0 << " is "
                                              << intermediate_results[0]);
      for (int i = 1; i < num_worker_threads; ++i) {
        LOG("reduced value from worker thread " << i << " is "
                                                << intermediate_results[i]);
      }
    };

    if (hpx::threads::get_self_ptr()) {
      f();
    } else {
      hpx::threads::run_as_hpx_thread(f);
    }
  }

  inline ParallelScan(const FunctorType &arg_functor, const Policy &arg_policy)
      : m_functor(arg_functor), m_policy(arg_policy) {}
};

// ParallelScanWithTotal returns the final total (instead of void).
template <class FunctorType, class ReturnType, class... Traits>
class ParallelScanWithTotal<FunctorType, Kokkos::RangePolicy<Traits...>,
                            ReturnType, Kokkos::HPX> {
private:
  typedef Kokkos::RangePolicy<Traits...> Policy;
  typedef typename Policy::work_tag WorkTag;
  typedef typename Policy::member_type Member;

  typedef FunctorAnalysis<FunctorPatternInterface::SCAN, Policy, FunctorType>
      Analysis;

  typedef Kokkos::Impl::FunctorValueInit<FunctorType, WorkTag> ValueInit;
  typedef Kokkos::Impl::FunctorValueJoin<FunctorType, WorkTag> ValueJoin;
  typedef Kokkos::Impl::FunctorValueOps<FunctorType, WorkTag> ValueOps;

  typedef typename Analysis::pointer_type pointer_type;
  typedef typename Analysis::reference_type reference_type;
  typedef typename Analysis::value_type value_type;

  const FunctorType m_functor;
  const Policy m_policy;
  ReturnType &m_returnvalue;

  template <class TagType>
  inline static
      typename std::enable_if<std::is_same<TagType, void>::value>::type
      exec_functor(const FunctorType &functor, const Member iwork,
                   reference_type update, const bool final) {
    functor(iwork, update, final);
  }

  template <class TagType>
  inline static
      typename std::enable_if<!std::is_same<TagType, void>::value>::type
      exec_functor(const FunctorType &functor, const Member iwork,
                   reference_type update, const bool final) {
    const TagType t{};
    functor(t, iwork, update, final);
  }

public:
  inline void execute() const {
    auto f = [this]() {
      // TODO: hpx::get_num_worker_threads() is wrong. Should be pool threads.
      // TODO: Should be bigger than pool threads (for uneven chunk sizes).
      auto const num_worker_threads = hpx::get_num_worker_threads();

      // This gets the size (in bytes) of the value we are reducing.
      const int value_count = Analysis::value_count(m_functor);
      const size_t value_size_bytes = 2 * Analysis::value_size(m_functor);

      // Need to get or allocate a pointer for the results. This would
      // normally come from an instance-specific scratch pool (only one
      // operation at a time). (HPXExec* m_instance)

      // NOTE: If we want to support multiple HPX backend parallel regions
      // running simultaneously we can't reuse the scratch space (without
      // additional checks).

      // NOTE: We can't reuse this one because there might not be enough
      // space for all threads. This only works on the serial backend.

      // ptr = m_result_ptr;

      std::vector<value_type> intermediate_results(num_worker_threads);
      std::vector<value_type> intermediate_results2(num_worker_threads); // ??

      //  NOTE: This structure is copied straight from the OpenMP backend. This
      //  is not necessarily the fastest or most elegant way of writing this.
      hpx::lcos::local::barrier barrier(num_worker_threads);
      hpx::parallel::execution::static_chunk_size s(1);

      hpx::parallel::for_loop(
          hpx::parallel::execution::par.with(s), 0, num_worker_threads,
          [this, &intermediate_results, &intermediate_results2, &barrier,
           num_worker_threads, value_count](std::size_t const t) {
            // This initializes the t:th reduction value to the appropriate
            // init value based on the functor.
            reference_type update_sum = ValueInit::init(
                m_functor, (pointer_type)(&intermediate_results[t]));

            // TODO: Use utilities that already exist.
            const typename Policy::member_type b = m_policy.begin();
            const typename Policy::member_type e = m_policy.end();
            const typename Policy::member_type n = e - b;
            const typename Policy::member_type chunk_size =
                (n - 1) / num_worker_threads + 1;

            const typename Policy::member_type b_local = b + t * chunk_size;
            Member e_local = b + (t + 1) * chunk_size;
            if (e_local > e) {
              e_local = e;
            };

            LOG("chunk range in thread " << hpx::get_worker_thread_num()
                                         << " is " << b_local << " to "
                                         << e_local);

            for (typename Policy::member_type i = b_local; i < e_local; ++i) {
              exec_functor<WorkTag>(m_functor, i, update_sum, false);
            }

            barrier.wait();

            if (t == 0) {
              ValueInit::init(m_functor,
                              (pointer_type)(&intermediate_results2[0]));

              for (int i = 1; i < num_worker_threads; ++i) {
                pointer_type ptr_prev =
                    (pointer_type)(&intermediate_results[i - 1]);
                pointer_type ptr2_prev =
                    (pointer_type)(&intermediate_results2[i - 1]);

                pointer_type ptr = (pointer_type)(&intermediate_results[i]);
                pointer_type ptr2 = (pointer_type)(&intermediate_results2[i]);

                for (int j = 0; j < value_count; ++j) {
                  ptr2[j] = ptr2_prev[j];
                }

                ValueJoin::join(m_functor, ptr2, ptr_prev);
              }
            }

            barrier.wait();

            reference_type update_base =
                ValueOps::reference((pointer_type)(&intermediate_results2[t]));

            for (typename Policy::member_type i = b_local; i < e_local; ++i) {
              exec_functor<WorkTag>(m_functor, i, update_base, true);
            }

            if (t == num_worker_threads - 1) {
              m_returnvalue = update_base;
            }
          });

      LOG("reduced value from worker thread " << 0 << " is "
                                              << intermediate_results[0]);
      for (int i = 1; i < num_worker_threads; ++i) {
        LOG("reduced value from worker thread " << i << " is "
                                                << intermediate_results[i]);
      }
    };

    if (hpx::threads::get_self_ptr()) {
      f();
    } else {
      hpx::threads::run_as_hpx_thread(f);
    }
  }

  inline ParallelScanWithTotal(const FunctorType &arg_functor,
                               const Policy &arg_policy,
                               ReturnType &arg_returnvalue)
      : m_functor(arg_functor), m_policy(arg_policy),
        m_returnvalue(arg_returnvalue) {}
};
} // namespace Impl
} // namespace Kokkos

namespace Kokkos {
namespace Impl {
// TODO
template <class FunctorType, class... Properties>
class ParallelFor<FunctorType, Kokkos::TeamPolicy<Properties...>, Kokkos::HPX> {
private:
  // TODO: Should not be needed. Only used to define a local constant.
  // enum { TEAM_REDUCE_SIZE = 512 };

  typedef TeamPolicyInternal<Kokkos::HPX, Properties...> Policy;
  typedef typename Policy::work_tag WorkTag;
  typedef typename Policy::member_type Member;
  typedef Kokkos::HostSpace memory_space;

  const FunctorType m_functor;
  const Policy m_policy;
  const int m_league;
  const int m_shared; // NOTE: Is this the size of shared scratch memory? Yes.
                      // But per team? Per thread?

  template <class TagType>
  inline static
      typename std::enable_if<std::is_same<TagType, void>::value>::type
      exec_functor(const FunctorType &functor, Member &&member) {
    functor(member);
  }

  template <class TagType>
  inline static
      typename std::enable_if<!std::is_same<TagType, void>::value>::type
      exec_functor(const FunctorType &functor, Member &&member) {
    const TagType t{};
    functor(t, member);
  }

public:
  inline void execute() const {
    // Kokkos::abort("ParallelFor<TeamPolicy>");
    auto f = [this]() {
      // TODO: Should distribute single teams onto NUMA domains.
      // TODO: hpx::get_num_worker_threads() is wrong. Should be pool threads.
      auto const team_size = m_policy.team_size();

      // TODO: What is team_iter? Maximum number of iterations each team will
      // take...?

      // if (m_policy.team_iter() < team_size) {
      //   team_size = m_policy.team_iter();
      // }
      auto const league_size = m_policy.league_size();

      auto hpx_policy = hpx::parallel::execution::par.with(
          hpx::parallel::execution::static_chunk_size(1));

      // TODO: Would like to use organize_team here to set up ... but it yields
      // OS threads. Is there a way to use HPX synchronization primitives
      // instead? Yes. Previous comment not true. Only OpenMP and Serial
      // backends use HostThreadTeamData. We will use HPXTeamMember.

      // TODO: Allocate scratch space: shared, team and thread. Should allocate
      // separate team scratch memories for each thread team so that multiple
      // teams and parallel for loops can be in flight at the same time.

      // TODO: Should allocate thread data per thread? Necessary? Should
      // allocate per NUMA domain.

      // TODO: Bogus value. How much space do I actually need to allocate?
      // team_shared_mem_size + team_size * thread_mem_size + ???.
      // const int scratch_size = 123123;
      // TODO: Wrap in smart pointer with custom deleter.
      // void *scratch = space.allocate();
      // std::unique_ptr<> scratch(space.allocate(m_policy.scratch_size(0)),
      //                           [space](void *ptr) { space.deallocate(ptr);
      //                           });

      // TODO: A better alternative might be:
      // for (0..league_size) {
      //   for (0..team_size) {
      //     futures.push_back(hpx::async(blah, team_rank, league_rank)):
      //   }
      // }
      // wait_all(futures);
      //
      // Otherwise the outer loop doesn't do much work.

      // May have to limit how many teams can be in flight simultaneously.
      // Potentially requires a lot of memory if the league size is large. More
      // than one team in flight can be beneficial if iterations differ in
      // length, but too many will only waste memory.

      // There is no memory shared across all teams. If that's needed simply
      // allocate a view before starting the parallel region.

      hpx::parallel::for_loop(
          hpx_policy, 0, league_size,
          [this, league_size, team_size,
           hpx_policy](std::size_t const league_rank) {
            std::shared_ptr<hpx::lcos::local::barrier> team_barrier(
                new hpx::lcos::local::barrier(team_size));
            // TODO: These memory chunks could all be allocated together.

            // This memory is shared across all threads in a team. This one is
            // used for user-facing scratch memory. Need to have two levels even
            // though the memory comes from the same hierarchy. The two levels
            // have to at least point to different memory.
            // m_policy.scratch_size(level) returns team_shared_bytes +
            // team_size * thread_local_bytes.
            const int scratch_size =
                m_policy.scratch_size(0) + m_policy.scratch_size(1);
            std::vector<char> scratch(scratch_size);
            char *scratch_data = scratch.data();

            printf("ParallelFor<TeamPolicy>: league_rank = %ul, scratch_data = "
                   "%p, scratch_size = %d\n",
                   league_rank, scratch_data, scratch_size);

            // It's allowed to do a team_reduce/team_scan inside a parallel_for.
            // So we need to allocate temporary memory for a reduction. This
            // memory is shared across all threads in a team. We have a fixed
            // upper limit on the space allocated for a reduction.
            const std::size_t max_thread_reduce_bytes = 512; // TODO: Is this
                                                             // enough? Dynamic?
                                                             // Move to class
                                                             // scope.
            const int reduce_size =
                max_thread_reduce_bytes * m_policy.team_size();
            std::vector<char> reduce_buffer(reduce_size);
            char *reduce_buffer_data = reduce_buffer.data();

            hpx::parallel::for_loop(
                hpx_policy, 0, team_size,
                [this, league_rank, league_size, team_size, scratch_data,
                 scratch_size, reduce_buffer_data, reduce_size,
                 team_barrier](std::size_t const team_rank) {
                  exec_functor<WorkTag>(m_functor,
                                        Member(m_policy, team_rank, league_rank,
                                               scratch_data, scratch_size,
                                               reduce_buffer_data, reduce_size,
                                               team_barrier));
                });
          });
    };

    if (hpx::threads::get_self_ptr()) {
      f();
    } else {
      hpx::threads::run_as_hpx_thread(f);
    }
  }

  ParallelFor(const FunctorType &arg_functor, const Policy &arg_policy)
      : m_functor(arg_functor), m_policy(arg_policy),
        m_league(arg_policy.league_size()),
        m_shared(arg_policy.scratch_size(0) + arg_policy.scratch_size(1) +
                 FunctorTeamShmemSize<FunctorType>::value(arg_functor, 1)) {}
};

// TODO
template <class FunctorType, class ReducerType, class... Properties>
class ParallelReduce<FunctorType, Kokkos::TeamPolicy<Properties...>,
                     ReducerType, Kokkos::HPX> {
private:
  // TODO: Should not be needed. Only used to define a local constant. But...
  // may be needed for old code. We apparently set a hard maximum size on some
  // of the scratch space. Why? enum { TEAM_REDUCE_SIZE = 512 };

  typedef TeamPolicyInternal<Kokkos::HPX, Properties...> Policy;

  typedef FunctorAnalysis<FunctorPatternInterface::REDUCE, Policy, FunctorType>
      Analysis;

  typedef typename Policy::member_type Member;
  typedef typename Policy::work_tag WorkTag;

  typedef Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                             FunctorType, ReducerType>
      ReducerConditional;
  typedef typename ReducerConditional::type ReducerTypeFwd;
  typedef
      typename Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                                  WorkTag, void>::type WorkTagFwd;

  typedef Kokkos::Impl::FunctorValueInit<ReducerTypeFwd, WorkTagFwd> ValueInit;

  typedef typename Analysis::pointer_type pointer_type;
  typedef typename Analysis::reference_type reference_type;

  const FunctorType m_functor;
  const int m_league;
  const ReducerType m_reducer;
  pointer_type m_result_ptr;
  const int m_shared;

public:
  inline void execute() const {
    Kokkos::abort("ParallelReduce<TeamPolicy>: execute\n");
  }

  template <class ViewType>
  ParallelReduce(
      const FunctorType &arg_functor, const Policy &arg_policy,
      const ViewType &arg_result,
      typename std::enable_if<Kokkos::is_view<ViewType>::value &&
                                  !Kokkos::is_reducer_type<ReducerType>::value,
                              void *>::type = NULL)
      : m_functor(arg_functor), m_league(arg_policy.league_size()),
        m_reducer(InvalidType()), m_result_ptr(arg_result.data()),
        m_shared(arg_policy.scratch_size(0) + arg_policy.scratch_size(1) +
                 FunctorTeamShmemSize<FunctorType>::value(m_functor, 1)) {}

  inline ParallelReduce(const FunctorType &arg_functor, Policy arg_policy,
                        const ReducerType &reducer)
      : m_functor(arg_functor), m_league(arg_policy.league_size()),
        m_reducer(reducer), m_result_ptr(reducer.view().data()),
        m_shared(arg_policy.scratch_size(0) + arg_policy.scratch_size(1) +
                 FunctorTeamShmemSize<FunctorType>::value(arg_functor, 1)) {}
};
} // namespace Impl
} // namespace Kokkos

namespace Kokkos {

template <typename iType>
KOKKOS_INLINE_FUNCTION
    Impl::TeamThreadRangeBoundariesStruct<iType, Impl::HPXTeamMember>
    TeamThreadRange(const Impl::HPXTeamMember &thread, const iType &count) {
  return Impl::TeamThreadRangeBoundariesStruct<iType, Impl::HPXTeamMember>(
      thread, count);
}

template <typename iType1, typename iType2>
KOKKOS_INLINE_FUNCTION Impl::TeamThreadRangeBoundariesStruct<
    typename std::common_type<iType1, iType2>::type, Impl::HPXTeamMember>
TeamThreadRange(const Impl::HPXTeamMember &thread, const iType1 &begin,
                const iType2 &end) {
  typedef typename std::common_type<iType1, iType2>::type iType;
  return Impl::TeamThreadRangeBoundariesStruct<iType, Impl::HPXTeamMember>(
      thread, iType(begin), iType(end));
}

template <typename iType>
KOKKOS_INLINE_FUNCTION
    Impl::ThreadVectorRangeBoundariesStruct<iType, Impl::HPXTeamMember>
    ThreadVectorRange(const Impl::HPXTeamMember &thread, const iType &count) {
  return Impl::ThreadVectorRangeBoundariesStruct<iType, Impl::HPXTeamMember>(
      thread, count);
}

template <typename iType>
KOKKOS_INLINE_FUNCTION
    Impl::ThreadVectorRangeBoundariesStruct<iType, Impl::HPXTeamMember>
    ThreadVectorRange(const Impl::HPXTeamMember &thread, const iType &arg_begin,
                      const iType &arg_end) {
  return Impl::ThreadVectorRangeBoundariesStruct<iType, Impl::HPXTeamMember>(
      thread, arg_begin, arg_end);
}

KOKKOS_INLINE_FUNCTION
Impl::ThreadSingleStruct<Impl::HPXTeamMember>
PerTeam(const Impl::HPXTeamMember &thread) {
  return Impl::ThreadSingleStruct<Impl::HPXTeamMember>(thread);
}

KOKKOS_INLINE_FUNCTION
Impl::VectorSingleStruct<Impl::HPXTeamMember>
PerThread(const Impl::HPXTeamMember &thread) {
  return Impl::VectorSingleStruct<Impl::HPXTeamMember>(thread);
}

/** \brief  Inter-thread parallel_for. Executes lambda(iType i) for each
 * i=0..N-1.
 *
 * The range i=0..N-1 is mapped to all threads of the the calling thread team.
 * This functionality requires C++11 support.*/
template <typename iType, class Lambda>
KOKKOS_INLINE_FUNCTION void parallel_for(
    const Impl::TeamThreadRangeBoundariesStruct<iType, Impl::HPXTeamMember>
        &loop_boundaries,
    const Lambda &lambda) {
  for (iType i = loop_boundaries.start; i < loop_boundaries.end;
       i += loop_boundaries.increment)
    lambda(i);
}

/** \brief  Inter-thread vector parallel_reduce. Executes lambda(iType i,
 * ValueType & val) for each i=0..N-1.
 *
 * The range i=0..N-1 is mapped to all threads of the the calling thread team
 * and a summation of val is performed and put into result. This functionality
 * requires C++11 support.*/
template <typename iType, class Lambda, typename ValueType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::TeamThreadRangeBoundariesStruct<iType, Impl::HPXTeamMember>
        &loop_boundaries,
    const Lambda &lambda, ValueType &result) {

  result = ValueType();

  for (iType i = loop_boundaries.start; i < loop_boundaries.end;
       i += loop_boundaries.increment) {
    ValueType tmp = ValueType();
    lambda(i, tmp);
    result += tmp;
  }

  result =
      loop_boundaries.thread.team_reduce(result, Impl::JoinAdd<ValueType>());
}

/** \brief  Intra-thread vector parallel_reduce. Executes lambda(iType i,
 * ValueType & val) for each i=0..N-1.
 *
 * The range i=0..N-1 is mapped to all vector lanes of the the calling thread
 * and a reduction of val is performed using JoinType(ValueType& val, const
 * ValueType& update) and put into init_result. The input value of init_result
 * is used as initializer for temporary variables of ValueType. Therefore the
 * input value should be the neutral element with respect to the join operation
 * (e.g. '0 for +-' or '1 for *'). This functionality requires C++11 support.*/
template <typename iType, class Lambda, typename ValueType, class JoinType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::TeamThreadRangeBoundariesStruct<iType, Impl::HPXTeamMember>
        &loop_boundaries,
    const Lambda &lambda, const JoinType &join, ValueType &init_result) {

  ValueType result = init_result;

  for (iType i = loop_boundaries.start; i < loop_boundaries.end;
       i += loop_boundaries.increment) {
    ValueType tmp = ValueType();
    lambda(i, tmp);
    join(result, tmp);
  }

  init_result = loop_boundaries.thread.team_reduce(
      result, Impl::JoinLambdaAdapter<ValueType, JoinType>(join));
}

/** \brief  Intra-thread vector parallel_for. Executes lambda(iType i) for each
 * i=0..N-1.
 *
 * The range i=0..N-1 is mapped to all vector lanes of the the calling thread.
 * This functionality requires C++11 support.*/
template <typename iType, class Lambda>
KOKKOS_INLINE_FUNCTION void parallel_for(
    const Impl::ThreadVectorRangeBoundariesStruct<iType, Impl::HPXTeamMember>
        &loop_boundaries,
    const Lambda &lambda) {
#ifdef KOKKOS_ENABLE_PRAGMA_IVDEP
#pragma ivdep
#endif
  for (iType i = loop_boundaries.start; i < loop_boundaries.end;
       i += loop_boundaries.increment)
    lambda(i);
}

/** \brief  Intra-thread vector parallel_reduce. Executes lambda(iType i,
 * ValueType & val) for each i=0..N-1.
 *
 * The range i=0..N-1 is mapped to all vector lanes of the the calling thread
 * and a summation of val is performed and put into result. This functionality
 * requires C++11 support.*/
template <typename iType, class Lambda, typename ValueType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::ThreadVectorRangeBoundariesStruct<iType, Impl::HPXTeamMember>
        &loop_boundaries,
    const Lambda &lambda, ValueType &result) {
  result = ValueType();
#ifdef KOKKOS_ENABLE_PRAGMA_IVDEP
#pragma ivdep
#endif
  for (iType i = loop_boundaries.start; i < loop_boundaries.end;
       i += loop_boundaries.increment) {
    ValueType tmp = ValueType();
    lambda(i, tmp);
    result += tmp;
  }
}

/** \brief  Intra-thread vector parallel_reduce. Executes lambda(iType i,
 * ValueType & val) for each i=0..N-1.
 *
 * The range i=0..N-1 is mapped to all vector lanes of the the calling thread
 * and a reduction of val is performed using JoinType(ValueType& val, const
 * ValueType& update) and put into init_result. The input value of init_result
 * is used as initializer for temporary variables of ValueType. Therefore the
 * input value should be the neutral element with respect to the join operation
 * (e.g. '0 for +-' or '1 for *'). This functionality requires C++11 support.*/
template <typename iType, class Lambda, typename ValueType, class JoinType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::ThreadVectorRangeBoundariesStruct<iType, Impl::HPXTeamMember>
        &loop_boundaries,
    const Lambda &lambda, const JoinType &join, ValueType &init_result) {

  ValueType result = init_result;
#ifdef KOKKOS_ENABLE_PRAGMA_IVDEP
#pragma ivdep
#endif
  for (iType i = loop_boundaries.start; i < loop_boundaries.end;
       i += loop_boundaries.increment) {
    ValueType tmp = ValueType();
    lambda(i, tmp);
    join(result, tmp);
  }
  init_result = result;
}

template <typename iType, class Lambda, typename ReducerType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::TeamThreadRangeBoundariesStruct<iType, Impl::HPXTeamMember>
        &loop_boundaries,
    const Lambda &lambda, const ReducerType &reducer) {
  reducer.init(reducer.reference());

#ifdef KOKKOS_ENABLE_PRAGMA_IVDEP
#pragma ivdep
#endif
  for (iType i = loop_boundaries.start; i < loop_boundaries.end;
       i += loop_boundaries.increment) {
    lambda(i, reducer.reference());
  }

  loop_boundaries.thread.team_reduce(reducer);
}

template <typename iType, class Lambda, typename ReducerType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::ThreadVectorRangeBoundariesStruct<iType, Impl::HPXTeamMember>
        &loop_boundaries,
    const Lambda &lambda, const ReducerType &reducer) {
  reducer.init(reducer.reference());

#ifdef KOKKOS_ENABLE_PRAGMA_IVDEP
#pragma ivdep
#endif
  for (iType i = loop_boundaries.start; i < loop_boundaries.end;
       i += loop_boundaries.increment) {
    lambda(i, reducer.reference());
  }
}

/** \brief  Intra-thread vector parallel exclusive prefix sum. Executes
 * lambda(iType i, ValueType & val, bool final) for each i=0..N-1.
 *
 * The range i=0..N-1 is mapped to all vector lanes in the thread and a scan
 * operation is performed. Depending on the target execution space the operator
 * might be called twice: once with final=false and once with final=true. When
 * final==true val contains the prefix sum value. The contribution of this "i"
 * needs to be added to val no matter whether final==true or not. In a serial
 * execution (i.e. team_size==1) the operator is only called once with
 * final==true. Scan_val will be set to the final sum value over all vector
 * lanes. This functionality requires C++11 support.*/
template <typename iType, class FunctorType>
KOKKOS_INLINE_FUNCTION void parallel_scan(
    const Impl::ThreadVectorRangeBoundariesStruct<iType, Impl::HPXTeamMember>
        &loop_boundaries,
    const FunctorType &lambda) {

  typedef Kokkos::Impl::FunctorValueTraits<FunctorType, void> ValueTraits;
  typedef typename ValueTraits::value_type value_type;

  value_type scan_val = value_type();

#ifdef KOKKOS_ENABLE_PRAGMA_IVDEP
#pragma ivdep
#endif
  for (iType i = loop_boundaries.start; i < loop_boundaries.end;
       i += loop_boundaries.increment) {
    lambda(i, scan_val, true);
  }
}

template <class FunctorType>
KOKKOS_INLINE_FUNCTION void
single(const Impl::VectorSingleStruct<Impl::HPXTeamMember> &single_struct,
       const FunctorType &lambda) {
  lambda();
}

template <class FunctorType>
KOKKOS_INLINE_FUNCTION void
single(const Impl::ThreadSingleStruct<Impl::HPXTeamMember> &single_struct,
       const FunctorType &lambda) {
  if (single_struct.team_member.team_rank() == 0)
    lambda();
}

template <class FunctorType, class ValueType>
KOKKOS_INLINE_FUNCTION void
single(const Impl::VectorSingleStruct<Impl::HPXTeamMember> &single_struct,
       const FunctorType &lambda, ValueType &val) {
  lambda(val);
}

template <class FunctorType, class ValueType>
KOKKOS_INLINE_FUNCTION void
single(const Impl::ThreadSingleStruct<Impl::HPXTeamMember> &single_struct,
       const FunctorType &lambda, ValueType &val) {
  if (single_struct.team_member.team_rank() == 0) {
    lambda(val);
  }
  single_struct.team_member.team_broadcast(val, 0);
}

} // namespace Kokkos

#endif /* #if defined( KOKKOS_ENABLE_HPX ) */
#endif /* #ifndef KOKKOS_HPX_HPP */
