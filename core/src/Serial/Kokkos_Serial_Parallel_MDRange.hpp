// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_SERIAL_PARALLEL_MDRANGE_HPP
#define KOKKOS_SERIAL_PARALLEL_MDRANGE_HPP

#include <concepts>
#include <Kokkos_Parallel.hpp>
#include <KokkosExp_MDRangePolicy.hpp>
#include <MDRange/Kokkos_FlatIterate.hpp>
#include <impl/Kokkos_Utilities.hpp>

namespace Kokkos {
namespace Impl {

template <typename IndexType, IndexType End, typename ItegerSequence>
struct make_reverse_integer_sequence_impl;

template <typename IndexType, IndexType End, IndexType... Indices>
struct make_reverse_integer_sequence_impl<
    IndexType, End, std::integer_sequence<IndexType, Indices...>>
    : std::type_identity<std::integer_sequence<IndexType, End - 1 - Indices...>> {};

template <typename IndexType, IndexType N>
using make_reverse_integer_sequence =
    typename make_reverse_integer_sequence_impl<
        IndexType, N, std::make_integer_sequence<IndexType, N>>::type;

template <class MDRP, class Functor, class Tag>
  requires std::same_as<typename MDRP::execution_space, Kokkos::Serial>
class FlatIterate<MDRP, Functor, Tag> {
 public:
  using range_policy         = typename MDRP::impl_range_policy;
  using index_type           = typename range_policy::index_type;
  using iteration_pattern    = typename MDRP::iteration_pattern;
  using point_type           = typename MDRP::point_type;
  static constexpr auto rank = MDRP::rank;

  FlatIterate(const MDRP& mdrp, const Functor& fun)
      : m_md_range_policy(mdrp), m_functor(fun) {}

  void exec() const {
    point_type p;
    if constexpr (iteration_pattern::inner_direction == Iterate::Left) {
      exec_rank(std::make_integer_sequence<int, rank>{}, p, m_tag);
    } else {
      exec_rank(make_reverse_integer_sequence<int, rank>{}, p, m_tag);
    }
  }

  const MDRP& policy() const noexcept { return m_md_range_policy; }

 private:
  struct NoTag {};

  void apply_to_functor(const point_type& point, NoTag) const {
    apply(m_functor, point);
  }

  template<typename AnyTag>
  void apply_to_functor(const point_type& point, AnyTag tag) const {
    apply(
        [tag, this]<typename... Args>(Args&&... args) {
          m_functor(tag, std::forward<Args>(args)...);
        },
        point);
  }

  template <int Rank, class TagOrNoTag>
  void exec_rank(std::integer_sequence<int, Rank>, point_type& point, TagOrNoTag tag) const {
    using array_index_type = typename MDRP::array_index_type;
    for (array_index_type i = m_md_range_policy.m_lower[Rank];
         i < m_md_range_policy.m_upper[Rank]; ++i) {
      point[Rank] = i;
      apply_to_functor(point, tag);
    }
  }

  template <int Rank, int SecondRank, int... RemRanks, class TagOrNoTag>
  void exec_rank(std::integer_sequence<int, Rank, SecondRank, RemRanks...>,
                 point_type& point, TagOrNoTag tag) const {
    using array_index_type = typename MDRP::array_index_type;
    for (array_index_type i = m_md_range_policy.m_lower[Rank];
         i < m_md_range_policy.m_upper[Rank]; ++i) {
      point[Rank] = i;
      exec_rank(std::integer_sequence<int, SecondRank, RemRanks...>{}, point,
                tag);
    }
  }

  const MDRP m_md_range_policy;
  const Functor m_functor;
  static constexpr std::conditional_t<std::is_void_v<Tag>, NoTag, Tag> m_tag{};
};

template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::MDRangePolicy<Traits...>,
                  Kokkos::Serial> {
 private:
  using MDRangePolicy = Kokkos::MDRangePolicy<Traits...>;
  using Policy        = typename MDRangePolicy::impl_range_policy;

  //using iterate_type = typename Kokkos::Impl::HostIterateTile<
  //    MDRangePolicy, FunctorType, typename MDRangePolicy::work_tag, void>;
  using iterate_type = FlatIterate<MDRangePolicy, FunctorType, typename MDRangePolicy::work_tag>;

  const iterate_type m_iter;

  void exec() const {
    //const typename Policy::member_type e = m_iter.policy().m_num_tiles;
    //for (typename Policy::member_type i = 0; i < e; ++i) {
    //  m_iter(i);
    //}
    m_iter.exec();
  }

 public:
  inline void execute() const {
    // caused a possibly codegen-related slowdown, especially in GCC 9-11
    // with KOKKOS_ARCH_NATIVE
    // https://github.com/kokkos/kokkos/issues/7268
#ifndef KOKKOS_ENABLE_ATOMICS_BYPASS
    // Make sure kernels are running sequentially even when using multiple
    // threads
    auto* internal_instance =
        m_iter.policy().space().impl_internal_space_instance();
    std::lock_guard<std::mutex> lock(internal_instance->m_instance_mutex);
#endif
    this->exec();
  }
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
      : m_iter(arg_policy, arg_functor) {}
};

template <class CombinedFunctorReducerType, class... Traits>
class ParallelReduce<CombinedFunctorReducerType,
                     Kokkos::MDRangePolicy<Traits...>, Kokkos::Serial> {
 private:
  using MDRangePolicy = Kokkos::MDRangePolicy<Traits...>;
  using Policy        = typename MDRangePolicy::impl_range_policy;
  using FunctorType   = typename CombinedFunctorReducerType::functor_type;
  using ReducerType   = typename CombinedFunctorReducerType::reducer_type;

  using WorkTag = typename MDRangePolicy::work_tag;

  using pointer_type   = typename ReducerType::pointer_type;
  using value_type     = typename ReducerType::value_type;
  using reference_type = typename ReducerType::reference_type;

  using iterate_type = typename Kokkos::Impl::HostIterateTile<
      MDRangePolicy, CombinedFunctorReducerType, WorkTag, reference_type>;
  const iterate_type m_iter;
  const pointer_type m_result_ptr;

  inline void exec(reference_type update) const {
    const typename Policy::member_type e = m_iter.m_rp.m_num_tiles;
    for (typename Policy::member_type i = 0; i < e; ++i) {
      m_iter(i, update);
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
    const ReducerType& reducer     = m_iter.m_func.get_reducer();
    const size_t pool_reduce_size  = reducer.value_size();
    const size_t team_reduce_size  = 0;  // Never shrinks
    const size_t team_shared_size  = 0;  // Never shrinks
    const size_t thread_local_size = 0;  // Never shrinks

    auto* internal_instance =
        m_iter.m_rp.space().impl_internal_space_instance();

    // caused a possibly codegen-related slowdown, especially in GCC 9-11
    // with KOKKOS_ARCH_NATIVE
    // https://github.com/kokkos/kokkos/issues/7268
#ifndef KOKKOS_ENABLE_ATOMICS_BYPASS
    // Make sure kernels are running sequentially even when using multiple
    // threads, lock resize_thread_team_data
    std::lock_guard<std::mutex> instance_lock(
        internal_instance->m_instance_mutex);
#endif
    internal_instance->resize_thread_team_data(
        pool_reduce_size, team_reduce_size, team_shared_size,
        thread_local_size);

    pointer_type ptr =
        m_result_ptr
            ? m_result_ptr
            : pointer_type(
                  internal_instance->m_thread_team_data.pool_reduce_local());

    reference_type update = reducer.init(ptr);

    this->exec(update);

    reducer.final(ptr);
  }

  template <class ViewType>
  ParallelReduce(const CombinedFunctorReducerType& arg_functor_reducer,
                 const MDRangePolicy& arg_policy,
                 const ViewType& arg_result_view)
      : m_iter(arg_policy, arg_functor_reducer),
        m_result_ptr(arg_result_view.data()) {
    static_assert(Kokkos::is_view<ViewType>::value,
                  "Kokkos::Serial reduce result must be a View");

    static_assert(
        Kokkos::Impl::MemorySpaceAccess<typename ViewType::memory_space,
                                        Kokkos::HostSpace>::accessible,
        "Kokkos::Serial reduce result must be a View accessible from "
        "HostSpace");
  }
};

}  // namespace Impl
}  // namespace Kokkos

#endif
