//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef KOKKOS_TEST_LOCAL_DEEP_COPY_HPP
#define KOKKOS_TEST_LOCAL_DEEP_COPY_HPP

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

namespace Test {

//-------------------------------------------------------------------------------------------------------------
// Utility functions
//-------------------------------------------------------------------------------------------------------------

template <typename ViewType>
bool check_view_copy(const ViewType& lhs, const ViewType& rhs) {
  using exec_space = typename ViewType::execution_space;

  bool result = true;

  Kokkos::parallel_reduce(
      "check_view_copy", Kokkos::RangePolicy<exec_space>(0, lhs.span()),
      KOKKOS_LAMBDA(int i, bool& local_result) {
        local_result = (lhs.data()[i] == rhs.data()[i]) && local_result;
      },
      Kokkos::LAnd<bool>(result));

  Kokkos::fence();
  return result;
}

template <typename ViewType>
bool check_value_copy(const ViewType& view,
                      typename ViewType::const_value_type& value) {
  using exec_space = typename ViewType::execution_space;

  bool result = true;

  Kokkos::parallel_reduce(
      "check_value_copy", Kokkos::RangePolicy<exec_space>(0, view.span()),
      KOKKOS_LAMBDA(int i, bool& local_result) {
        local_result = (view.data()[i] == value) && local_result;
      },
      Kokkos::LAnd<bool>(result));

  Kokkos::fence();
  return result;
}

template <typename ViewType>
void view_init(ViewType& view) {
  using exec_space = typename ViewType::execution_space;

  Kokkos::parallel_for(
      "initialize array", Kokkos::RangePolicy<exec_space>(0, view.span()),
      KOKKOS_LAMBDA(int i) { view.data()[i] = i; });
  Kokkos::fence();
}

template <typename ViewType, std::size_t... Ints>
ViewType view_create(std::string label, const int N,
                     std::index_sequence<Ints...>) {
  return ViewType(label, ((void)Ints, N)...);
}

// Create a view with a given label and dimensions
template <typename ViewType>
ViewType view_create(std::string label, const int N) {
  return view_create<ViewType>(label, N,
                               std::make_index_sequence<ViewType::rank>{});
}

template <typename ViewType, typename Bounds, std::size_t... Ints>
KOKKOS_INLINE_FUNCTION auto extract_subview(ViewType& src, int lid,
                                            Bounds bounds,
                                            std::index_sequence<Ints...>) {
  return Kokkos::subview(src, lid, bounds, ((void)Ints, Kokkos::ALL)...);
}

// Extract a subview from a view to run our tests
template <typename ViewType, typename Bounds>
KOKKOS_INLINE_FUNCTION auto extract_subview(ViewType& src, int lid,
                                            Bounds bounds) {
  return extract_subview(src, lid, bounds,
                         std::make_index_sequence<ViewType::rank - 2>{});
}

template <typename ViewType>
void reset(ViewType B) {
  Kokkos::deep_copy(B, 0);
}

//-------------------------------------------------------------------------------------------------------------
// Helper functor
//-------------------------------------------------------------------------------------------------------------

/** \brief  Functor used to call the different local deep copy overloads in the
 * tests */
template <typename Base>
struct copy_functor {
 private:
  Base base;

 public:
  template <typename... Args>
  copy_functor(Args... args) : base(args...) {}

  template <typename PolicyType,
            typename = std::enable_if_t<
                Kokkos::Experimental::Impl::is_team_policy_v<PolicyType>>>
  void KOKKOS_INLINE_FUNCTION operator()(PolicyType policy, int start,
                                         int stop) const {
    base.copy(policy, policy.member.league_rank(), start, stop);
  }

  void KOKKOS_INLINE_FUNCTION
  operator()(const Kokkos::Experimental::Impl::CopySeqTag policy, int idx,
             int stop) const {
    base.copy(policy, idx, 0, stop);
  }

#if defined(KOKKOS_ENABLE_DEPRECATED_CODE_4)
  template <typename MemberType,
            typename = std::enable_if_t<
                !Kokkos::Experimental::Impl::is_team_policy_v<MemberType>>>
  void KOKKOS_INLINE_FUNCTION operator()(const MemberType& member, int start,
                                         int stop) const {
    base.copy(member, member.league_rank(), start, stop);
  }

  void KOKKOS_INLINE_FUNCTION operator()(int idx, int stop) const {
    base.copy(nullptr, idx, 0, stop);
  }
#endif

  bool success() const { return base.success(); }
};

template <typename ViewType>
struct copy_view {
 private:
  ViewType src;
  ViewType dst;
  bool mismatch;

 public:
  template <typename PolicyType>
  void KOKKOS_INLINE_FUNCTION copy(const PolicyType& policy, int first_dim,
                                   int start, int stop) const {
    auto subSrc =
        extract_subview(src, first_dim, Kokkos::make_pair(start, stop));
    // modify the range [start, stop) to make the extents mismatched
    if (mismatch) {
      if (stop < dst.extent_int(1)) {
        stop++;
      } else if (start > 0) {
        start--;
      } else {
        start++;
      }
    }
    auto subDst =
        extract_subview(dst, first_dim, Kokkos::make_pair(start, stop));
    if constexpr (Kokkos::Experimental::Impl::is_team_policy_v<PolicyType> ||
                  std::is_same_v<PolicyType,
                                 Kokkos::Experimental::Impl::CopySeqTag>) {
      Kokkos::Experimental::deep_copy(policy, subDst, subSrc);
#if defined(KOKKOS_ENABLE_DEPRECATED_CODE_4)
#if defined(KOKKOS_ENABLE_DEPRECATION_WARNINGS)
      KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_PUSH()
#endif
    } else if constexpr (!std::is_null_pointer_v<PolicyType>) {
      Kokkos::Experimental::local_deep_copy(policy, subDst, subSrc);
    } else {
      Kokkos::Experimental::local_deep_copy(subDst, subSrc);
#if defined(KOKKOS_ENABLE_DEPRECATION_WARNINGS)
      KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_POP()
#endif
#endif
    }
  }

  copy_view(ViewType src_, ViewType dst_, bool mismatch_ = false)
      : src(src_), dst(dst_), mismatch(mismatch_) {}

  bool success() const { return check_view_copy(src, dst); }
};

template <typename ViewType>
struct copy_value {
 private:
  ViewType dst;
  typename ViewType::const_value_type value;

 public:
  template <typename PolicyType>
  void KOKKOS_INLINE_FUNCTION copy(const PolicyType& policy, int first_dim,
                                   int start, int stop) const {
    auto subDst =
        extract_subview(dst, first_dim, Kokkos::make_pair(start, stop));
    if constexpr (Kokkos::Experimental::Impl::is_team_policy_v<PolicyType> ||
                  std::is_same_v<PolicyType,
                                 Kokkos::Experimental::Impl::CopySeqTag>) {
      Kokkos::Experimental::deep_copy(policy, subDst, value);
#if defined(KOKKOS_ENABLE_DEPRECATED_CODE_4)
#if defined(KOKKOS_ENABLE_DEPRECATION_WARNINGS)
      KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_PUSH()
#endif
    } else if constexpr (!std::is_null_pointer_v<PolicyType>) {
      Kokkos::Experimental::local_deep_copy(policy, subDst, value);
    } else {
      Kokkos::Experimental::local_deep_copy(subDst, value);
#if defined(KOKKOS_ENABLE_DEPRECATION_WARNINGS)
      KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_POP()
#endif
#endif
    }
  }

  copy_value(ViewType dst_, typename ViewType::const_value_type value_)
      : dst(dst_), value(value_) {}

  bool success() const { return check_value_copy(dst, value); }
};

//-------------------------------------------------------------------------------------------------------------
// Testing code
//-------------------------------------------------------------------------------------------------------------

template <typename ExecSpace, typename Functor>
void test_local_deep_copy_team_vector_range(const int N,
                                            const Functor& functor) {
  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

  Kokkos::parallel_for(
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        functor(Kokkos::TeamVectorRange(teamMember, 0), 0, N);
      });

  Kokkos::fence();
  ASSERT_TRUE(functor.success());
}

template <typename ExecSpace, typename Functor>
void test_local_deep_copy_team_thread_range(const int N,
                                            const Functor& functor) {
  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

  Kokkos::parallel_for(
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        functor(Kokkos::TeamThreadRange(teamMember, 0), 0, N);
      });

  Kokkos::fence();
  ASSERT_TRUE(functor.success());
}

template <typename ExecSpace, typename Functor>
void test_local_deep_copy_thread_vector_range(const int N,
                                              const Functor& functor) {
  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

  Kokkos::parallel_for(
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        // Compute the number of units of work per thread
        auto thread_number = teamMember.league_size();
        auto unitsOfWork   = N / thread_number;
        if (N % thread_number) {
          unitsOfWork += 1;
        }
        auto numberOfBatches = N / unitsOfWork;

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(teamMember, numberOfBatches),
            [=](const int indexWithinBatch) {
              const int idx = indexWithinBatch;

              auto start = idx * unitsOfWork;
              auto stop  = (idx + 1) * unitsOfWork;
              stop       = Kokkos::clamp(stop, 0, N);
              functor(Kokkos::ThreadVectorRange(teamMember, 0), start, stop);
            });
      });

  Kokkos::fence();
  ASSERT_TRUE(functor.success());
}

template <typename ExecSpace, typename Functor>
void test_local_deep_copy_sequential(const int N, const Functor& functor) {
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(const int& idx) {
        functor(Kokkos::Experimental::copy_seq(), idx, N);
      });

  Kokkos::fence();
  ASSERT_TRUE(functor.success());
}

#define KOKKOS_IMPL_LOCAL_DEEP_COPY_TEST(name)                                 \
  template <typename ViewType, typename ExecSpace>                             \
  void run_##name##_policy(const int N) {                                      \
    ViewType A = view_create<ViewType>("A", N);                                \
    ViewType B = view_create<ViewType>("B", N);                                \
                                                                               \
    view_init(A);                                                              \
                                                                               \
    test_local_deep_copy_##name<ExecSpace>(                                    \
        N, copy_functor<copy_view<ViewType>>(B, A));                           \
    reset(B);                                                                  \
    test_local_deep_copy_##name<ExecSpace>(                                    \
        N, copy_functor<copy_value<ViewType>>(B, 20));                         \
  }                                                                            \
                                                                               \
  TEST(TEST_CATEGORY, local_deep_copy_##name##_layoutleft) {                   \
    using ExecSpace = TEST_EXECSPACE;                                          \
    using Layout    = Kokkos::LayoutLeft;                                      \
                                                                               \
    run_##name##_policy<Kokkos::View<double**, Layout, ExecSpace>, ExecSpace>( \
        8);                                                                    \
    run_##name##_policy<Kokkos::View<double***, Layout, ExecSpace>,            \
                        ExecSpace>(8);                                         \
    run_##name##_policy<Kokkos::View<double****, Layout, ExecSpace>,           \
                        ExecSpace>(8);                                         \
    run_##name##_policy<Kokkos::View<double*****, Layout, ExecSpace>,          \
                        ExecSpace>(8);                                         \
    run_##name##_policy<Kokkos::View<double******, Layout, ExecSpace>,         \
                        ExecSpace>(8);                                         \
    run_##name##_policy<Kokkos::View<double*******, Layout, ExecSpace>,        \
                        ExecSpace>(8);                                         \
    run_##name##_policy<Kokkos::View<double********, Layout, ExecSpace>,       \
                        ExecSpace>(8);                                         \
  }                                                                            \
  TEST(TEST_CATEGORY, local_deep_copy_##name##_layoutright) {                  \
    using ExecSpace = TEST_EXECSPACE;                                          \
    using Layout    = Kokkos::LayoutRight;                                     \
                                                                               \
    run_##name##_policy<Kokkos::View<double**, Layout, ExecSpace>, ExecSpace>( \
        8);                                                                    \
    run_##name##_policy<Kokkos::View<double***, Layout, ExecSpace>,            \
                        ExecSpace>(8);                                         \
    run_##name##_policy<Kokkos::View<double****, Layout, ExecSpace>,           \
                        ExecSpace>(8);                                         \
    run_##name##_policy<Kokkos::View<double*****, Layout, ExecSpace>,          \
                        ExecSpace>(8);                                         \
    run_##name##_policy<Kokkos::View<double******, Layout, ExecSpace>,         \
                        ExecSpace>(8);                                         \
    run_##name##_policy<Kokkos::View<double*******, Layout, ExecSpace>,        \
                        ExecSpace>(8);                                         \
    run_##name##_policy<Kokkos::View<double********, Layout, ExecSpace>,       \
                        ExecSpace>(8);                                         \
  }

KOKKOS_IMPL_LOCAL_DEEP_COPY_TEST(team_vector_range)
KOKKOS_IMPL_LOCAL_DEEP_COPY_TEST(team_thread_range)
KOKKOS_IMPL_LOCAL_DEEP_COPY_TEST(thread_vector_range)
KOKKOS_IMPL_LOCAL_DEEP_COPY_TEST(sequential)

#if (defined(KOKKOS_ENABLE_SYCL) && defined(NDEBUG)) || \
    defined(KOKKOS_ENABLE_OPENMPTARGET) || defined(KOKKOS_ENABLE_OPENACC)
#define KOKKOS_IMPL_LOCAL_DEEP_COPY_DEATH_SKIP                         \
  GTEST_SKIP()                                                         \
      << "Kokkos::abort() does not terminate the program on sycl (in " \
         "release mode), openmptarget and openacc";
#else
#define KOKKOS_IMPL_LOCAL_DEEP_COPY_DEATH_SKIP
#endif

#define KOKKOS_IMPL_LOCAL_DEEP_COPY_DEATH_TEST(name)                       \
  template <typename ViewType, typename ExecSpace>                         \
  void run_##name##_policy_extents_mismatch(const int N) {                 \
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";                \
                                                                           \
    ViewType A = view_create<ViewType>("A", N);                            \
    ViewType B = view_create<ViewType>("B", N);                            \
                                                                           \
    ASSERT_DEATH((test_local_deep_copy_##name<ExecSpace>(                  \
                     N, copy_functor<copy_view<ViewType>>(B, A, true))),   \
                 "Error: Kokkos::deep_copy extents of views don't match"); \
  }                                                                        \
                                                                           \
  TEST(TEST_CATEGORY, local_deep_copy_##name##_extents_mismatch) {         \
    using ExecSpace = TEST_EXECSPACE;                                      \
    using Layout    = Kokkos::LayoutRight;                                 \
    KOKKOS_IMPL_LOCAL_DEEP_COPY_DEATH_SKIP                                 \
    run_##name##_policy_extents_mismatch<                                  \
        Kokkos::View<double**, Layout, ExecSpace>, ExecSpace>(8);          \
    run_##name##_policy_extents_mismatch<                                  \
        Kokkos::View<double***, Layout, ExecSpace>, ExecSpace>(8);         \
    run_##name##_policy_extents_mismatch<                                  \
        Kokkos::View<double****, Layout, ExecSpace>, ExecSpace>(8);        \
    run_##name##_policy_extents_mismatch<                                  \
        Kokkos::View<double*****, Layout, ExecSpace>, ExecSpace>(8);       \
    run_##name##_policy_extents_mismatch<                                  \
        Kokkos::View<double******, Layout, ExecSpace>, ExecSpace>(8);      \
    run_##name##_policy_extents_mismatch<                                  \
        Kokkos::View<double*******, Layout, ExecSpace>, ExecSpace>(8);     \
    run_##name##_policy_extents_mismatch<                                  \
        Kokkos::View<double********, Layout, ExecSpace>, ExecSpace>(8);    \
  }

KOKKOS_IMPL_LOCAL_DEEP_COPY_DEATH_TEST(team_vector_range)
KOKKOS_IMPL_LOCAL_DEEP_COPY_DEATH_TEST(team_thread_range)
KOKKOS_IMPL_LOCAL_DEEP_COPY_DEATH_TEST(thread_vector_range)
KOKKOS_IMPL_LOCAL_DEEP_COPY_DEATH_TEST(sequential)

//-------------------------------------------------------------------------------------------------------------

#if defined(KOKKOS_ENABLE_DEPRECATED_CODE_4)

#if defined(KOKKOS_ENABLE_DEPRECATION_WARNINGS)
KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_PUSH()
#endif

template <typename ExecSpace, typename Functor>
void test_local_deep_copy_deprecated_team_member(const int N,
                                                 const Functor& functor) {
  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

  Kokkos::parallel_for(
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        functor(teamMember, 0, N);
      });

  Kokkos::fence();
  ASSERT_TRUE(functor.success());
}

template <typename ExecSpace, typename Functor>
void test_local_deep_copy_deprecated_sequential(const int N,
                                                const Functor& functor) {
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(0, N),
      KOKKOS_LAMBDA(const int& idx) { functor(idx, N); });

  Kokkos::fence();
  ASSERT_TRUE(functor.success());
}

KOKKOS_IMPL_LOCAL_DEEP_COPY_TEST(deprecated_team_member)
KOKKOS_IMPL_LOCAL_DEEP_COPY_TEST(deprecated_sequential)

#if defined(KOKKOS_ENABLE_DEPRECATION_WARNINGS)
KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_POP()
#endif

#endif

namespace Impl {
template <typename T, typename ShMemType>
using ShMemView =
    Kokkos::View<T, Kokkos::LayoutRight, ShMemType, Kokkos::MemoryUnmanaged>;
}  // namespace Impl

void test_local_deep_copy_scratch(int scratch_level) {
  const int N = 8;
  const int bytes_per_team =
      Impl::ShMemView<double**,
                      TEST_EXECSPACE::scratch_memory_space>::shmem_size(N, 1);

  Kokkos::TeamPolicy<TEST_EXECSPACE> policy(1, Kokkos::AUTO);
  auto team_exec =
      policy.set_scratch_size(scratch_level, Kokkos::PerTeam(bytes_per_team));

  Kokkos::View<double*, TEST_EXECSPACE::memory_space> check_view_1("check_1",
                                                                   N);
  Kokkos::View<double*, TEST_EXECSPACE::memory_space> check_view_2("check_2",
                                                                   N);
  Kokkos::parallel_for(
      team_exec,
      KOKKOS_LAMBDA(
          const Kokkos::TeamPolicy<TEST_EXECSPACE>::member_type& team) {
        using ShmemType = TEST_EXECSPACE::scratch_memory_space;
        auto shview     = Impl::ShMemView<double**, ShmemType>(
            team.team_scratch(scratch_level), N, 1);

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, N), [=](const size_t index) {
              auto thread_shview =
                  Kokkos::subview(shview, index, Kokkos::ALL());
              Kokkos::Experimental::deep_copy(
                  Kokkos::ThreadVectorRange(team, 0), thread_shview, index);
            });

        if (scratch_level == 0) {
          Kokkos::Experimental::deep_copy(
              Kokkos::TeamThreadRange(team, 0), check_view_1,
              Kokkos::subview(shview, Kokkos::ALL(), 0));

          Kokkos::Experimental::deep_copy(Kokkos::TeamThreadRange(team, 0),
                                          shview, 6.);
          Kokkos::Experimental::deep_copy(
              Kokkos::TeamThreadRange(team, 0), check_view_2,
              Kokkos::subview(shview, Kokkos::ALL(), 0));
        } else {
          Kokkos::Experimental::deep_copy(
              Kokkos::TeamVectorRange(team, 0), check_view_1,
              Kokkos::subview(shview, Kokkos::ALL(), 0));

          Kokkos::Experimental::deep_copy(Kokkos::TeamVectorRange(team, 0),
                                          shview, 6.);
          Kokkos::Experimental::deep_copy(
              Kokkos::TeamVectorRange(team, 0), check_view_2,
              Kokkos::subview(shview, Kokkos::ALL(), 0));
        }
      });

  auto host_copy_1 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), check_view_1);
  auto host_copy_2 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), check_view_2);

  for (unsigned int i = 0; i < N; ++i) {
    ASSERT_EQ(host_copy_1(i), i);
    ASSERT_EQ(host_copy_2(i), 6.0);
  }
}

TEST(TEST_CATEGORY, local_deep_copy_team_scratch) {
  test_local_deep_copy_scratch(1);
}

TEST(TEST_CATEGORY, local_deep_copy_thread_scratch) {
  test_local_deep_copy_scratch(0);
}

}  // namespace Test

#undef KOKKOS_IMPL_LOCAL_DEEP_COPY_TEST
#undef KOKKOS_IMPL_LOCAL_DEEP_COPY_DEATH_TEST
#undef KOKKOS_IMPL_LOCAL_DEEP_COPY_DEATH_SKIP

#endif
