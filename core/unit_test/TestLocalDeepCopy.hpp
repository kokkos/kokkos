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
bool check_copy(const ViewType& lhs, const ViewType& rhs) {
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
bool check_copy(const ViewType& view,
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

template <typename ViewType, std::size_t... Ints>
KOKKOS_INLINE_FUNCTION auto extract_subview(ViewType& src, int start, int stop,
                                            std::index_sequence<Ints...>) {
  return Kokkos::subview(src, Kokkos::make_pair(start, stop),
                         ((void)Ints, Kokkos::ALL)...);
}

// Extract a subview from a view to run our tests
template <typename ViewType>
KOKKOS_INLINE_FUNCTION auto extract_subview(ViewType& src, int start,
                                            int stop) {
  return extract_subview(src, start, stop,
                         std::make_index_sequence<ViewType::rank - 1>{});
}

template <typename ViewType>
void reset(ViewType B) {
  Kokkos::deep_copy(B, 0);
}

// local deep copy on a subview
template <typename PolicyType, typename ViewType>
void KOKKOS_INLINE_FUNCTION copy_view_helper(const PolicyType& policy, int idx,
                                             const ViewType& dst,
                                             const ViewType& src,
                                             bool mismatch = false) {
  const int start = 2 * idx;
  const int stop  = 2 * (idx + 1);
  auto subSrc     = extract_subview(src, start, stop);
  auto subDst     = extract_subview(dst, start, mismatch ? stop - 1 : stop);

#if defined(KOKKOS_ENABLE_DEPRECATED_CODE_4)
#if defined(KOKKOS_ENABLE_DEPRECATION_WARNINGS)
  KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_PUSH()
#endif
  if constexpr (std::is_null_pointer_v<PolicyType>) {
    Kokkos::Experimental::local_deep_copy(subDst, subSrc);
  } else {
#if defined(KOKKOS_ENABLE_DEPRECATION_WARNINGS)
    KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_POP()
#endif
#else
  {
#endif
    Kokkos::Experimental::deep_copy(policy, subDst, subSrc);
  }
}

template <typename PolicyType, typename ViewType>
void KOKKOS_INLINE_FUNCTION
copy_view_helper(const PolicyType& policy, int idx, const ViewType& dst,
                 typename ViewType::const_value_type& value, bool mismatch) {
  const int start = 2 * idx;
  const int stop  = 2 * (idx + 1);
  auto subDst     = extract_subview(dst, start, mismatch ? stop - 1 : stop);

#if defined(KOKKOS_ENABLE_DEPRECATED_CODE_4)
#if defined(KOKKOS_ENABLE_DEPRECATION_WARNINGS)
  KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_PUSH()
#endif
  if constexpr (std::is_null_pointer_v<PolicyType>) {
    Kokkos::Experimental::local_deep_copy(subDst, value);
  } else {
#if defined(KOKKOS_ENABLE_DEPRECATION_WARNINGS)
    KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_POP()
#endif
#else
  {
#endif
    Kokkos::Experimental::deep_copy(policy, subDst, value);
  }
}

//-------------------------------------------------------------------------------------------------------------
// Testing code
//-------------------------------------------------------------------------------------------------------------

template <typename ExecSpace, typename ViewType, typename ValueType>
void test_local_deep_copy_team_vector_range(const int N, const ViewType& dst,
                                            const ValueType& src,
                                            bool mismatch = false) {
  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

  Kokkos::parallel_for(
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        if (mismatch) {
          Kokkos::Experimental::deep_copy(
              Kokkos::TeamVectorRange(teamMember, 0),
              extract_subview(dst, 0, dst.extent_int(0) - 1), src);
        } else {
          Kokkos::Experimental::deep_copy(
              Kokkos::TeamVectorRange(teamMember, 0), dst, src);
        }
      });

  Kokkos::fence();
  ASSERT_TRUE(check_copy(dst, src));
}

template <typename ExecSpace, typename ViewType, typename ValueType>
void test_local_deep_copy_team_thread_range(const int N, const ViewType& dst,
                                            const ValueType& src,
                                            bool mismatch = false) {
  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

  Kokkos::parallel_for(
      team_policy(N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& teamMember) {
        if (mismatch) {
          Kokkos::Experimental::deep_copy(
              Kokkos::TeamThreadRange(teamMember, 0),
              extract_subview(dst, 0, dst.extent_int(0) - 1), src);
        } else {
          Kokkos::Experimental::deep_copy(
              Kokkos::TeamThreadRange(teamMember, 0), dst, src);
        }
      });

  Kokkos::fence();
  ASSERT_TRUE(check_copy(dst, src));
}

template <typename ExecSpace, typename ViewType, typename ValueType>
void test_local_deep_copy_thread_vector_range(const int N, const ViewType& dst,
                                              const ValueType& src,
                                              bool mismatch = false) {
  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

  // each thread processes 2 indices in the leading dimension
  Kokkos::parallel_for(
      team_policy(N / 2, 1), KOKKOS_LAMBDA(const member_type& teamMember) {
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(teamMember, N / 2), [=](const int idx) {
              copy_view_helper(Kokkos::ThreadVectorRange(teamMember, 0), idx,
                               dst, src, mismatch);
            });
      });

  Kokkos::fence();
  ASSERT_TRUE(check_copy(dst, src));
}

template <typename ExecSpace, typename ViewType, typename ValueType>
void test_local_deep_copy_sequential(const int N, const ViewType& dst,
                                     const ValueType& src,
                                     bool mismatch = false) {
  // each thread processes 2 indices in the leading dimension
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(0, N / 2), KOKKOS_LAMBDA(const int& idx) {
        copy_view_helper(Kokkos::Experimental::copy_seq(), idx, dst, src,
                         mismatch);
      });

  Kokkos::fence();
  ASSERT_TRUE(check_copy(dst, src));
}

#define KOKKOS_IMPL_LOCAL_DEEP_COPY_TEST(name)                                 \
  template <typename ViewType, typename ExecSpace>                             \
  void run_##name##_policy(const int N) {                                      \
    ViewType A = view_create<ViewType>("A", N);                                \
    ViewType B = view_create<ViewType>("B", N);                                \
                                                                               \
    view_init(A);                                                              \
                                                                               \
    test_local_deep_copy_##name<ExecSpace>(N, B, A);                           \
    reset(B);                                                                  \
    test_local_deep_copy_##name<ExecSpace>(N, B, 20.0);                        \
  }                                                                            \
                                                                               \
  TEST(TEST_CATEGORY, local_deep_copy_##name##_layoutleft) {                   \
    using ExecSpace = TEST_EXECSPACE;                                          \
    using Layout    = Kokkos::LayoutLeft;                                      \
                                                                               \
    run_##name##_policy<Kokkos::View<double*, Layout, ExecSpace>, ExecSpace>(  \
        8);                                                                    \
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
    run_##name##_policy<Kokkos::View<double*, Layout, ExecSpace>, ExecSpace>(  \
        8);                                                                    \
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

#if (defined(KOKKOS_ENABLE_SYCL) && defined(NDEBUG)) || \
    defined(KOKKOS_ENABLE_OPENMPTARGET) || defined(KOKKOS_ENABLE_OPENACC)
#define KOKKOS_IMPL_LOCAL_DEEP_COPY_DEATH_SKIP                               \
  if constexpr (!std::is_same_v<TEST_EXECSPACE,                              \
                                Kokkos::DefaultHostExecutionSpace>) {        \
    GTEST_SKIP() << "device Kokkos::abort() does not terminate the program " \
                    "on sycl (in release mode), openmptarget and openacc";   \
  }
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
    ASSERT_DEATH((test_local_deep_copy_##name<ExecSpace>(N, B, A, true)),  \
                 "Error: Kokkos::deep_copy extents of views don't match"); \
  }                                                                        \
                                                                           \
  TEST(TEST_CATEGORY, local_deep_copy_##name##_extents_mismatch) {         \
    using ExecSpace = TEST_EXECSPACE;                                      \
    using Layout    = Kokkos::LayoutRight;                                 \
    KOKKOS_IMPL_LOCAL_DEEP_COPY_DEATH_SKIP                                 \
    run_##name##_policy_extents_mismatch<                                  \
        Kokkos::View<double*, Layout, ExecSpace>, ExecSpace>(8);           \
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

#if !defined(KOKKOS_LOCAL_DEEP_COPY_SKIP_1)
KOKKOS_IMPL_LOCAL_DEEP_COPY_TEST(team_vector_range)
KOKKOS_IMPL_LOCAL_DEEP_COPY_TEST(team_thread_range)
KOKKOS_IMPL_LOCAL_DEEP_COPY_TEST(thread_vector_range)
KOKKOS_IMPL_LOCAL_DEEP_COPY_TEST(sequential)
#endif

#if !defined(KOKKOS_LOCAL_DEEP_COPY_SKIP_2)
KOKKOS_IMPL_LOCAL_DEEP_COPY_DEATH_TEST(team_vector_range)
KOKKOS_IMPL_LOCAL_DEEP_COPY_DEATH_TEST(team_thread_range)
KOKKOS_IMPL_LOCAL_DEEP_COPY_DEATH_TEST(thread_vector_range)
KOKKOS_IMPL_LOCAL_DEEP_COPY_DEATH_TEST(sequential)

//-------------------------------------------------------------------------------------------------------------

#if defined(KOKKOS_ENABLE_DEPRECATED_CODE_4)

#if defined(KOKKOS_ENABLE_DEPRECATION_WARNINGS)
KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_PUSH()
#endif

template <typename ExecSpace, typename ViewType, typename ValueType>
void test_local_deep_copy_deprecated_team_member(const int N,
                                                 const ViewType& dst,
                                                 const ValueType& src) {
  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

  Kokkos::parallel_for(
      team_policy(N, 1), KOKKOS_LAMBDA(const member_type& teamMember) {
        Kokkos::Experimental::local_deep_copy(teamMember, dst, src);
      });

  Kokkos::fence();
  ASSERT_TRUE(check_copy(dst, src));
}

template <typename ExecSpace, typename ViewType, typename ValueType>
void test_local_deep_copy_deprecated_sequential(const int N,
                                                const ViewType& dst,
                                                const ValueType& src) {
  // each thread processes 2 indices in the leading dimension
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(0, N / 2), KOKKOS_LAMBDA(const int& idx) {
        copy_view_helper(nullptr, idx, dst, src, false);
      });

  Kokkos::fence();
  ASSERT_TRUE(check_copy(dst, src));
}

KOKKOS_IMPL_LOCAL_DEEP_COPY_TEST(deprecated_team_member)
KOKKOS_IMPL_LOCAL_DEEP_COPY_TEST(deprecated_sequential)

#if defined(KOKKOS_ENABLE_DEPRECATION_WARNINGS)
KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_POP()
#endif

#endif
#endif

#if !defined(KOKKOS_LOCAL_DEEP_COPY_SKIP_1)
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

#endif

}  // namespace Test

#undef KOKKOS_IMPL_LOCAL_DEEP_COPY_TEST
#undef KOKKOS_IMPL_LOCAL_DEEP_COPY_DEATH_TEST
#undef KOKKOS_IMPL_LOCAL_DEEP_COPY_DEATH_SKIP

#endif
