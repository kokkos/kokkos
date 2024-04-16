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

#include <Kokkos_Core.hpp>
#include <thread>

namespace {

#ifdef KOKKOS_ENABLE_OPENMP
template <class Lambda1, class Lambda2>
void run_threaded_test(const Lambda1 l1, const Lambda2 l2) {
#pragma omp parallel num_threads(2)
  {
    if (omp_get_thread_num() == 0) l1();
    if (omp_get_thread_num() == 1) l2();
  }
}
// We cannot run the multithreaded test when threads or HPX is enabled because
// we cannot launch a thread from inside another thread
#elif !defined(KOKKOS_ENABLE_THREADS) && !defined(KOKKOS_ENABLE_HPX)
template <class Lambda1, class Lambda2>
void run_threaded_test(const Lambda1 l1, const Lambda2 l2) {
  std::thread t1(l1);
  std::thread t2(l2);
  t1.join();
  t2.join();
}
#else
template <class Lambda1, class Lambda2>
void run_threaded_test(const Lambda1 l1, const Lambda2 l2) {
  l1();
  l2();
}
#endif

void run_exec_space_thread_safety_range() {
  constexpr int N = 1000;
  constexpr int M = 1000;

  Kokkos::View<int *, TEST_EXECSPACE> view("view", N);
  Kokkos::View<Kokkos::pair<int, int>, TEST_EXECSPACE> error_index(
      "error_index");

  run_threaded_test(
      [=]() {
        TEST_EXECSPACE exec;
        for (int j = 0; j < M; ++j) {
          Kokkos::parallel_for(
              "bar", Kokkos::RangePolicy<TEST_EXECSPACE>(exec, 0, N),
              KOKKOS_LAMBDA(int i) {
                if (i < N - 1 &&
                    (Kokkos::atomic_load(view.data() + (i + 1)) - view(i)) > 1)
                  Kokkos::atomic_store(error_index.data(),
                                       Kokkos::pair{j + 1, i});
                ++view(i);
              });
        }
        exec.fence();
      },
      [=]() {
        TEST_EXECSPACE exec;
        for (int j = 0; j < M; ++j) {
          Kokkos::parallel_for(
              "foo", Kokkos::RangePolicy<TEST_EXECSPACE>(exec, 0, N),
              KOKKOS_LAMBDA(int i) {
                if (i < N - 1 &&
                    (Kokkos::atomic_load(view.data() + (N - 2 - i)) -
                     view(N - 1 - i)) > 1)
                  Kokkos::atomic_store(error_index.data(),
                                       Kokkos::pair{j + 1, N - 1 - i});
                ++view(N - 1 - i);
              });
        }
        exec.fence();
      });

  auto host_error_index =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, error_index);
  auto [iteration, element] = Kokkos::pair<int, int>(host_error_index());
  ASSERT_EQ(iteration, 0) << " failing at " << iteration << ", " << element;
  auto host_view =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
  for (int i = 0; i < N; ++i)
    ASSERT_EQ(host_view(i), 2 * M) << " failing at " << i;
}

TEST(TEST_CATEGORY, exec_space_thread_safety_range) {
#ifdef KOKKOS_ENABLE_OPENMPTARGET
  if (std::is_same_v<TEST_EXECSPACE, Kokkos::Experimental::OpenMPTarget>)
    GTEST_SKIP() << "skipping since test is known to fail for OpenMPTarget";
#endif
  run_exec_space_thread_safety_range();
}

void run_exec_space_thread_safety_mdrange() {
  constexpr int N = 1000;
  constexpr int M = 1000;

  Kokkos::View<int *, TEST_EXECSPACE> view("view", N);
  Kokkos::View<Kokkos::pair<int, int>, TEST_EXECSPACE> error_index(
      "error_index");

  run_threaded_test(
      [=]() {
        TEST_EXECSPACE exec;
        for (int j = 0; j < M; ++j) {
          Kokkos::parallel_for(
              "bar",
              Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>(
                  exec, {0, 0}, {N, 1}),
              KOKKOS_LAMBDA(int i, int) {
                if (i < N - 1 &&
                    (Kokkos::atomic_load(view.data() + (i + 1)) - view(i)) > 1)
                  Kokkos::atomic_store(error_index.data(),
                                       Kokkos::pair{j + 1, i});
                ++view(i);
              });
        }
        exec.fence();
      },
      [=]() {
        TEST_EXECSPACE exec;
        for (int j = 0; j < M; ++j) {
          Kokkos::parallel_for(
              "foo",
              Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>(
                  exec, {0, 0}, {N, 1}),
              KOKKOS_LAMBDA(int i, int) {
                if (i < N - 1 &&
                    (Kokkos::atomic_load(view.data() + (N - 2 - i)) -
                     view(N - 1 - i)) > 1)
                  Kokkos::atomic_store(error_index.data(),
                                       Kokkos::pair{j + 1, N - 1 - i});
                ++view(N - 1 - i);
              });
        }
        exec.fence();
      });

  auto host_error_index =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, error_index);
  auto [iteration, element] = Kokkos::pair<int, int>(host_error_index());
  ASSERT_EQ(iteration, 0) << " failing at " << iteration << ", " << element;
  auto host_view =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
  for (int i = 0; i < N; ++i)
    ASSERT_EQ(host_view(i), 2 * M) << " failing at " << i;
}

TEST(TEST_CATEGORY, exec_space_thread_safety_mdrange) {
#ifdef KOKKOS_ENABLE_OPENMPTARGET
  if (std::is_same_v<TEST_EXECSPACE, Kokkos::Experimental::OpenMPTarget>)
    GTEST_SKIP() << "skipping since test is known to fail for OpenMPTarget";
#endif
  run_exec_space_thread_safety_mdrange();
}

void run_exec_space_thread_safety_team_policy() {
  constexpr int N = 1000;
  constexpr int M = 1000;

  Kokkos::View<int *, TEST_EXECSPACE> view("view", N);
  Kokkos::View<int, TEST_EXECSPACE> error_index("error_index");

  run_threaded_test(
      [=]() {
        TEST_EXECSPACE exec;
        for (int j = 0; j < M; ++j) {
          Kokkos::parallel_for(
              "bar", Kokkos::TeamPolicy<TEST_EXECSPACE>(exec, N, 1, 1),
              KOKKOS_LAMBDA(
                  const Kokkos::TeamPolicy<TEST_EXECSPACE>::member_type
                      &team_member) {
                int i = team_member.league_rank();
                if (i < N - 1 &&
                    (Kokkos::atomic_load(view.data() + (i + 1)) - view(i)) > 1)
                  Kokkos::atomic_store(error_index.data(), 1);
                ++view(i);
              });
        }
        exec.fence();
      },
      [=]() {
        TEST_EXECSPACE exec;
        for (int j = 0; j < M; ++j) {
          Kokkos::parallel_for(
              "foo", Kokkos::TeamPolicy<TEST_EXECSPACE>(exec, N, 1, 1),
              KOKKOS_LAMBDA(
                  const Kokkos::TeamPolicy<TEST_EXECSPACE>::member_type
                      &team_member) {
                int i = team_member.league_rank();
                if (i < N - 1 &&
                    (Kokkos::atomic_load(view.data() + (N - 2 - i)) -
                     view(N - 1 - i)) > 1)
                  Kokkos::atomic_store(error_index.data(), 1);
                ++view(N - 1 - i);
              });
        }
        exec.fence();
      });

  auto host_error_index =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, error_index);
  ASSERT_EQ(host_error_index(), 0);
  auto host_view =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
  for (int i = 0; i < N; ++i)
    ASSERT_EQ(host_view(i), 2 * M) << " failing at " << i;
}

TEST(TEST_CATEGORY, exec_space_thread_safety_team_policy) {
// FIXME_OPENMPTARGET
#ifdef KOKKOS_ENABLE_OPENMPTARGET
  if (std::is_same_v<TEST_EXECSPACE, Kokkos::Experimental::OpenMPTarget>)
    GTEST_SKIP() << "skipping for OpenMPTarget since the test is designed to "
                    "run with vector_length=1";
#endif
  run_exec_space_thread_safety_team_policy();
}

void run_exec_space_thread_safety_range_reduce() {
  constexpr int N = 1000;
  constexpr int M = 1000;

  Kokkos::View<int *, TEST_EXECSPACE> view("view", N);
  Kokkos::View<Kokkos::pair<int, int>, TEST_EXECSPACE> error_index(
      "error_index");

  run_threaded_test(
      [=]() {
        int dummy;
        TEST_EXECSPACE exec;
        for (int j = 0; j < M; ++j) {
          Kokkos::parallel_reduce(
              "bar", Kokkos::RangePolicy<TEST_EXECSPACE>(exec, 0, N),
              KOKKOS_LAMBDA(int i, int &) {
                if (i < N - 1 &&
                    (Kokkos::atomic_load(view.data() + (i + 1)) - view(i)) > 1)
                  Kokkos::atomic_store(error_index.data(),
                                       Kokkos::pair{j + 1, i});
                ++view(i);
              },
              dummy);
        }
        exec.fence();
      },
      [=]() {
        int dummy;
        TEST_EXECSPACE exec;
        for (int j = 0; j < M; ++j) {
          Kokkos::parallel_reduce(
              "foo", Kokkos::RangePolicy<TEST_EXECSPACE>(exec, 0, N),
              KOKKOS_LAMBDA(int i, int &) {
                if (i < N - 1 &&
                    (Kokkos::atomic_load(view.data() + (N - 2 - i)) -
                     view(N - 1 - i)) > 1)
                  Kokkos::atomic_store(error_index.data(),
                                       Kokkos::pair{j + 1, N - 1 - i});
                ++view(N - 1 - i);
              },
              dummy);
        }
        exec.fence();
      });

  auto host_error_index =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, error_index);
  auto [iteration, element] = Kokkos::pair<int, int>(host_error_index());
  ASSERT_EQ(iteration, 0) << " failing at " << iteration << ", " << element;
  auto host_view =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
  for (int i = 0; i < N; ++i)
    ASSERT_EQ(host_view(i), 2 * M) << " failing at " << i;
}

TEST(TEST_CATEGORY, exec_space_thread_safety_range_reduce) {
  run_exec_space_thread_safety_range_reduce();
}

void run_exec_space_thread_safety_mdrange_reduce() {
  constexpr int N = 1000;
  constexpr int M = 1000;

  Kokkos::View<int *, TEST_EXECSPACE> view("view", N);
  Kokkos::View<Kokkos::pair<int, int>, TEST_EXECSPACE> error_index(
      "error_index");

  run_threaded_test(
      [=]() {
        int dummy;
        TEST_EXECSPACE exec;
        for (int j = 0; j < M; ++j) {
          Kokkos::parallel_reduce(
              "bar",
              Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>(
                  exec, {0, 0}, {N, 1}),
              KOKKOS_LAMBDA(int i, int, int &) {
                if (i < N - 1 &&
                    (Kokkos::atomic_load(view.data() + (i + 1)) - view(i)) > 1)
                  Kokkos::atomic_store(error_index.data(),
                                       Kokkos::pair{j + 1, i});
                ++view(i);
              },
              dummy);
        }
        exec.fence();
      },
      [=]() {
        int dummy;
        TEST_EXECSPACE exec;
        for (int j = 0; j < M; ++j) {
          Kokkos::parallel_reduce(
              "foo",
              Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>(
                  exec, {0, 0}, {N, 1}),
              KOKKOS_LAMBDA(int i, int, int &) {
                if (i < N - 1 &&
                    (Kokkos::atomic_load(view.data() + (N - 2 - i)) -
                     view(N - 1 - i)) > 1)
                  Kokkos::atomic_store(error_index.data(),
                                       Kokkos::pair{j + 1, N - 1 - i});
                ++view(N - 1 - i);
              },
              dummy);
        }
        exec.fence();
      });

  auto host_error_index =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, error_index);
  auto [iteration, element] = Kokkos::pair<int, int>(host_error_index());
  ASSERT_EQ(iteration, 0) << " failing at " << iteration << ", " << element;
  auto host_view =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
  for (int i = 0; i < N; ++i)
    ASSERT_EQ(host_view(i), 2 * M) << " failing at " << i;
}

TEST(TEST_CATEGORY, exec_space_thread_safety_mdrange_reduce) {
// FIXME_INTEL
#ifdef KOKKOS_COMPILER_INTEL
  if (std::is_same_v<TEST_EXECSPACE, Kokkos::OpenMP>)
    GTEST_SKIP() << "skipping since test is known to fail for OpenMP using the "
                    "legacy Intel compiler";
#endif
  run_exec_space_thread_safety_mdrange_reduce();
}

void run_exec_space_thread_safety_team_policy_reduce() {
  constexpr int N = 1000;
  constexpr int M = 1000;

  Kokkos::View<int *, TEST_EXECSPACE> view("view", N);
  Kokkos::View<int, TEST_EXECSPACE> error_index("error_index");

  run_threaded_test(
      [=]() {
        int dummy;
        TEST_EXECSPACE exec;
        for (int j = 0; j < M; ++j) {
          Kokkos::parallel_reduce(
              "bar", Kokkos::TeamPolicy<TEST_EXECSPACE>(exec, N, 1, 1),
              KOKKOS_LAMBDA(
                  const Kokkos::TeamPolicy<TEST_EXECSPACE>::member_type
                      &team_member,
                  int &) {
                int i = team_member.league_rank();
                if (i < N - 1 &&
                    (Kokkos::atomic_load(view.data() + (i + 1)) - view(i)) > 1)
                  Kokkos::atomic_store(error_index.data(), 1);
                ++view(i);
              },
              dummy);
        }
        exec.fence();
      },
      [=]() {
        int dummy;
        TEST_EXECSPACE exec;
        for (int j = 0; j < M; ++j) {
          Kokkos::parallel_reduce(
              "foo", Kokkos::TeamPolicy<TEST_EXECSPACE>(exec, N, 1, 1),
              KOKKOS_LAMBDA(
                  const Kokkos::TeamPolicy<TEST_EXECSPACE>::member_type
                      &team_member,
                  int &) {
                int i = team_member.league_rank();
                if (i < N - 1 &&
                    (Kokkos::atomic_load(view.data() + (N - 2 - i)) -
                     view(N - 1 - i)) > 1)
                  Kokkos::atomic_store(error_index.data(), 1);
                ++view(N - 1 - i);
              },
              dummy);
        }
        exec.fence();
      });

  auto host_error_index =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, error_index);
  ASSERT_EQ(host_error_index(), 0);
  auto host_view =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
  for (int i = 0; i < N; ++i)
    ASSERT_EQ(host_view(i), 2 * M) << " failing at " << i;
}

TEST(TEST_CATEGORY, exec_space_thread_safety_team_policy_reduce) {
// FIXME_OPENMPTARGET
#ifdef KOKKOS_ENABLE_OPENMPTARGET
  if (std::is_same_v<TEST_EXECSPACE, Kokkos::Experimental::OpenMPTarget>)
    GTEST_SKIP() << "skipping for OpenMPTarget since the test is designed to "
                    "run with vector_length=1";
#endif
    // FIXME_SYCL
#if defined(KOKKOS_ENABLE_SYCL) && defined(KOKKOS_IMPL_ARCH_NVIDIA_GPU)
  if (std::is_same_v<TEST_EXECSPACE, Kokkos::Experimental::SYCL>)
    GTEST_SKIP() << "skipping since test is know to fail with SYCL+Cuda";
#endif
  run_exec_space_thread_safety_team_policy_reduce();
}

void run_exec_space_thread_safety_range_scan() {
  constexpr int N = 1000;
  constexpr int M = 1000;

  Kokkos::View<int *, TEST_EXECSPACE> view("view", N);
  Kokkos::View<Kokkos::pair<int, int>, TEST_EXECSPACE> error_index(
      "error_index");

  run_threaded_test(
      [=]() {
        int dummy;
        TEST_EXECSPACE exec;
        for (int j = 0; j < M; ++j) {
          Kokkos::parallel_scan(
              "bar", Kokkos::RangePolicy<TEST_EXECSPACE>(exec, 0, N),
              KOKKOS_LAMBDA(const int i, int &, const bool final) {
                if (final) {
                  if (i < N - 1 && (Kokkos::atomic_load(view.data() + (i + 1)) -
                                    view(i)) > 1)
                    Kokkos::atomic_store(error_index.data(),
                                         Kokkos::pair{j + 1, i});
                  ++view(i);
                }
              },
              dummy);
        }
        exec.fence();
      },
      [=]() {
        int dummy;
        TEST_EXECSPACE exec;
        for (int j = 0; j < M; ++j) {
          Kokkos::parallel_scan(
              "foo", Kokkos::RangePolicy<TEST_EXECSPACE>(exec, 0, N),
              KOKKOS_LAMBDA(const int i, int &, const bool final) {
                if (final) {
                  if (i < N - 1 &&
                      (Kokkos::atomic_load(view.data() + (N - 2 - i)) -
                       view(N - 1 - i)) > 1)
                    Kokkos::atomic_store(error_index.data(),
                                         Kokkos::pair{j + 1, N - 1 - i});
                  ++view(N - 1 - i);
                }
              },
              dummy);
        }
        exec.fence();
      });

  auto host_error_index =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, error_index);
  auto [iteration, element] = Kokkos::pair<int, int>(host_error_index());
  ASSERT_EQ(iteration, 0) << " failing at " << iteration << ", " << element;
  auto host_view =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
  for (int i = 0; i < N; ++i)
    ASSERT_EQ(host_view(i), 2 * M) << " failing at " << i;
}

TEST(TEST_CATEGORY, exec_space_thread_safety_range_scan) {
  run_exec_space_thread_safety_range_scan();
}

}  // namespace
