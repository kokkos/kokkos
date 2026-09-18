// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>
#include <Kokkos_Random.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Core.hpp>

template <class Space, class FloatType>
struct TestFloatPrinting {
  TestFloatPrinting() { run(); }
  void run() const {
    int errors = 0;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<Space>(0, 1), *this, errors);
    ASSERT_EQ(errors, 0);
  }

#define CHECK(val, ref) to_chars_helper_f(val, ref, __FILE__, __LINE__)

  KOKKOS_FUNCTION constexpr int to_chars_helper_f(FloatType val,
                                                  char const* ref,
                                                  char const* file,
                                                  int line) const {
    using Kokkos::Impl::strcmp;
    using Kokkos::Impl::strlen;
    using Kokkos::Impl::to_chars_f;
    constexpr int BUFFER_SIZE = 21;

    int errors = 0;
    char buffer[BUFFER_SIZE];
    char* ptr      = to_chars_f(buffer, buffer + BUFFER_SIZE, val).ptr;
    *ptr           = '\0';
    int ref_length = strlen(ref);
    if (buffer + ref_length != ptr) {
      Kokkos::printf("Error %s:%i:\n", file, line);
      Kokkos::printf(
          "  String size of reference (%s) is %i while size of result (%s) is "
          "%lu\n",
          ref, ref_length, buffer, ptr - buffer);
      if constexpr (std::is_same_v<FloatType, float>) {
        Kokkos::printf("For float %s (0x%x)\n", ref,
                       Kokkos::bit_cast<uint32_t>(val));
      } else {
        Kokkos::printf("For double %s (0x%lx)\n", ref,
                       Kokkos::bit_cast<uint64_t>(val));
      }
      ++errors;
    }

    if (strcmp(buffer, ref) != 0) {
      Kokkos::printf("Error %s:%i:\n", file, line);
      Kokkos::printf("  %s != %s\n", buffer, ref);
      if constexpr (std::is_same_v<FloatType, float>) {
        Kokkos::printf("For float %s (0x%x)\n", ref,
                       Kokkos::bit_cast<uint32_t>(val));
      } else {
        Kokkos::printf("For double %s (0x%lx)\n", ref,
                       Kokkos::bit_cast<uint64_t>(val));
      }
      ++errors;
    }

    return errors;
  }

  KOKKOS_FUNCTION void operator()(int, int& errors) const {
    errors += CHECK(Kokkos::infinity_v<FloatType>, "inf");
    errors += CHECK(-Kokkos::infinity_v<FloatType>, "-inf");
    errors += CHECK(Kokkos::quiet_NaN_v<FloatType>, "nan");
    errors += CHECK(-Kokkos::quiet_NaN_v<FloatType>, "-nan");
#if !(defined(KOKKOS_COMPILER_NVHPC) && defined(KOKKOS_ENABLE_OPENACC))
    // With nvhpc + OpenACC, signaling NaN becomes +inf when copied into a
    // variable
    errors += CHECK(Kokkos::signaling_NaN_v<FloatType>, "nan");
#endif
    errors += CHECK(-Kokkos::signaling_NaN_v<FloatType>, "-nan");
    errors += CHECK(FloatType(0.), "0.000000e+00");
    errors += CHECK(FloatType(-0.), "-0.000000e+00");
    errors += CHECK(FloatType(1.), "1.000000e+00");
    errors += CHECK(FloatType(-1.), "-1.000000e+00");
    errors += CHECK(FloatType(1.2), "1.200000e+00");
    errors += CHECK(FloatType(-1.2), "-1.200000e+00");
    errors += CHECK(FloatType(5.), "5.000000e+00");
    errors += CHECK(FloatType(50.), "5.000000e+01");
    errors += CHECK(FloatType(0.5), "5.000000e-01");
    errors += CHECK(FloatType(6e20), "6.000000e+20");
    errors += CHECK(FloatType(6e37), "6.000000e+37");

    // Because of ties-to-even, 10000005 should be rounded down and 10000015
    // should be rounded up.
    errors += CHECK(FloatType(10000005.0), "1.000000e+07");
    errors += CHECK(FloatType(10000015.0), "1.000002e+07");

    if constexpr (std::is_same_v<FloatType, double>) {
      errors += CHECK(FloatType(1e100), "1.000000e+100");
      errors += CHECK(Kokkos::denorm_min_v<FloatType>, "4.940656e-324");
      errors += CHECK(-Kokkos::denorm_min_v<FloatType>, "-4.940656e-324");
      errors += CHECK(Kokkos::finite_max_v<FloatType>, "1.797693e+308");
      errors += CHECK(-Kokkos::finite_max_v<FloatType>, "-1.797693e+308");

      // Numbers just before or after the point were we go from 12 characters
      // to 13.
      errors +=
          CHECK(Kokkos::bit_cast<double>(0x2b617f7d402b1835), "1.000000e-99");
      errors +=
          CHECK(Kokkos::bit_cast<double>(0x2b617f7d402b1834), "1.000000e-99");
      errors +=
          CHECK(Kokkos::bit_cast<double>(0x2b617f7d402b1833), "9.999999e-100");
      errors +=
          CHECK(Kokkos::bit_cast<double>(0x2b617f7d402b1832), "9.999999e-100");

      errors +=
          CHECK(Kokkos::bit_cast<double>(0x54b249ad163d7d24), "9.999999e+99");
      errors +=
          CHECK(Kokkos::bit_cast<double>(0x54b249ad163d7d25), "9.999999e+99");
      errors +=
          CHECK(Kokkos::bit_cast<double>(0x54b249ad163d7d26), "1.000000e+100");
      errors +=
          CHECK(Kokkos::bit_cast<double>(0x54b249ad163d7d27), "1.000000e+100");

      // Round to nearest, ties to even: 0x51c9bcdf962a0f9f is exactly
      // "9.9999995E85", it should round up)
      errors +=
          CHECK(Kokkos::bit_cast<double>(0x51c9bcdf962a0f9e), "9.999999e+85");
      errors +=
          CHECK(Kokkos::bit_cast<double>(0x51c9bcdf962a0f9f), "1.000000e+86");

      // Numbers that were incorrectly displayed at some point during the
      // debugging process
      errors +=
          CHECK(Kokkos::bit_cast<double>(0x0000080000000003), "4.345847e-311");
      errors +=
          CHECK(Kokkos::bit_cast<double>(0x00072800000002AF), "9.951990e-309");
      errors +=
          CHECK(Kokkos::bit_cast<double>(0x1cef5f80c024cc14), "2.597821e-169");
    }

    if constexpr (std::is_same_v<FloatType, float>) {
      errors += CHECK(Kokkos::denorm_min_v<FloatType>, "1.401298e-45");
      errors += CHECK(-Kokkos::denorm_min_v<FloatType>, "-1.401298e-45");
      errors += CHECK(Kokkos::finite_max_v<FloatType>, "3.402823e+38");
      errors += CHECK(-Kokkos::finite_max_v<FloatType>, "-3.402823e+38");

      // Numbers that were incorrectly displayed at some point during the
      // debugging process
      errors += CHECK(Kokkos::bit_cast<float>(0x9aab), "5.548441e-41");
    }
  }
};

// #define FLOAT_DEVELOPMENT_TESTS
#ifdef FLOAT_DEVELOPMENT_TESTS
// This test will check that the output of Kokkos::to_chars_f(d) is identical
// to the result of std::printf("%e", d) for a fraction of the existing double,
// it can be very long (even when testing only 1/2^31 doubles, it will still
// take hours), so it should only be run when checking the results of changes
// in Kokkos::to_chars_f and not when doing standard CI.
// It can only run on host since it needs to run sprintf().
template <typename T>
bool check(T d) {
  bool ret = true;

  char ref[30] = {};
  sprintf(ref, "%e", d);

  char buffer[30] = {};
  using Kokkos::Impl::to_chars_f;
  to_chars_f(buffer, buffer + 30, d);

  int i = 0;
  while (ref[i] != '\0') {
    if (ref[i] != buffer[i]) {
      char err[512];
      char* err_buf = err;
      if constexpr (std::is_same_v<T, float>) {
        err_buf += sprintf(err_buf, "0x%x\n", Kokkos::bit_cast<uint32_t>(d));
      } else {
        err_buf += sprintf(err_buf, "0x%lx\n", Kokkos::bit_cast<uint64_t>(d));
      }
      err_buf += sprintf(err_buf, "%s !=\n", ref);
      err_buf += sprintf(err_buf, "%s\n", buffer);

      for (int j = 0; j < i; ++j) {
        err_buf += sprintf(err_buf, " ");
      }
      err_buf += sprintf(err_buf, "^\n");
      printf("%s", err);

      ret = false;
      break;
    }
    ++i;
  }

  return ret;
}

void test_all_floats() {
  uint32_t max = ~0x0u;

  Kokkos::View<uint64_t, Kokkos::DefaultHostExecutionSpace::memory_space> total(
      "total", 1);

  Kokkos::deep_copy(total, 0);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, max),
      KOKKOS_LAMBDA(uint32_t counter) {
        float f = Kokkos::bit_cast<float>(counter);

        ASSERT_TRUE(check(f));

        uint64_t tmp = Kokkos::atomic_inc_fetch(&total());
        if (tmp % 1'000'000 == 0) {
          Kokkos::printf("%f%% (%llu/%llu) - %e\n",
                         (double)tmp / (double)max * 100., tmp, max, f);
        }
      });
}

void do_random_test() {
  Kokkos::Random_XorShift64_Pool<Kokkos::DefaultHostExecutionSpace> random_pool(
      /*seed=*/12345);

  uint64_t max = 0x1llu << 30;
  Kokkos::View<uint64_t, Kokkos::DefaultHostExecutionSpace::memory_space> total(
      "total", 1);

  Kokkos::deep_copy(total, 0);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, max),
      KOKKOS_LAMBDA(uint64_t counter) {
        auto generator = random_pool.get_state();
        double d       = Kokkos::bit_cast<double>(generator.urand64());
        random_pool.free_state(generator);

        ASSERT_TRUE(check(d));

        uint64_t tmp = Kokkos::atomic_inc_fetch(&total());
        if (tmp % 1'000'000 == 0) {
          Kokkos::printf("%f%% (%llu/%llu) - %e\n",
                         (double)tmp / (double)max * 100., tmp, max, d);
        }
      });
}
#endif

TEST(TEST_CATEGORY, FloatPrinting) {
  TestFloatPrinting<TEST_EXECSPACE, double>();
  TestFloatPrinting<TEST_EXECSPACE, float>();

#ifdef FLOAT_DEVELOPMENT_TESTS
  test_all_floats();
  do_random_test();
#endif
}
