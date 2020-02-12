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

#ifndef KOKKOS_TEST_DUALVIEW_HPP
#define KOKKOS_TEST_DUALVIEW_HPP

#include <gtest/gtest.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <impl/Kokkos_Timer.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <cmath>
#include <chrono>

namespace Test {

namespace Impl {

// This test runs the random number generators and uses some statistic tests to
// check the 'goodness' of the random numbers:
//    (i)   mean:         the mean is expected to be 0.5*RAND_MAX
//    (ii)  variance:     the variance is 1/3*mean*mean
//    (iii) covariance:   the covariance is 0
//    (iv)  1-tupledistr: the mean, variance and covariance of a 1D Histrogram
//    of random numbers (v)   3-tupledistr: the mean, variance and covariance of
//    a 3D Histrogram of random numbers

#define HIST_DIM3D 24
#define HIST_DIM1D (HIST_DIM3D * HIST_DIM3D * HIST_DIM3D)

struct RandomProperties {
  uint64_t count;
  double mean;
  double variance;
  double covariance;
  double min;
  double max;

  KOKKOS_INLINE_FUNCTION
  RandomProperties() {
    count      = 0;
    mean       = 0.0;
    variance   = 0.0;
    covariance = 0.0;
    min        = 1e64;
    max        = -1e64;
  }

  KOKKOS_INLINE_FUNCTION
  RandomProperties& operator+=(const RandomProperties& add) {
    count += add.count;
    mean += add.mean;
    variance += add.variance;
    covariance += add.covariance;
    min = add.min < min ? add.min : min;
    max = add.max > max ? add.max : max;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  void operator+=(const volatile RandomProperties& add) volatile {
    count += add.count;
    mean += add.mean;
    variance += add.variance;
    covariance += add.covariance;
    min = add.min < min ? add.min : min;
    max = add.max > max ? add.max : max;
  }
};

template <class GeneratorPool, class Scalar>
struct test_random_functor {
  typedef typename GeneratorPool::generator_type rnd_type;

  typedef RandomProperties value_type;
  typedef typename GeneratorPool::device_type device_type;

  GeneratorPool rand_pool;
  const double mean;

  // NOTE (mfh 03 Nov 2014): Kokkos::rand::max() is supposed to define
  // an exclusive upper bound on the range of random numbers that
  // draw() can generate.  However, for the float specialization, some
  // implementations might violate this upper bound, due to rounding
  // error.  Just in case, we leave an extra space at the end of each
  // dimension, in the View types below.
  typedef Kokkos::View<int[HIST_DIM1D + 1], typename GeneratorPool::device_type>
      type_1d;
  type_1d density_1d;
  typedef Kokkos::View<int[HIST_DIM3D + 1][HIST_DIM3D + 1][HIST_DIM3D + 1],
                       typename GeneratorPool::device_type>
      type_3d;
  type_3d density_3d;

  test_random_functor(GeneratorPool rand_pool_, type_1d d1d, type_3d d3d)
      : rand_pool(rand_pool_),
        mean(0.5 * Kokkos::rand<rnd_type, Scalar>::max()),
        density_1d(d1d),
        density_3d(d3d) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(int i, RandomProperties& prop) const {
    using Kokkos::atomic_fetch_add;

    rnd_type rand_gen = rand_pool.get_state();
    for (int k = 0; k < 1024; ++k) {
      const Scalar tmp = Kokkos::rand<rnd_type, Scalar>::draw(rand_gen);
      prop.count++;
      prop.mean += tmp;
      prop.variance += (tmp - mean) * (tmp - mean);
      const Scalar tmp2 = Kokkos::rand<rnd_type, Scalar>::draw(rand_gen);
      prop.count++;
      prop.mean += tmp2;
      prop.variance += (tmp2 - mean) * (tmp2 - mean);
      prop.covariance += (tmp - mean) * (tmp2 - mean);
      const Scalar tmp3 = Kokkos::rand<rnd_type, Scalar>::draw(rand_gen);
      prop.count++;
      prop.mean += tmp3;
      prop.variance += (tmp3 - mean) * (tmp3 - mean);
      prop.covariance += (tmp2 - mean) * (tmp3 - mean);

      // NOTE (mfh 03 Nov 2014): Kokkos::rand::max() is supposed to
      // define an exclusive upper bound on the range of random
      // numbers that draw() can generate.  However, for the float
      // specialization, some implementations might violate this upper
      // bound, due to rounding error.  Just in case, we have left an
      // extra space at the end of each dimension of density_1d and
      // density_3d.
      //
      // Please note that those extra entries might not get counted in
      // the histograms.  However, if Kokkos::rand is broken and only
      // returns values of max(), the histograms will still catch this
      // indirectly, since none of the other values will be filled in.

      const Scalar theMax = Kokkos::rand<rnd_type, Scalar>::max();

      const uint64_t ind1_1d =
          static_cast<uint64_t>(1.0 * HIST_DIM1D * tmp / theMax);
      const uint64_t ind2_1d =
          static_cast<uint64_t>(1.0 * HIST_DIM1D * tmp2 / theMax);
      const uint64_t ind3_1d =
          static_cast<uint64_t>(1.0 * HIST_DIM1D * tmp3 / theMax);

      const uint64_t ind1_3d =
          static_cast<uint64_t>(1.0 * HIST_DIM3D * tmp / theMax);
      const uint64_t ind2_3d =
          static_cast<uint64_t>(1.0 * HIST_DIM3D * tmp2 / theMax);
      const uint64_t ind3_3d =
          static_cast<uint64_t>(1.0 * HIST_DIM3D * tmp3 / theMax);

      atomic_fetch_add(&density_1d(ind1_1d), 1);
      atomic_fetch_add(&density_1d(ind2_1d), 1);
      atomic_fetch_add(&density_1d(ind3_1d), 1);
      atomic_fetch_add(&density_3d(ind1_3d, ind2_3d, ind3_3d), 1);
    }
    rand_pool.free_state(rand_gen);
  }
};

template <class DeviceType>
struct test_histogram1d_functor {
  typedef RandomProperties value_type;
  typedef typename DeviceType::execution_space execution_space;
  typedef typename DeviceType::memory_space memory_space;

  // NOTE (mfh 03 Nov 2014): Kokkos::rand::max() is supposed to define
  // an exclusive upper bound on the range of random numbers that
  // draw() can generate.  However, for the float specialization, some
  // implementations might violate this upper bound, due to rounding
  // error.  Just in case, we leave an extra space at the end of each
  // dimension, in the View type below.
  typedef Kokkos::View<int[HIST_DIM1D + 1], memory_space> type_1d;
  type_1d density_1d;
  double mean;

  test_histogram1d_functor(type_1d d1d, int num_draws)
      : density_1d(d1d), mean(1.0 * num_draws / HIST_DIM1D * 3) {}

  KOKKOS_INLINE_FUNCTION void operator()(
      const typename memory_space::size_type i, RandomProperties& prop) const {
    typedef typename memory_space::size_type size_type;
    const double count = density_1d(i);
    prop.mean += count;
    prop.variance += 1.0 * (count - mean) * (count - mean);
    // prop.covariance += 1.0*count*count;
    prop.min = count < prop.min ? count : prop.min;
    prop.max = count > prop.max ? count : prop.max;
    if (i < static_cast<size_type>(HIST_DIM1D - 1)) {
      prop.covariance += (count - mean) * (density_1d(i + 1) - mean);
    }
  }
};

template <class DeviceType>
struct test_histogram3d_functor {
  typedef RandomProperties value_type;
  typedef typename DeviceType::execution_space execution_space;
  typedef typename DeviceType::memory_space memory_space;

  // NOTE (mfh 03 Nov 2014): Kokkos::rand::max() is supposed to define
  // an exclusive upper bound on the range of random numbers that
  // draw() can generate.  However, for the float specialization, some
  // implementations might violate this upper bound, due to rounding
  // error.  Just in case, we leave an extra space at the end of each
  // dimension, in the View type below.
  typedef Kokkos::View<int[HIST_DIM3D + 1][HIST_DIM3D + 1][HIST_DIM3D + 1],
                       memory_space>
      type_3d;
  type_3d density_3d;
  double mean;

  test_histogram3d_functor(type_3d d3d, int num_draws)
      : density_3d(d3d), mean(1.0 * num_draws / HIST_DIM1D) {}

  KOKKOS_INLINE_FUNCTION void operator()(
      const typename memory_space::size_type i, RandomProperties& prop) const {
    typedef typename memory_space::size_type size_type;
    const double count = density_3d(
        i / (HIST_DIM3D * HIST_DIM3D),
        (i % (HIST_DIM3D * HIST_DIM3D)) / HIST_DIM3D, i % HIST_DIM3D);
    prop.mean += count;
    prop.variance += (count - mean) * (count - mean);
    if (i < static_cast<size_type>(HIST_DIM1D - 1)) {
      const double count_next =
          density_3d((i + 1) / (HIST_DIM3D * HIST_DIM3D),
                     ((i + 1) % (HIST_DIM3D * HIST_DIM3D)) / HIST_DIM3D,
                     (i + 1) % HIST_DIM3D);
      prop.covariance += (count - mean) * (count_next - mean);
    }
  }
};

//
// Templated test that uses the above functors.
//
template <class RandomGenerator, class Scalar>
struct test_random_scalar {
  typedef typename RandomGenerator::generator_type rnd_type;

  int pass_mean, pass_var, pass_covar;
  int pass_hist1d_mean, pass_hist1d_var, pass_hist1d_covar;
  int pass_hist3d_mean, pass_hist3d_var, pass_hist3d_covar;

  test_random_scalar(
      typename test_random_functor<RandomGenerator, int>::type_1d& density_1d,
      typename test_random_functor<RandomGenerator, int>::type_3d& density_3d,
      RandomGenerator& pool, unsigned int num_draws) {
    using Kokkos::parallel_reduce;
    using std::cout;
    using std::endl;

    {
      cout << " -- Testing randomness properties" << endl;

      RandomProperties result;
      typedef test_random_functor<RandomGenerator, Scalar> functor_type;
      parallel_reduce(num_draws / 1024,
                      functor_type(pool, density_1d, density_3d), result);

      // printf("Result: %lf %lf
      // %lf\n",result.mean/num_draws/3,result.variance/num_draws/3,result.covariance/num_draws/2);
      double tolerance       = 1.6 * std::sqrt(1.0 / num_draws);
      double mean_expect     = 0.5 * Kokkos::rand<rnd_type, Scalar>::max();
      double variance_expect = 1.0 / 3.0 * mean_expect * mean_expect;
      double mean_eps = mean_expect / (result.mean / num_draws / 3) - 1.0;
      double variance_eps =
          variance_expect / (result.variance / num_draws / 3) - 1.0;
      double covariance_eps =
          result.covariance / num_draws / 2 / variance_expect;
      pass_mean = ((-tolerance < mean_eps) && (tolerance > mean_eps)) ? 1 : 0;
      pass_var  = ((-1.5 * tolerance < variance_eps) &&
                  (1.5 * tolerance > variance_eps))
                     ? 1
                     : 0;
      pass_covar = ((-2.0 * tolerance < covariance_eps) &&
                    (2.0 * tolerance > covariance_eps))
                       ? 1
                       : 0;
      cout << "Pass: " << pass_mean << " " << pass_var << " " << mean_eps << " "
           << variance_eps << " " << covariance_eps << " || " << tolerance
           << endl;
    }
    {
      cout << " -- Testing 1-D histogram" << endl;

      RandomProperties result;
      typedef test_histogram1d_functor<typename RandomGenerator::device_type>
          functor_type;
      parallel_reduce(HIST_DIM1D, functor_type(density_1d, num_draws), result);

      double tolerance   = 6 * std::sqrt(1.0 / HIST_DIM1D);
      double mean_expect = 1.0 * num_draws * 3 / HIST_DIM1D;
      double variance_expect =
          1.0 * num_draws * 3 / HIST_DIM1D * (1.0 - 1.0 / HIST_DIM1D);
      double covariance_expect = -1.0 * num_draws * 3 / HIST_DIM1D / HIST_DIM1D;
      double mean_eps          = mean_expect / (result.mean / HIST_DIM1D) - 1.0;
      double variance_eps =
          variance_expect / (result.variance / HIST_DIM1D) - 1.0;
      double covariance_eps =
          (result.covariance / HIST_DIM1D - covariance_expect) / mean_expect;
      pass_hist1d_mean = ((-0.0001 < mean_eps) && (0.0001 > mean_eps)) ? 1 : 0;
      pass_hist1d_var =
          ((-0.07 < variance_eps) && (0.07 > variance_eps)) ? 1 : 0;
      pass_hist1d_covar =
          ((-0.06 < covariance_eps) && (0.06 > covariance_eps)) ? 1 : 0;

      cout << "Density 1D: " << mean_eps << " " << variance_eps << " "
           << (result.covariance / HIST_DIM1D / HIST_DIM1D) << " || "
           << tolerance << " " << result.min << " " << result.max << " || "
           << result.variance / HIST_DIM1D << " "
           << 1.0 * num_draws * 3 / HIST_DIM1D * (1.0 - 1.0 / HIST_DIM1D)
           << " || " << result.covariance / HIST_DIM1D << " "
           << -1.0 * num_draws * 3 / HIST_DIM1D / HIST_DIM1D << endl;
    }
    {
      cout << " -- Testing 3-D histogram" << endl;

      RandomProperties result;
      typedef test_histogram3d_functor<typename RandomGenerator::device_type>
          functor_type;
      parallel_reduce(HIST_DIM1D, functor_type(density_3d, num_draws), result);

      double tolerance   = 6 * std::sqrt(1.0 / HIST_DIM1D);
      double mean_expect = 1.0 * num_draws / HIST_DIM1D;
      double variance_expect =
          1.0 * num_draws / HIST_DIM1D * (1.0 - 1.0 / HIST_DIM1D);
      double covariance_expect = -1.0 * num_draws / HIST_DIM1D / HIST_DIM1D;
      double mean_eps          = mean_expect / (result.mean / HIST_DIM1D) - 1.0;
      double variance_eps =
          variance_expect / (result.variance / HIST_DIM1D) - 1.0;
      double covariance_eps =
          (result.covariance / HIST_DIM1D - covariance_expect) / mean_expect;
      pass_hist3d_mean =
          ((-tolerance < mean_eps) && (tolerance > mean_eps)) ? 1 : 0;
      pass_hist3d_var = ((-1.2 * tolerance < variance_eps) &&
                         (1.2 * tolerance > variance_eps))
                            ? 1
                            : 0;
      pass_hist3d_covar =
          ((-tolerance < covariance_eps) && (tolerance > covariance_eps)) ? 1
                                                                          : 0;

      cout << "Density 3D: " << mean_eps << " " << variance_eps << " "
           << result.covariance / HIST_DIM1D / HIST_DIM1D << " || " << tolerance
           << " " << result.min << " " << result.max << endl;
    }
  }
};

template <class RandomGenerator>
void test_random(unsigned int num_draws) {
  using std::cout;
  using std::endl;
  typename test_random_functor<RandomGenerator, int>::type_1d density_1d("D1d");
  typename test_random_functor<RandomGenerator, int>::type_3d density_3d("D3d");

  uint64_t ticks =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  cout << "Test Seed:" << ticks << endl;

  RandomGenerator pool(ticks);

  cout << "Test Scalar=int" << endl;
  test_random_scalar<RandomGenerator, int> test_int(density_1d, density_3d,
                                                    pool, num_draws);
  ASSERT_EQ(test_int.pass_mean, 1);
  ASSERT_EQ(test_int.pass_var, 1);
  ASSERT_EQ(test_int.pass_covar, 1);
  ASSERT_EQ(test_int.pass_hist1d_mean, 1);
  ASSERT_EQ(test_int.pass_hist1d_var, 1);
  ASSERT_EQ(test_int.pass_hist1d_covar, 1);
  ASSERT_EQ(test_int.pass_hist3d_mean, 1);
  ASSERT_EQ(test_int.pass_hist3d_var, 1);
  ASSERT_EQ(test_int.pass_hist3d_covar, 1);
  deep_copy(density_1d, 0);
  deep_copy(density_3d, 0);

  cout << "Test Scalar=unsigned int" << endl;
  test_random_scalar<RandomGenerator, unsigned int> test_uint(
      density_1d, density_3d, pool, num_draws);
  ASSERT_EQ(test_uint.pass_mean, 1);
  ASSERT_EQ(test_uint.pass_var, 1);
  ASSERT_EQ(test_uint.pass_covar, 1);
  ASSERT_EQ(test_uint.pass_hist1d_mean, 1);
  ASSERT_EQ(test_uint.pass_hist1d_var, 1);
  ASSERT_EQ(test_uint.pass_hist1d_covar, 1);
  ASSERT_EQ(test_uint.pass_hist3d_mean, 1);
  ASSERT_EQ(test_uint.pass_hist3d_var, 1);
  ASSERT_EQ(test_uint.pass_hist3d_covar, 1);
  deep_copy(density_1d, 0);
  deep_copy(density_3d, 0);

  cout << "Test Scalar=int64_t" << endl;
  test_random_scalar<RandomGenerator, int64_t> test_int64(
      density_1d, density_3d, pool, num_draws);
  ASSERT_EQ(test_int64.pass_mean, 1);
  ASSERT_EQ(test_int64.pass_var, 1);
  ASSERT_EQ(test_int64.pass_covar, 1);
  ASSERT_EQ(test_int64.pass_hist1d_mean, 1);
  ASSERT_EQ(test_int64.pass_hist1d_var, 1);
  ASSERT_EQ(test_int64.pass_hist1d_covar, 1);
  ASSERT_EQ(test_int64.pass_hist3d_mean, 1);
  ASSERT_EQ(test_int64.pass_hist3d_var, 1);
  ASSERT_EQ(test_int64.pass_hist3d_covar, 1);
  deep_copy(density_1d, 0);
  deep_copy(density_3d, 0);

  cout << "Test Scalar=uint64_t" << endl;
  test_random_scalar<RandomGenerator, uint64_t> test_uint64(
      density_1d, density_3d, pool, num_draws);
  ASSERT_EQ(test_uint64.pass_mean, 1);
  ASSERT_EQ(test_uint64.pass_var, 1);
  ASSERT_EQ(test_uint64.pass_covar, 1);
  ASSERT_EQ(test_uint64.pass_hist1d_mean, 1);
  ASSERT_EQ(test_uint64.pass_hist1d_var, 1);
  ASSERT_EQ(test_uint64.pass_hist1d_covar, 1);
  ASSERT_EQ(test_uint64.pass_hist3d_mean, 1);
  ASSERT_EQ(test_uint64.pass_hist3d_var, 1);
  ASSERT_EQ(test_uint64.pass_hist3d_covar, 1);
  deep_copy(density_1d, 0);
  deep_copy(density_3d, 0);

  cout << "Test Scalar=float" << endl;
  test_random_scalar<RandomGenerator, float> test_float(density_1d, density_3d,
                                                        pool, num_draws);
  ASSERT_EQ(test_float.pass_mean, 1);
  ASSERT_EQ(test_float.pass_var, 1);
  ASSERT_EQ(test_float.pass_covar, 1);
  ASSERT_EQ(test_float.pass_hist1d_mean, 1);
  ASSERT_EQ(test_float.pass_hist1d_var, 1);
  ASSERT_EQ(test_float.pass_hist1d_covar, 1);
  ASSERT_EQ(test_float.pass_hist3d_mean, 1);
  ASSERT_EQ(test_float.pass_hist3d_var, 1);
  ASSERT_EQ(test_float.pass_hist3d_covar, 1);
  deep_copy(density_1d, 0);
  deep_copy(density_3d, 0);

  cout << "Test Scalar=double" << endl;
  test_random_scalar<RandomGenerator, double> test_double(
      density_1d, density_3d, pool, num_draws);
  ASSERT_EQ(test_double.pass_mean, 1);
  ASSERT_EQ(test_double.pass_var, 1);
  ASSERT_EQ(test_double.pass_covar, 1);
  ASSERT_EQ(test_double.pass_hist1d_mean, 1);
  ASSERT_EQ(test_double.pass_hist1d_var, 1);
  ASSERT_EQ(test_double.pass_hist1d_covar, 1);
  ASSERT_EQ(test_double.pass_hist3d_mean, 1);
  ASSERT_EQ(test_double.pass_hist3d_var, 1);
  ASSERT_EQ(test_double.pass_hist3d_covar, 1);
}
}  // namespace Impl

}  // namespace Test

#endif  // KOKKOS_TEST_UNORDERED_MAP_HPP
