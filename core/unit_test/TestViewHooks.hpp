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
#ifndef TESTVIEWHOOKS_HPP_
#define TESTVIEWHOOKS_HPP_

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_View.hpp>
#include <View/Hooks/Kokkos_ViewHooks.hpp>
#include <Kokkos_ViewHolder.hpp>

namespace TestViewHooks {
struct TestSubscriber;

static_assert(
    Kokkos::Experimental::is_hooks_policy<
        Kokkos::Experimental::SubscribableViewHooks<TestSubscriber> >::value,
    "Must be a hooks policy");

using test_view_type =
    Kokkos::View<double **,
                 Kokkos::Experimental::SubscribableViewHooks<TestSubscriber> >;

struct TestSubscriber {
  static test_view_type *self_ptr;
  static const test_view_type *other_ptr;

  template <typename View>
  static void copy_constructed(View &self, const View &other) {
    self_ptr  = &self;
    other_ptr = &other;
  }

  template <typename View>
  static void move_constructed(View &self, const View &other) {
    self_ptr  = &self;
    other_ptr = &other;
  }

  template <typename View>
  static void copy_assigned(View &self, const View &other) {
    self_ptr  = &self;
    other_ptr = &other;
  }

  template <typename View>
  static void move_assigned(View &self, const View &other) {
    self_ptr  = &self;
    other_ptr = &other;
  }

  static void reset() {
    self_ptr  = nullptr;
    other_ptr = nullptr;
  }
};

test_view_type *TestSubscriber::self_ptr        = nullptr;
const test_view_type *TestSubscriber::other_ptr = nullptr;

template <class DeviceType>
void testViewHooksCopyConstruct() {
  TestSubscriber::reset();
  test_view_type testa;

  test_view_type testb(testa);
  EXPECT_EQ(TestSubscriber::self_ptr, &testb);
  EXPECT_EQ(TestSubscriber::other_ptr, &testa);
}

template <class DeviceType>
void testViewHooksMoveConstruct() {
  TestSubscriber::reset();
  test_view_type testa;

  test_view_type testb(std::move(testa));
  EXPECT_EQ(TestSubscriber::self_ptr, &testb);

  // This is valid, even if the view is moved-from
  EXPECT_EQ(TestSubscriber::other_ptr, &testa);
}

template <class DeviceType>
void testViewHooksCopyAssign() {
  TestSubscriber::reset();
  test_view_type testa;

  test_view_type testb;
  testb = testa;
  EXPECT_EQ(TestSubscriber::self_ptr, &testb);
  EXPECT_EQ(TestSubscriber::other_ptr, &testa);
}

template <class DeviceType>
void testViewHooksMoveAssign() {
  TestSubscriber::reset();
  test_view_type testa;

  test_view_type testb;
  testb = std::move(testa);
  EXPECT_EQ(TestSubscriber::self_ptr, &testb);

  // This is valid, even if the view is moved-from
  EXPECT_EQ(TestSubscriber::other_ptr, &testa);
}
}  // namespace TestViewHooks

namespace TestDynamicViewHooks {

using test_view_type =
    Kokkos::View<double **,
                 Kokkos::Experimental::SubscribableViewHooks<
                     Kokkos::Experimental::DynamicViewHooksSubscriber> >;
using const_test_view_type =
    Kokkos::View<const double **,
                 Kokkos::Experimental::SubscribableViewHooks<
                     Kokkos::Experimental::DynamicViewHooksSubscriber> >;

template <class DeviceType>
void testDynamicViewHooksCopyConstruct() {
  Kokkos::Experimental::ViewHolder holder;
  Kokkos::Experimental::ConstViewHolder const_holder;

  Kokkos::Experimental::DynamicViewHooks::reset();

  Kokkos::Experimental::DynamicViewHooks::copy_constructor_set.set_callback(
      [&holder](const Kokkos::Experimental::ViewHolder &vh) mutable {
        holder = vh;
      });

  Kokkos::Experimental::DynamicViewHooks::copy_constructor_set
      .set_const_callback(
          [&const_holder](
              const Kokkos::Experimental::ConstViewHolder &vh) mutable {
            const_holder = vh;
          });

  test_view_type testa("testa", 10, 10);
  test_view_type testb(testa);
  EXPECT_EQ(testa.data(), holder.data());
  EXPECT_EQ(const_holder.data(), nullptr);
  const_test_view_type testa_const(
      testa);  // Won't trigger the callback since this is not a copy
               // constructor call
  const_test_view_type testb_const(testa_const);
  EXPECT_EQ(testa_const.data(), const_holder.data());
}

template <class DeviceType>
void testDynamicViewHooksMoveConstruct() {
  Kokkos::Experimental::ViewHolder holder;
  Kokkos::Experimental::ConstViewHolder const_holder;

  Kokkos::Experimental::DynamicViewHooks::reset();

  Kokkos::Experimental::DynamicViewHooks::move_constructor_set.set_callback(
      [&holder](const Kokkos::Experimental::ViewHolder &vh) mutable {
        holder = vh;
      });

  Kokkos::Experimental::DynamicViewHooks::move_constructor_set
      .set_const_callback(
          [&const_holder](
              const Kokkos::Experimental::ConstViewHolder &vh) mutable {
            const_holder = vh;
          });

  test_view_type testa("testa", 10, 10);
  void *cmp = testa.data();
  test_view_type testb(std::move(testa));
  EXPECT_EQ(cmp, holder.data());
  EXPECT_EQ(const_holder.data(), nullptr);
  const_test_view_type testa_const(
      testb);  // Won't trigger the callback since this is not a copy
               // constructor call
  const_test_view_type testb_const(std::move(testa_const));
  EXPECT_EQ(cmp, const_holder.data());
}

template <class DeviceType>
void testDynamicViewHooksCopyAssign() {
  Kokkos::Experimental::ViewHolder holder;
  Kokkos::Experimental::ConstViewHolder const_holder;

  Kokkos::Experimental::DynamicViewHooks::reset();

  Kokkos::Experimental::DynamicViewHooks::copy_assignment_set.set_callback(
      [&holder](const Kokkos::Experimental::ViewHolder &vh) mutable {
        holder = vh;
      });

  Kokkos::Experimental::DynamicViewHooks::copy_assignment_set
      .set_const_callback(
          [&const_holder](
              const Kokkos::Experimental::ConstViewHolder &vh) mutable {
            const_holder = vh;
          });

  test_view_type testa("testa", 10, 10);
  test_view_type testb;
  testb = testa;
  EXPECT_EQ(testa.data(), holder.data());
  EXPECT_EQ(const_holder.data(), nullptr);
  const_test_view_type testa_const(
      testa);  // Won't trigger the callback since this is not a copy
               // constructor call
  const_test_view_type testb_const;
  testb_const = testa_const;
  EXPECT_EQ(testa_const.data(), const_holder.data());
}

template <class DeviceType>
void testDynamicViewHooksMoveAssign() {
  Kokkos::Experimental::ViewHolder holder;
  Kokkos::Experimental::ConstViewHolder const_holder;

  Kokkos::Experimental::DynamicViewHooks::reset();

  Kokkos::Experimental::DynamicViewHooks::move_assignment_set.set_callback(
      [&holder](const Kokkos::Experimental::ViewHolder &vh) mutable {
        holder = vh;
      });

  Kokkos::Experimental::DynamicViewHooks::move_assignment_set
      .set_const_callback(
          [&const_holder](
              const Kokkos::Experimental::ConstViewHolder &vh) mutable {
            const_holder = vh;
          });

  test_view_type testa("testa", 10, 10);
  void *cmp = testa.data();
  test_view_type testb;
  testb = std::move(testa);
  EXPECT_EQ(cmp, holder.data());
  EXPECT_EQ(const_holder.data(), nullptr);
  const_test_view_type testa_const(
      testa);  // Won't trigger the callback since this is not a copy
               // constructor call
  const_test_view_type testb_const;
  testb_const = std::move(testa_const);
  EXPECT_EQ(cmp, const_holder.data());
}
}  // namespace TestDynamicViewHooks

namespace Test {
TEST(TEST_CATEGORY, view_hooks) {
  using ExecSpace = TEST_EXECSPACE;
  TestViewHooks::testViewHooksCopyConstruct<ExecSpace>();
  TestViewHooks::testViewHooksMoveConstruct<ExecSpace>();
  TestViewHooks::testViewHooksCopyAssign<ExecSpace>();
  TestViewHooks::testViewHooksMoveAssign<ExecSpace>();
}

TEST(TEST_CATEGORY, dynamic_view_hooks) {
  using ExecSpace = TEST_EXECSPACE;
  TestDynamicViewHooks::testDynamicViewHooksCopyConstruct<ExecSpace>();
  TestDynamicViewHooks::testDynamicViewHooksMoveConstruct<ExecSpace>();
  TestDynamicViewHooks::testDynamicViewHooksCopyAssign<ExecSpace>();
  TestDynamicViewHooks::testDynamicViewHooksMoveAssign<ExecSpace>();
}

}  // namespace Test
#endif  // TESTVIEWHOOKS_HPP_
