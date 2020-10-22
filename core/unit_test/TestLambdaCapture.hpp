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

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <vector>
#include <string>

namespace Test {

#if defined(KOKKOS_ENABLE_VIEW_HOOKS)

struct SetViewsFunctor {
  std::vector<Kokkos::Experimental::ViewHolderBase *> views;

  SetViewsFunctor() = default;
  SetViewsFunctor(const SetViewsFunctor &rhs) : views(rhs.views) {}
  SetViewsFunctor(SetViewsFunctor &&) = default;

  void operator()(Kokkos::Experimental::ViewHolderBase &view) {
    views.emplace_back(view.clone());
  }
  size_t size() { return views.size(); }
};

template <typename F>
void do_view_list_functor(F &&_fun, SetViewsFunctor &sv) {
  auto vhc = Kokkos::Experimental::ViewHooks::create_view_hook_caller(sv);
  Kokkos::Experimental::ViewHooks::set("functor", vhc);
  auto f = _fun;

  Kokkos::Experimental::ViewHooks::clear("functor", vhc);

  f();
}

template <typename F>
auto get_view_list(F &&_fun) {
  std::vector<std::unique_ptr<Kokkos::Experimental::ViewHolderBase> > views;

  auto vhc = Kokkos::Experimental::ViewHooks::create_view_hook_caller(
      [&views](Kokkos::Experimental::ViewHolderBase &view) {
        views.emplace_back(view.clone());
      },
      [](Kokkos::Experimental::ViewHolderBase &) {});

  Kokkos::Experimental::ViewHooks::set("lambda", vhc);

  auto f = _fun;

  Kokkos::Experimental::ViewHooks::clear("lambda", vhc);

  f();

  return views;
}

template <typename View>
bool capture_list_contains(
    std::vector<std::unique_ptr<Kokkos::Experimental::ViewHolderBase> > &_list,
    View &&_v) {
  auto pos = std::find_if(_list.begin(), _list.end(), [&_v](auto &&_hold) {
    return _hold->data() == _v.data();
  });
  for (std::vector<std::unique_ptr<Kokkos::Experimental::ViewHolderBase> >::
           iterator it = _list.begin();
       it != _list.end(); it++) {
  }
  return pos != _list.end();
}

template <typename View>
bool capture_list_contains(
    std::vector<Kokkos::Experimental::ViewHolderBase *> &_list, View &&_v) {
  auto pos = std::find_if(_list.begin(), _list.end(), [&_v](auto _hold) {
    return _hold->data() == _v.data();
  });
  return pos != _list.end();
}

struct mixed_data {
  mixed_data() : x("test", 5), y(false) {}

  Kokkos::View<double *> x;
  bool y;

  void work() { y = true; };
};

struct FunctorValidator {
  int val = 0;
  mixed_data md;
  FunctorValidator(const int i, mixed_data &_md) : val(i), md(_md) {}

  void operator()() { md.work(); }
};

TEST(LambdaCapture, functor) {
  mixed_data dat;
  FunctorValidator fv(2, dat);

  auto captures = get_view_list(fv);

  EXPECT_FALSE(dat.y);
  EXPECT_TRUE(capture_list_contains(captures, dat.x));
}

TEST(LambdaCapture, value_functor) {
  mixed_data dat;

  SetViewsFunctor sv;
  do_view_list_functor([=]() mutable { dat.work(); }, sv);

  EXPECT_FALSE(dat.y);
  EXPECT_TRUE(capture_list_contains(sv.views, dat.x));
}

TEST(LambdaCapture, value) {
  mixed_data dat;

  auto captures = get_view_list([=]() mutable { dat.work(); });

  EXPECT_FALSE(dat.y);
  EXPECT_TRUE(capture_list_contains(captures, dat.x));
}

TEST(LambdaCapture, reference) {
  mixed_data dat;
  auto &ref = dat;

  auto captures = get_view_list([&]() mutable { ref.work(); });

  EXPECT_TRUE(dat.y);
  EXPECT_FALSE(capture_list_contains(captures, dat.x));
}

TEST(LambdaCapture, clone_holder) {
  auto dat = mixed_data();

  auto holder = Kokkos::Experimental::ViewHolderCopy<decltype(dat.x)>(dat.x);
  auto *h2    = holder.clone();

  EXPECT_EQ(holder.data(), dat.x.data());
  EXPECT_EQ(holder.data(), h2->data());
  EXPECT_EQ(h2->data(), dat.x.data());
}

#endif  // KOKKOS_ENABLE_VIEW_HOOKS
}  // namespace Test
