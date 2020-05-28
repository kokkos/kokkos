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

#include <cstdio>

#include <Kokkos_Core.hpp>

namespace Kokkos {
namespace Experimental {

template <class ViewAttorneyType>
class ViewHookUpdate<
    ViewAttorneyType,
    typename std::enable_if<
        (std::is_const<typename ViewAttorneyType::view_type>::value ||
         std::is_same<
             Kokkos::AnonymousSpace,
             typename ViewAttorneyType::view_type::memory_space>::value),
        void>::type> {
 public:
  using view_att_type = ViewAttorneyType;

  static inline void update_view(view_att_type &) {}

  static constexpr const char *m_name = "ConstImpl";
};

// need to template off of the attorney class, so that we can update the view
// data handle
template <class ViewAttorneyType>
class ViewHookUpdate<
    ViewAttorneyType,
    typename std::enable_if<
        !(std::is_const<typename ViewAttorneyType::view_type>::value ||
          std::is_same<
              Kokkos::AnonymousSpace,
              typename ViewAttorneyType::view_type::memory_space>::value),
        void>::type> {
 public:
  using view_att_type   = ViewAttorneyType;
  using view_type       = typename view_att_type::view_type;
  using view_track_type = Kokkos::Impl::SharedAllocationHeader;
  static inline void update_view(view_att_type &view) {
    using mem_space    = typename view_type::memory_space;
    using exec_space   = typename mem_space::execution_space;
    using view_traits  = typename view_type::traits;
    using value_type   = typename view_traits::value_type;
    using pointer_type = typename view_traits::value_type *;
    using handle_type =
        typename Kokkos::Impl::ViewDataHandle<view_traits>::handle_type;
    using functor_type = Kokkos::Impl::ViewValueFunctor<exec_space, value_type>;
    using record_type =
        Kokkos::Impl::SharedAllocationRecord<mem_space, functor_type>;

    record_type *orig_rec =
        (record_type *)(view.rec_ptr());  // get the original record

    std::string label = orig_rec->get_label();
    record_type *const record =
        record_type::allocate(mem_space(), label, orig_rec->size());

    // need to assign the new record to the view map / handle
    view.update_data_handle(
        handle_type(reinterpret_cast<pointer_type>(record->data())));

    record->m_destroy = functor_type(
        exec_space(), (value_type *)view.get_view().impl_map().m_impl_handle,
        view.get_view().span(), label);

    // Construct values
    record->m_destroy.construct_shared_allocation();

    // This should disconnect the duplicate view from the original record and
    // attach the duplicated data to the tracker
    view.assign_view_record(record);
  }
  static constexpr const char *m_name = "Non-ConstImpl";
};

}  // namespace Experimental
}  // namespace Kokkos

namespace Test {

namespace {

template <class T, class ExecSpace>
struct TestViewHooks {
  struct ViewHookUpdateFunctor {
    using view_hook_functor_type = Kokkos::Experimental::ViewHookRefFunctor;
    void operator()(Kokkos::Experimental::ViewHolderBase &vt) {
      vt.update_view();
    }
  };
  int N          = 0;
  using mat_type = Kokkos::View<T **, typename ExecSpace::memory_space>;
  mat_type m1;
  using view_type = Kokkos::View<T *, typename ExecSpace::memory_space>;
  view_type v1;
  using range_policy = Kokkos::RangePolicy<ExecSpace>;

  // default view constructor will initialize data to 0
  TestViewHooks(const int n_) : N(n_), m1("m1", N, N), v1("v1", N) {}

  TestViewHooks(const TestViewHooks &rhs) : N(rhs.N), m1(rhs.m1), v1(rhs.v1) {}
  TestViewHooks &operator=(const TestViewHooks &rhs) {
    N  = rhs.N;
    m1 = rhs.m1;
    v1 = rhs.v1;
  }

  // accumulate data
  KOKKOS_FUNCTION
  void operator()(const int i) const {
    v1(i) += 1;
    m1(i, i) += 2;
  }

  void run_test() {
    TestViewHooks tv1(*this);
    // 1, 2 start
    Kokkos::parallel_for("run_functor", range_policy(0, N), tv1);
    Kokkos::fence();

    Kokkos::Experimental::add_view_hook_caller(
        "update",
        [](Kokkos::Experimental::ViewHolderBase &vt) { vt.update_view(); });
    TestViewHooks tv2(*this);
    Kokkos::Experimental::remove_view_hook_caller("update");
    // reset view contents before running functor.
    // 1, 2
    Kokkos::parallel_for("run_functor", range_policy(0, N), tv2);
    Kokkos::fence();

    // 2, 4
    TestViewHooks tv3(*this);
    Kokkos::parallel_for("run_functor", range_policy(0, N), tv3);
    Kokkos::fence();

    typename view_type::HostMirror v1_h = Kokkos::create_mirror_view(v1);
    typename mat_type::HostMirror m1_h  = Kokkos::create_mirror_view(m1);

    Kokkos::deep_copy(v1_h, v1);
    Kokkos::deep_copy(m1_h, m1);

    for (int i = 0; i < N; i++) {
      ASSERT_EQ(v1_h(i), 2);
      ASSERT_EQ(m1_h(i, i), 4);
    }
  }
};

TEST(TEST_CATEGORY, view_hooks) {
  TestViewHooks<long, TEST_EXECSPACE> f(10);
  f.run_test();
}

}  // namespace

}  // namespace Test
