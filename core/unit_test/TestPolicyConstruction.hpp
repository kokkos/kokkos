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
#include <sstream>
#include <iostream>
#include <type_traits>

namespace Test {
struct SomeExecutionSpace {
  using execution_space = SomeExecutionSpace;
  using size_type       = size_t;
  static constexpr int concurrency() { return 0; }
};
static_assert(Kokkos::is_execution_space_v<SomeExecutionSpace>);

}  // namespace Test

namespace Kokkos::Impl {

template <typename... Properties>
struct TeamPolicyInternal<Test::SomeExecutionSpace, Properties...> {
  template <typename... Args>
  TeamPolicyInternal(Args&&...) {}
};

}  // namespace Kokkos::Impl

namespace Test {

struct ImplicitlyConvertibleToDefaultExecutionSpace {
  operator Kokkos::DefaultExecutionSpace() const {
    return Kokkos::DefaultExecutionSpace();
  }
};
static_assert(!Kokkos::is_execution_space_v<
              ImplicitlyConvertibleToDefaultExecutionSpace>);

struct SomeTag {};

template <class ExecutionSpace>
class TestRangePolicyConstruction {
 public:
  TestRangePolicyConstruction() {
    test_compile_time_parameters();
    test_compile_time_deduction_guides();
    test_runtime_parameters();
  }

 private:
  void test_compile_time_parameters() {
    {
      using policy_t        = Kokkos::RangePolicy<>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((
          std::is_same<execution_space, Kokkos::DefaultExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type,
                                typename execution_space::size_type>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Static>>::value));
      ASSERT_TRUE((std::is_void<work_tag>::value));
    }

    {
      using policy_t        = Kokkos::RangePolicy<ExecutionSpace>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((std::is_same<execution_space, ExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type,
                                typename execution_space::size_type>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Static>>::value));
      ASSERT_TRUE((std::is_void<work_tag>::value));
    }

    {
      using policy_t        = Kokkos::RangePolicy<ExecutionSpace,
                                           Kokkos::Schedule<Kokkos::Dynamic>>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((std::is_same<execution_space, ExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type,
                                typename execution_space::size_type>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_void<work_tag>::value));
    }

    {
      using policy_t =
          Kokkos::RangePolicy<ExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>,
                              Kokkos::IndexType<long>>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((std::is_same<execution_space, ExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_void<work_tag>::value));
    }

    {
      using policy_t =
          Kokkos::RangePolicy<Kokkos::IndexType<long>, ExecutionSpace,
                              Kokkos::Schedule<Kokkos::Dynamic>>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((std::is_same<execution_space, ExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_void<work_tag>::value));
    }

    {
      using policy_t =
          Kokkos::RangePolicy<ExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>,
                              Kokkos::IndexType<long>, SomeTag>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((std::is_same<execution_space, ExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_same<work_tag, SomeTag>::value));
    }

    {
      using policy_t =
          Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>, ExecutionSpace,
                              Kokkos::IndexType<long>, SomeTag>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((std::is_same<execution_space, ExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_same<work_tag, SomeTag>::value));
    }

    {
      using policy_t =
          Kokkos::RangePolicy<SomeTag, Kokkos::Schedule<Kokkos::Dynamic>,
                              Kokkos::IndexType<long>, ExecutionSpace>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((std::is_same<execution_space, ExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_same<work_tag, SomeTag>::value));
    }

    {
      using policy_t = Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((
          std::is_same<execution_space, Kokkos::DefaultExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type,
                                typename execution_space::size_type>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_void<work_tag>::value));
    }

    {
      using policy_t = Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>,
                                           Kokkos::IndexType<long>>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((
          std::is_same<execution_space, Kokkos::DefaultExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_void<work_tag>::value));
    }

    {
      using policy_t        = Kokkos::RangePolicy<Kokkos::IndexType<long>,
                                           Kokkos::Schedule<Kokkos::Dynamic>>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((
          std::is_same<execution_space, Kokkos::DefaultExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_void<work_tag>::value));
    }

    {
      using policy_t = Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>,
                                           Kokkos::IndexType<long>, SomeTag>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((
          std::is_same<execution_space, Kokkos::DefaultExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_same<work_tag, SomeTag>::value));
    }

    {
      using policy_t = Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>,
                                           Kokkos::IndexType<long>, SomeTag>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((
          std::is_same<execution_space, Kokkos::DefaultExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_same<work_tag, SomeTag>::value));
    }

    {
      using policy_t =
          Kokkos::RangePolicy<SomeTag, Kokkos::Schedule<Kokkos::Dynamic>,
                              Kokkos::IndexType<long>>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((
          std::is_same<execution_space, Kokkos::DefaultExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_same<work_tag, SomeTag>::value));
    }
  }

  void test_compile_time_deduction_guides() {
    Kokkos::DefaultExecutionSpace des{};
    ImplicitlyConvertibleToDefaultExecutionSpace notEs{};
    SomeExecutionSpace ses{};
    ExecutionSpace es{};

    using RangePolicyExecSpace = std::conditional_t<
        std::is_same_v<ExecutionSpace, Kokkos::DefaultExecutionSpace>,
        Kokkos::RangePolicy<>, Kokkos::RangePolicy<ExecutionSpace>>;

    int64_t i64{};
    int32_t i32{};
    Kokkos::ChunkSize cs{0};

    // Copy/move
    Kokkos::RangePolicy<SomeExecutionSpace> rptc;
    Kokkos::RangePolicy rpdc(rptc);
    ASSERT_TRUE((std::is_same_v<decltype(rptc), decltype(rpdc)>));

    Kokkos::RangePolicy<SomeExecutionSpace> rptm;
    Kokkos::RangePolicy rpdm(std::move(rptm));
    ASSERT_TRUE((std::is_same_v<decltype(rptm), decltype(rpdm)>));

    // RangePolicy()

    Kokkos::RangePolicy<> pt0;
    Kokkos::RangePolicy pd0;
    ASSERT_TRUE((std::is_same_v<decltype(pd0), decltype(pt0)>));

    // RangePolicy(execution_space, index_type, index_type)

    Kokkos::RangePolicy<> pt1(des, i64, i64);
    Kokkos::RangePolicy pd1(des, i64, i64);
    ASSERT_TRUE((std::is_same_v<decltype(pd1), decltype(pt1)>));

    Kokkos::RangePolicy<> pt2(notEs, i64, i64);
    Kokkos::RangePolicy pd2(notEs, i64, i64);
    ASSERT_TRUE((std::is_same_v<decltype(pd2), decltype(pt2)>));

    Kokkos::RangePolicy<SomeExecutionSpace> pt3(ses, i64, i64);
    Kokkos::RangePolicy pd3(ses, i64, i64);
    ASSERT_TRUE((std::is_same_v<decltype(pd3), decltype(pt3)>));

    RangePolicyExecSpace pt4(es, i64, i64);
    Kokkos::RangePolicy pd4(es, i64, i64);
    ASSERT_TRUE((std::is_same_v<decltype(pd4), decltype(pt4)>));

    Kokkos::RangePolicy<> pt5(des, i32, i32);
    Kokkos::RangePolicy pd5(des, i32, i32);
    ASSERT_TRUE((std::is_same_v<decltype(pd5), decltype(pt5)>));

    Kokkos::RangePolicy<> pt6(notEs, i32, i32);
    Kokkos::RangePolicy pd6(notEs, i32, i32);
    ASSERT_TRUE((std::is_same_v<decltype(pd6), decltype(pt6)>));

    Kokkos::RangePolicy<SomeExecutionSpace> pt7(ses, i32, i32);
    Kokkos::RangePolicy pd7(ses, i32, i32);
    ASSERT_TRUE((std::is_same_v<decltype(pd7), decltype(pt7)>));

    RangePolicyExecSpace pt8(es, i32, i32);
    Kokkos::RangePolicy pd8(es, i32, i32);
    ASSERT_TRUE((std::is_same_v<decltype(pd8), decltype(pt8)>));

    // RangePolicy(index_type, index_type)

    Kokkos::RangePolicy<> pt9(i64, i64);
    Kokkos::RangePolicy pd9(i64, i64);
    ASSERT_TRUE((std::is_same_v<decltype(pd9), decltype(pt9)>));

    Kokkos::RangePolicy<> pt10(i32, i32);
    Kokkos::RangePolicy pd10(i32, i32);
    ASSERT_TRUE((std::is_same_v<decltype(pd10), decltype(pt10)>));

    // RangePolicy(execution_space, index_type, index_type, Args...)

    Kokkos::RangePolicy<> pt11(des, i64, i64, cs);
    Kokkos::RangePolicy pd11(des, i64, i64, cs);
    ASSERT_TRUE((std::is_same_v<decltype(pd11), decltype(pt11)>));

    Kokkos::RangePolicy<> pt12(notEs, i64, i64, cs);
    Kokkos::RangePolicy pd12(notEs, i64, i64, cs);
    ASSERT_TRUE((std::is_same_v<decltype(pd12), decltype(pt12)>));

    Kokkos::RangePolicy<SomeExecutionSpace> pt13(ses, i64, i64, cs);
    Kokkos::RangePolicy pd13(ses, i64, i64, cs);
    ASSERT_TRUE((std::is_same_v<decltype(pd13), decltype(pt13)>));

    RangePolicyExecSpace pt14(es, i64, i64, cs);
    Kokkos::RangePolicy pd14(es, i64, i64, cs);
    ASSERT_TRUE((std::is_same_v<decltype(pd14), decltype(pt14)>));

    Kokkos::RangePolicy<> pt15(des, i32, i32, cs);
    Kokkos::RangePolicy pd15(des, i32, i32, cs);
    ASSERT_TRUE((std::is_same_v<decltype(pd15), decltype(pt15)>));

    Kokkos::RangePolicy<> pt16(notEs, i32, i32, cs);
    Kokkos::RangePolicy pd16(notEs, i32, i32, cs);
    ASSERT_TRUE((std::is_same_v<decltype(pd16), decltype(pt16)>));

    Kokkos::RangePolicy<SomeExecutionSpace> pt17(ses, i32, i32, cs);
    Kokkos::RangePolicy pd17(ses, i32, i32, cs);
    ASSERT_TRUE((std::is_same_v<decltype(pd17), decltype(pt17)>));

    RangePolicyExecSpace pt18(es, i32, i32, cs);
    Kokkos::RangePolicy pd18(es, i32, i32, cs);
    ASSERT_TRUE((std::is_same_v<decltype(pd18), decltype(pt18)>));

    // RangePolicy(index_type, index_type, Args...)

    Kokkos::RangePolicy<> pt19(i64, i64, cs);
    Kokkos::RangePolicy pd19(i64, i64, cs);
    ASSERT_TRUE((std::is_same_v<decltype(pt19), decltype(pd19)>));

    Kokkos::RangePolicy<> pt20(i32, i32, cs);
    Kokkos::RangePolicy pd20(i32, i32, cs);
    ASSERT_TRUE((std::is_same_v<decltype(pd20), decltype(pt20)>));
  }

  void test_runtime_parameters() {
    using policy_t     = Kokkos::RangePolicy<>;
    using index_t      = policy_t::index_type;
    index_t work_begin = 5;
    index_t work_end   = 15;
    index_t chunk_size = 10;
    {
      policy_t p(work_begin, work_end);
      ASSERT_EQ(p.begin(), work_begin);
      ASSERT_EQ(p.end(), work_end);
    }
    {
      policy_t p(Kokkos::DefaultExecutionSpace(), work_begin, work_end);
      ASSERT_EQ(p.begin(), work_begin);
      ASSERT_EQ(p.end(), work_end);
    }
    {
      policy_t p(work_begin, work_end, Kokkos::ChunkSize(chunk_size));
      ASSERT_EQ(p.begin(), work_begin);
      ASSERT_EQ(p.end(), work_end);
      ASSERT_EQ(p.chunk_size(), chunk_size);
    }
    {
      policy_t p(Kokkos::DefaultExecutionSpace(), work_begin, work_end,
                 Kokkos::ChunkSize(chunk_size));
      ASSERT_EQ(p.begin(), work_begin);
      ASSERT_EQ(p.end(), work_end);
      ASSERT_EQ(p.chunk_size(), chunk_size);
    }
    {
      policy_t p;
      ASSERT_EQ(p.begin(), index_t(0));
      ASSERT_EQ(p.end(), index_t(0));
      p = policy_t(work_begin, work_end, Kokkos::ChunkSize(chunk_size));
      ASSERT_EQ(p.begin(), work_begin);
      ASSERT_EQ(p.end(), work_end);
      ASSERT_EQ(p.chunk_size(), chunk_size);
    }
  }
};

template <class ExecutionSpace>
class TestTeamPolicyConstruction {
 public:
  TestTeamPolicyConstruction() {
    test_compile_time_parameters();
    test_compile_time_deduction_guides();
    test_run_time_parameters();
  }

 private:
  void test_compile_time_parameters() {
    {
      using policy_t        = Kokkos::TeamPolicy<>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((
          std::is_same<execution_space, Kokkos::DefaultExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type,
                                typename execution_space::size_type>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Static>>::value));
      ASSERT_TRUE((std::is_void<work_tag>::value));
    }

    {
      using policy_t        = Kokkos::TeamPolicy<ExecutionSpace>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((std::is_same<execution_space, ExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type,
                                typename execution_space::size_type>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Static>>::value));
      ASSERT_TRUE((std::is_void<work_tag>::value));
    }

    {
      using policy_t =
          Kokkos::TeamPolicy<ExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((std::is_same<execution_space, ExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type,
                                typename execution_space::size_type>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_void<work_tag>::value));
    }

    {
      using policy_t =
          Kokkos::TeamPolicy<ExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>,
                             Kokkos::IndexType<long>>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((std::is_same<execution_space, ExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_void<work_tag>::value));
    }

    {
      using policy_t =
          Kokkos::TeamPolicy<Kokkos::IndexType<long>, ExecutionSpace,
                             Kokkos::Schedule<Kokkos::Dynamic>>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((std::is_same<execution_space, ExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_void<work_tag>::value));
    }

    {
      using policy_t =
          Kokkos::TeamPolicy<ExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>,
                             Kokkos::IndexType<long>, SomeTag>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((std::is_same<execution_space, ExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_same<work_tag, SomeTag>::value));
    }

    {
      using policy_t =
          Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>, ExecutionSpace,
                             Kokkos::IndexType<long>, SomeTag>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((std::is_same<execution_space, ExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_same<work_tag, SomeTag>::value));
    }

    {
      using policy_t =
          Kokkos::TeamPolicy<SomeTag, Kokkos::Schedule<Kokkos::Dynamic>,
                             Kokkos::IndexType<long>, ExecutionSpace>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((std::is_same<execution_space, ExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_same<work_tag, SomeTag>::value));
    }

    {
      using policy_t = Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((
          std::is_same<execution_space, Kokkos::DefaultExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type,
                                typename execution_space::size_type>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_void<work_tag>::value));
    }

    {
      using policy_t = Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>,
                                          Kokkos::IndexType<long>>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((
          std::is_same<execution_space, Kokkos::DefaultExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_void<work_tag>::value));
    }

    {
      using policy_t        = Kokkos::TeamPolicy<Kokkos::IndexType<long>,
                                          Kokkos::Schedule<Kokkos::Dynamic>>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((
          std::is_same<execution_space, Kokkos::DefaultExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_void<work_tag>::value));
    }

    {
      using policy_t = Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>,
                                          Kokkos::IndexType<long>, SomeTag>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((
          std::is_same<execution_space, Kokkos::DefaultExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_same<work_tag, SomeTag>::value));
    }

    {
      using policy_t = Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>,
                                          Kokkos::IndexType<long>, SomeTag>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((
          std::is_same<execution_space, Kokkos::DefaultExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_same<work_tag, SomeTag>::value));
    }

    {
      using policy_t =
          Kokkos::TeamPolicy<SomeTag, Kokkos::Schedule<Kokkos::Dynamic>,
                             Kokkos::IndexType<long>>;
      using execution_space = typename policy_t::execution_space;
      using index_type      = typename policy_t::index_type;
      using schedule_type   = typename policy_t::schedule_type;
      using work_tag        = typename policy_t::work_tag;

      ASSERT_TRUE((
          std::is_same<execution_space, Kokkos::DefaultExecutionSpace>::value));
      ASSERT_TRUE((std::is_same<index_type, long>::value));
      ASSERT_TRUE((std::is_same<schedule_type,
                                Kokkos::Schedule<Kokkos::Dynamic>>::value));
      ASSERT_TRUE((std::is_same<work_tag, SomeTag>::value));
    }
  }

  void test_compile_time_deduction_guides() {
    Kokkos::DefaultExecutionSpace des{};
    ImplicitlyConvertibleToDefaultExecutionSpace notEs{};
    SomeExecutionSpace ses{};
    ExecutionSpace es{};

    using PolicyExecSpace = std::conditional_t<
        std::is_same_v<ExecutionSpace, Kokkos::DefaultExecutionSpace>,
        Kokkos::TeamPolicy<>, Kokkos::TeamPolicy<ExecutionSpace>>;

    int i{1};

    // Copy/Move

    Kokkos::TeamPolicy<SomeExecutionSpace> ptc;
    Kokkos::TeamPolicy pdc(ptc);
    ASSERT_TRUE((std::is_same_v<decltype(pdc), decltype(ptc)>));

    Kokkos::TeamPolicy<SomeExecutionSpace> ptm;
    Kokkos::TeamPolicy pdm(std::move(ptm));
    ASSERT_TRUE((std::is_same_v<decltype(pdm), decltype(ptm)>));

    // Execution space not provided deduces to TeamPolicy<>

    Kokkos::TeamPolicy<> pt0;
    Kokkos::TeamPolicy pd0;
    ASSERT_TRUE((std::is_same_v<decltype(pd0), decltype(pt0)>));

    Kokkos::TeamPolicy<> pt1(i, i);
    Kokkos::TeamPolicy pd1(i, i);
    ASSERT_TRUE((std::is_same_v<decltype(pd1), decltype(pt1)>));

    Kokkos::TeamPolicy<> pt2(i, i, i);
    Kokkos::TeamPolicy pd2(i, i, i);
    ASSERT_TRUE((std::is_same_v<decltype(pd2), decltype(pt2)>));

    Kokkos::TeamPolicy<> pt3(i, Kokkos::AUTO);
    Kokkos::TeamPolicy pd3(i, Kokkos::AUTO);
    ASSERT_TRUE((std::is_same_v<decltype(pd3), decltype(pt3)>));

    Kokkos::TeamPolicy<> pt4(i, Kokkos::AUTO, i);
    Kokkos::TeamPolicy pd4(i, Kokkos::AUTO, i);
    ASSERT_TRUE((std::is_same_v<decltype(pd4), decltype(pt4)>));

    Kokkos::TeamPolicy<> pt5(i, Kokkos::AUTO, Kokkos::AUTO);
    Kokkos::TeamPolicy pd5(i, Kokkos::AUTO, Kokkos::AUTO);
    ASSERT_TRUE((std::is_same_v<decltype(pt5), decltype(pd5)>));

    Kokkos::TeamPolicy<> pt6(i, i, Kokkos::AUTO);
    Kokkos::TeamPolicy pd6(i, i, Kokkos::AUTO);
    ASSERT_TRUE((std::is_same_v<decltype(pd6), decltype(pt6)>));

    // DefaultExecutionSpace deduces to TeamPolicy<>

    Kokkos::TeamPolicy<> pt7(des, i, i);
    Kokkos::TeamPolicy pd7(des, i, i);
    ASSERT_TRUE((std::is_same_v<decltype(pt7), decltype(pd7)>));

    Kokkos::TeamPolicy<> pt8(des, i, i, i);
    Kokkos::TeamPolicy pd8(des, i, i, i);
    ASSERT_TRUE((std::is_same_v<decltype(pt8), decltype(pd8)>));

    Kokkos::TeamPolicy<> pt9(des, i, Kokkos::AUTO);
    Kokkos::TeamPolicy pd9(des, i, Kokkos::AUTO);
    ASSERT_TRUE((std::is_same_v<decltype(pd9), decltype(pt9)>));

    Kokkos::TeamPolicy<> pt10(des, i, Kokkos::AUTO, i);
    Kokkos::TeamPolicy pd10(des, i, Kokkos::AUTO, i);
    ASSERT_TRUE((std::is_same_v<decltype(pd10), decltype(pt10)>));

    Kokkos::TeamPolicy<> pt11(des, i, Kokkos::AUTO, Kokkos::AUTO);
    Kokkos::TeamPolicy pd11(des, i, Kokkos::AUTO, Kokkos::AUTO);
    ASSERT_TRUE((std::is_same_v<decltype(pt11), decltype(pd11)>));

    Kokkos::TeamPolicy<> pt12(des, i, i, Kokkos::AUTO);
    Kokkos::TeamPolicy pd12(des, i, i, Kokkos::AUTO);
    ASSERT_TRUE((std::is_same_v<decltype(pd12), decltype(pt12)>));

    // Convertible to DefaultExecutionSpace deduces to TeamPolicy<>

    Kokkos::TeamPolicy<> pt13(notEs, i, i);
    Kokkos::TeamPolicy pd13(notEs, i, i);
    ASSERT_TRUE((std::is_same_v<decltype(pd13), decltype(pt13)>));

    Kokkos::TeamPolicy<> pt14(notEs, i, i, i);
    Kokkos::TeamPolicy pd14(notEs, i, i, i);
    ASSERT_TRUE((std::is_same_v<decltype(pd14), decltype(pt14)>));

    Kokkos::TeamPolicy<> pt15(notEs, i, Kokkos::AUTO);
    Kokkos::TeamPolicy pd15(notEs, i, Kokkos::AUTO);
    ASSERT_TRUE((std::is_same_v<decltype(pd15), decltype(pt15)>));

    Kokkos::TeamPolicy<> pt16(notEs, i, Kokkos::AUTO, i);
    Kokkos::TeamPolicy pd16(notEs, i, Kokkos::AUTO, i);
    ASSERT_TRUE((std::is_same_v<decltype(pd16), decltype(pt16)>));

    Kokkos::TeamPolicy<> pt17(notEs, i, Kokkos::AUTO, Kokkos::AUTO);
    Kokkos::TeamPolicy pd17(notEs, i, Kokkos::AUTO, Kokkos::AUTO);
    ASSERT_TRUE((std::is_same_v<decltype(pd17), decltype(pt17)>));

    Kokkos::TeamPolicy<> pt18(notEs, i, i, Kokkos::AUTO);
    Kokkos::TeamPolicy pd18(notEs, i, i, Kokkos::AUTO);
    ASSERT_TRUE((std::is_same_v<decltype(pd18), decltype(pt18)>));

    // ES != DefaultExecutionSpacea deduces to TeamPolicy<ES>

    Kokkos::TeamPolicy<SomeExecutionSpace> pt19(ses, i, i);
    Kokkos::TeamPolicy pd19(ses, i, i);
    ASSERT_TRUE((std::is_same_v<decltype(pd19), decltype(pt19)>));

    Kokkos::TeamPolicy<SomeExecutionSpace> pt20(ses, i, i, i);
    Kokkos::TeamPolicy pd20(ses, i, i, i);
    ASSERT_TRUE((std::is_same_v<decltype(pd20), decltype(pt20)>));

    Kokkos::TeamPolicy<SomeExecutionSpace> pt21(ses, i, Kokkos::AUTO);
    Kokkos::TeamPolicy pd21(ses, i, Kokkos::AUTO);
    ASSERT_TRUE((std::is_same_v<decltype(pd21), decltype(pt21)>));

    Kokkos::TeamPolicy<SomeExecutionSpace> pt22(ses, i, Kokkos::AUTO, i);
    Kokkos::TeamPolicy pd22(ses, i, Kokkos::AUTO, i);
    ASSERT_TRUE((std::is_same_v<decltype(pd22), decltype(pt22)>));

    Kokkos::TeamPolicy<SomeExecutionSpace> pt23(ses, i, Kokkos::AUTO,
                                                Kokkos::AUTO);
    Kokkos::TeamPolicy pd23(ses, i, Kokkos::AUTO, Kokkos::AUTO);
    ASSERT_TRUE((std::is_same_v<decltype(pd23), decltype(pt23)>));

    Kokkos::TeamPolicy<SomeExecutionSpace> pt24(ses, i, i, Kokkos::AUTO);
    Kokkos::TeamPolicy pd24(ses, i, i, Kokkos::AUTO);
    ASSERT_TRUE((std::is_same_v<decltype(pd24), decltype(pt24)>));

    // Confirm it works for the ExecutionSpace template parameter passed in

    PolicyExecSpace pt25(es, i, i);
    Kokkos::TeamPolicy pd25(es, i, i);
    ASSERT_TRUE((std::is_same_v<decltype(pd25), decltype(pt25)>));

    PolicyExecSpace pt26(es, i, i, i);
    Kokkos::TeamPolicy pd26(es, i, i, i);
    ASSERT_TRUE((std::is_same_v<decltype(pd26), decltype(pt26)>));

    PolicyExecSpace pt27(es, i, Kokkos::AUTO);
    Kokkos::TeamPolicy pd27(es, i, Kokkos::AUTO);
    ASSERT_TRUE((std::is_same_v<decltype(pd27), decltype(pt27)>));

    PolicyExecSpace pt28(es, i, Kokkos::AUTO, i);
    Kokkos::TeamPolicy pd28(es, i, Kokkos::AUTO, i);
    ASSERT_TRUE((std::is_same_v<decltype(pd28), decltype(pt28)>));

    PolicyExecSpace pt29(es, i, Kokkos::AUTO, Kokkos::AUTO);
    Kokkos::TeamPolicy pd29(es, i, Kokkos::AUTO, Kokkos::AUTO);
    ASSERT_TRUE((std::is_same_v<decltype(pd29), decltype(pt29)>));

    PolicyExecSpace pt30(es, i, i, Kokkos::AUTO);
    Kokkos::TeamPolicy pd30(es, i, i, Kokkos::AUTO);
    ASSERT_TRUE((std::is_same_v<decltype(pd30), decltype(pt30)>));
  }

  template <class policy_t>
  void test_run_time_parameters_type() {
    int league_size = 131;
    int team_size   = 4 < policy_t::execution_space::concurrency()
                        ? 4
                        : policy_t::execution_space::concurrency();
#ifdef KOKKOS_ENABLE_HPX
    team_size = 1;
#endif
#ifdef KOKKOS_ENABLE_OPENMPTARGET
    if (std::is_same<typename policy_t::execution_space,
                     Kokkos::Experimental::OpenMPTarget>::value)
      team_size = 32;
#endif
    int chunk_size         = 4;
    int per_team_scratch   = 1024;
    int per_thread_scratch = 16;
    int scratch_size       = per_team_scratch + per_thread_scratch * team_size;

    policy_t p1(league_size, team_size);
    ASSERT_EQ(p1.league_size(), league_size);
    ASSERT_EQ(p1.team_size(), team_size);
    ASSERT_GT(p1.chunk_size(), 0);
    ASSERT_EQ(size_t(p1.scratch_size(0)), 0u);

    policy_t p2 = p1.set_chunk_size(chunk_size);
    ASSERT_EQ(p1.league_size(), league_size);
    ASSERT_EQ(p1.team_size(), team_size);
    ASSERT_EQ(p1.chunk_size(), chunk_size);
    ASSERT_EQ(size_t(p1.scratch_size(0)), 0u);

    ASSERT_EQ(p2.league_size(), league_size);
    ASSERT_EQ(p2.team_size(), team_size);
    ASSERT_EQ(p2.chunk_size(), chunk_size);
    ASSERT_EQ(size_t(p2.scratch_size(0)), 0u);

    policy_t p3 = p2.set_scratch_size(0, Kokkos::PerTeam(per_team_scratch));
    ASSERT_EQ(p2.league_size(), league_size);
    ASSERT_EQ(p2.team_size(), team_size);
    ASSERT_EQ(p2.chunk_size(), chunk_size);
    ASSERT_EQ(size_t(p2.scratch_size(0)), size_t(per_team_scratch));
    ASSERT_EQ(p3.league_size(), league_size);
    ASSERT_EQ(p3.team_size(), team_size);
    ASSERT_EQ(p3.chunk_size(), chunk_size);
    ASSERT_EQ(size_t(p3.scratch_size(0)), size_t(per_team_scratch));

    policy_t p4 = p2.set_scratch_size(0, Kokkos::PerThread(per_thread_scratch));
    ASSERT_EQ(p2.league_size(), league_size);
    ASSERT_EQ(p2.team_size(), team_size);
    ASSERT_EQ(p2.chunk_size(), chunk_size);
    ASSERT_EQ(size_t(p2.scratch_size(0)), size_t(scratch_size));
    ASSERT_EQ(p4.league_size(), league_size);
    ASSERT_EQ(p4.team_size(), team_size);
    ASSERT_EQ(p4.chunk_size(), chunk_size);
    ASSERT_EQ(size_t(p4.scratch_size(0)), size_t(scratch_size));

    policy_t p5 = p2.set_scratch_size(0, Kokkos::PerThread(per_thread_scratch),
                                      Kokkos::PerTeam(per_team_scratch));
    ASSERT_EQ(p2.league_size(), league_size);
    ASSERT_EQ(p2.team_size(), team_size);
    ASSERT_EQ(p2.chunk_size(), chunk_size);
    ASSERT_EQ(size_t(p2.scratch_size(0)), size_t(scratch_size));
    ASSERT_EQ(p5.league_size(), league_size);
    ASSERT_EQ(p5.team_size(), team_size);
    ASSERT_EQ(p5.chunk_size(), chunk_size);
    ASSERT_EQ(size_t(p5.scratch_size(0)), size_t(scratch_size));

    policy_t p6 = p2.set_scratch_size(0, Kokkos::PerTeam(per_team_scratch),
                                      Kokkos::PerThread(per_thread_scratch));
    ASSERT_EQ(p2.league_size(), league_size);
    ASSERT_EQ(p2.team_size(), team_size);
    ASSERT_EQ(p2.chunk_size(), chunk_size);
    ASSERT_EQ(size_t(p2.scratch_size(0)), size_t(scratch_size));
    ASSERT_EQ(p6.league_size(), league_size);
    ASSERT_EQ(p6.team_size(), team_size);
    ASSERT_EQ(p6.chunk_size(), chunk_size);
    ASSERT_EQ(size_t(p6.scratch_size(0)), size_t(scratch_size));

    policy_t p7 = p3.set_scratch_size(0, Kokkos::PerTeam(per_team_scratch),
                                      Kokkos::PerThread(per_thread_scratch));
    ASSERT_EQ(p3.league_size(), league_size);
    ASSERT_EQ(p3.team_size(), team_size);
    ASSERT_EQ(p3.chunk_size(), chunk_size);
    ASSERT_EQ(size_t(p3.scratch_size(0)), size_t(scratch_size));
    ASSERT_EQ(p7.league_size(), league_size);
    ASSERT_EQ(p7.team_size(), team_size);
    ASSERT_EQ(p7.chunk_size(), chunk_size);
    ASSERT_EQ(size_t(p7.scratch_size(0)), size_t(scratch_size));

    policy_t p8;  // default constructed
    ASSERT_EQ(p8.league_size(), 0);
    ASSERT_EQ(size_t(p8.scratch_size(0)), 0u);
    p8 = p3;  // call assignment operator
    ASSERT_EQ(p3.league_size(), league_size);
    ASSERT_EQ(p3.team_size(), team_size);
    ASSERT_EQ(p3.chunk_size(), chunk_size);
    ASSERT_EQ(size_t(p3.scratch_size(0)), size_t(scratch_size));
    ASSERT_EQ(p8.league_size(), league_size);
    ASSERT_EQ(p8.team_size(), team_size);
    ASSERT_EQ(p8.chunk_size(), chunk_size);
    ASSERT_EQ(size_t(p8.scratch_size(0)), size_t(scratch_size));
  }

  void test_run_time_parameters() {
    test_run_time_parameters_type<Kokkos::TeamPolicy<ExecutionSpace>>();
    test_run_time_parameters_type<
        Kokkos::TeamPolicy<ExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>,
                           Kokkos::IndexType<long>>>();
    test_run_time_parameters_type<
        Kokkos::TeamPolicy<Kokkos::IndexType<long>, ExecutionSpace,
                           Kokkos::Schedule<Kokkos::Dynamic>>>();
    test_run_time_parameters_type<
        Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>,
                           Kokkos::IndexType<long>, ExecutionSpace, SomeTag>>();
  }
};

// semiregular is copyable and default initializable
// (regular requires equality comparable)
template <class Policy>
void check_semiregular() {
  static_assert(std::is_default_constructible<Policy>::value, "");
  static_assert(std::is_copy_constructible<Policy>::value, "");
  static_assert(std::is_move_constructible<Policy>::value, "");
  static_assert(std::is_copy_assignable<Policy>::value, "");
  static_assert(std::is_move_assignable<Policy>::value, "");
  static_assert(std::is_destructible<Policy>::value, "");
}

TEST(TEST_CATEGORY, policy_construction) {
  check_semiregular<Kokkos::RangePolicy<TEST_EXECSPACE>>();
  check_semiregular<Kokkos::TeamPolicy<TEST_EXECSPACE>>();
  check_semiregular<Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>>();

  TestRangePolicyConstruction<TEST_EXECSPACE>();
  TestTeamPolicyConstruction<TEST_EXECSPACE>();
}

template <template <class...> class Policy, class... Args>
void check_converting_constructor_add_work_tag(Policy<Args...> const& policy) {
  // Not the greatest but at least checking it compiles
  struct WorkTag {};
  Policy<Args..., WorkTag> policy_with_tag = policy;
  (void)policy_with_tag;
}

TEST(TEST_CATEGORY, policy_converting_constructor_from_other_policy) {
  check_converting_constructor_add_work_tag(
      Kokkos::RangePolicy<TEST_EXECSPACE>{});
  check_converting_constructor_add_work_tag(
      Kokkos::TeamPolicy<TEST_EXECSPACE>{});
  check_converting_constructor_add_work_tag(
      Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>{});
}

#ifndef KOKKOS_COMPILER_NVHPC       // FIXME_NVHPC
#ifndef KOKKOS_ENABLE_OPENMPTARGET  // FIXME_OPENMPTARGET
TEST(TEST_CATEGORY_DEATH, policy_bounds_unsafe_narrowing_conversions) {
  using Policy = Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>,
                                       Kokkos::IndexType<unsigned>>;

  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_DEATH(
      {
        (void)Policy({-1, 0}, {2, 3});
      },
      "unsafe narrowing conversion");
}
#endif
#endif

template <class Policy>
void test_prefer_desired_occupancy(Policy const& policy) {
  static_assert(!Policy::experimental_contains_desired_occupancy, "");

  // MaximizeOccupancy -> MaximizeOccupancy
  auto const policy_still_no_occ = Kokkos::Experimental::prefer(
      policy, Kokkos::Experimental::MaximizeOccupancy{});
  static_assert(
      !decltype(policy_still_no_occ)::experimental_contains_desired_occupancy,
      "");

  // MaximizeOccupancy -> DesiredOccupancy
  auto const policy_with_occ = Kokkos::Experimental::prefer(
      policy, Kokkos::Experimental::DesiredOccupancy{33});
  static_assert(
      decltype(policy_with_occ)::experimental_contains_desired_occupancy, "");
  EXPECT_EQ(policy_with_occ.impl_get_desired_occupancy().value(), 33);

  // DesiredOccupancy -> DesiredOccupancy
  auto const policy_change_occ = Kokkos::Experimental::prefer(
      policy_with_occ, Kokkos::Experimental::DesiredOccupancy{24});
  static_assert(
      decltype(policy_change_occ)::experimental_contains_desired_occupancy, "");
  EXPECT_EQ(policy_change_occ.impl_get_desired_occupancy().value(), 24);

  // DesiredOccupancy -> MaximizeOccupancy
  auto const policy_drop_occ = Kokkos::Experimental::prefer(
      policy_with_occ, Kokkos::Experimental::MaximizeOccupancy{});
  static_assert(
      !decltype(policy_drop_occ)::experimental_contains_desired_occupancy, "");
}

template <class... Args>
struct DummyPolicy : Kokkos::Impl::PolicyTraits<Args...> {
  using execution_policy = DummyPolicy;

  using base_t = Kokkos::Impl::PolicyTraits<Args...>;
  using base_t::base_t;
};

TEST(TEST_CATEGORY, desired_occupancy_prefer) {
  test_prefer_desired_occupancy(DummyPolicy<TEST_EXECSPACE>{});
  test_prefer_desired_occupancy(Kokkos::RangePolicy<TEST_EXECSPACE>{});
  test_prefer_desired_occupancy(
      Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>{});
  test_prefer_desired_occupancy(Kokkos::TeamPolicy<TEST_EXECSPACE>{});
}

// For a more informative static assertion:
template <size_t>
struct static_assert_dummy_policy_must_be_size_one;
template <>
struct static_assert_dummy_policy_must_be_size_one<1> {};
template <size_t, size_t>
struct static_assert_dummy_policy_must_be_size_of_desired_occupancy;
template <>
struct static_assert_dummy_policy_must_be_size_of_desired_occupancy<
    sizeof(Kokkos::Experimental::DesiredOccupancy),
    sizeof(Kokkos::Experimental::DesiredOccupancy)> {};

// EBO failure with VS 16.11.3 and CUDA 11.4.2
#if !(defined(_WIN32) && defined(KOKKOS_ENABLE_CUDA))
TEST(TEST_CATEGORY, desired_occupancy_empty_base_optimization) {
  DummyPolicy<TEST_EXECSPACE> const policy{};
  static_assert(sizeof(decltype(policy)) == 1, "");
  static_assert_dummy_policy_must_be_size_one<sizeof(decltype(policy))>
      _assert1{};
  (void)&_assert1;  // avoid unused variable warning

  using Kokkos::Experimental::DesiredOccupancy;
  auto policy_with_occ =
      Kokkos::Experimental::prefer(policy, DesiredOccupancy{50});
  static_assert(sizeof(decltype(policy_with_occ)) == sizeof(DesiredOccupancy),
                "");
  static_assert_dummy_policy_must_be_size_of_desired_occupancy<
      sizeof(decltype(policy_with_occ)), sizeof(DesiredOccupancy)>
      _assert2{};
  (void)&_assert2;  // avoid unused variable warning
}
#endif

template <typename Policy>
void test_desired_occupancy_converting_constructors(Policy const& policy) {
  auto policy_with_occ = Kokkos::Experimental::prefer(
      policy, Kokkos::Experimental::DesiredOccupancy{50});
  EXPECT_EQ(policy_with_occ.impl_get_desired_occupancy().value(), 50);

  auto policy_with_hint = Kokkos::Experimental::require(
      policy_with_occ, Kokkos::Experimental::WorkItemProperty::HintLightWeight);
  EXPECT_EQ(policy_with_hint.impl_get_desired_occupancy().value(), 50);
}

TEST(TEST_CATEGORY, desired_occupancy_converting_constructors) {
  test_desired_occupancy_converting_constructors(
      Kokkos::RangePolicy<TEST_EXECSPACE>{});
  test_desired_occupancy_converting_constructors(
      Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>{});
  test_desired_occupancy_converting_constructors(
      Kokkos::TeamPolicy<TEST_EXECSPACE>{});
}

template <class T>
void more_md_range_policy_construction_test() {
  (void)Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>{
      Kokkos::Array<T, 2>{}, Kokkos::Array<T, 2>{}};

  (void)Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>{{{T(0), T(0)}},
                                                               {{T(2), T(2)}}};

  (void)Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>{{T(0), T(0)},
                                                               {T(2), T(2)}};
}

TEST(TEST_CATEGORY, md_range_policy_compile_time_deduction_guides) {
  Kokkos::DefaultExecutionSpace des{};
  ImplicitlyConvertibleToDefaultExecutionSpace notEs{};
  SomeExecutionSpace ses{};
  TEST_EXECSPACE es{};

  constexpr int t[5]       = {};
  constexpr int64_t tt[5]  = {};
  Kokkos::Array<int, 3> a  = {};
  Kokkos::Array<int, 2> aa = {};

  Kokkos::MDRangePolicy<Kokkos::Rank<2>> ptc;
  Kokkos::MDRangePolicy pdc(ptc);
  ASSERT_TRUE((std::is_same_v<decltype(pdc), decltype(ptc)>));

  Kokkos::MDRangePolicy<Kokkos::Rank<2>> ptm;
  Kokkos::MDRangePolicy pdm(std::move(ptm));
  ASSERT_TRUE((std::is_same_v<decltype(pdm), decltype(ptm)>));

  Kokkos::MDRangePolicy<Kokkos::Rank<std::size(t)>> pt1(t, t);
  Kokkos::MDRangePolicy pd1(t, t);
  ASSERT_TRUE((std::is_same_v<decltype(pd1), decltype(pt1)>));

  Kokkos::MDRangePolicy<Kokkos::Rank<std::size(t)>> pt2(t, t, tt);
  Kokkos::MDRangePolicy pd2(t, t, tt);
  ASSERT_TRUE((std::is_same_v<decltype(pd2), decltype(pt2)>));

  Kokkos::MDRangePolicy<Kokkos::Rank<std::size(t)>> pt3(des, t, t);
  Kokkos::MDRangePolicy pd3(des, t, t);
  ASSERT_TRUE((std::is_same_v<decltype(pd3), decltype(pt3)>));

  Kokkos::MDRangePolicy<Kokkos::Rank<std::size(t)>> pt4(notEs, t, t);
  Kokkos::MDRangePolicy pd4(notEs, t, t);
  ASSERT_TRUE((std::is_same_v<decltype(pd4), decltype(pt4)>));

  Kokkos::MDRangePolicy<SomeExecutionSpace, Kokkos::Rank<std::size(t)>> pt5(
      ses, t, t);
  Kokkos::MDRangePolicy pd5(ses, t, t);
  ASSERT_TRUE((std::is_same_v<decltype(pd5), decltype(pt5)>));

  using TestExecSpaceMDRangePolicyT = std::conditional_t<
      std::is_same_v<TEST_EXECSPACE, Kokkos::DefaultExecutionSpace>,
      Kokkos::MDRangePolicy<Kokkos::Rank<std::size(t)>>,
      Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<std::size(t)>>>;

  TestExecSpaceMDRangePolicyT pt6(es, t, t);
  Kokkos::MDRangePolicy pd6(es, t, t);
  ASSERT_TRUE((std::is_same_v<decltype(pd6), decltype(pt6)>));

  Kokkos::MDRangePolicy<Kokkos::Rank<std::size(a)>> pt7(a, a);
  Kokkos::MDRangePolicy pd7(a, a);
  ASSERT_TRUE((std::is_same_v<decltype(pd7), decltype(pt7)>));

  Kokkos::MDRangePolicy<Kokkos::Rank<std::size(a)>> pt8(a, a, aa);
  Kokkos::MDRangePolicy pd8(a, a, aa);
  ASSERT_TRUE((std::is_same_v<decltype(pd8), decltype(pt8)>));

  Kokkos::MDRangePolicy<Kokkos::Rank<std::size(a)>> pt9(des, a, a);
  Kokkos::MDRangePolicy pd9(des, a, a);
  ASSERT_TRUE((std::is_same_v<decltype(pd9), decltype(pt9)>));

  Kokkos::MDRangePolicy<Kokkos::Rank<std::size(a)>> pt10(notEs, a, a);
  Kokkos::MDRangePolicy pd10(notEs, a, a);
  ASSERT_TRUE((std::is_same_v<decltype(pd10), decltype(pt10)>));

  Kokkos::MDRangePolicy<SomeExecutionSpace, Kokkos::Rank<std::size(a)>> pt11(
      ses, a, a);
  Kokkos::MDRangePolicy pd11(ses, a, a);
  ASSERT_TRUE((std::is_same_v<decltype(pd9), decltype(pt9)>));

  using TestExecSpaceMDRangePolicyA = std::conditional_t<
      std::is_same_v<TEST_EXECSPACE, Kokkos::DefaultExecutionSpace>,
      Kokkos::MDRangePolicy<Kokkos::Rank<std::size(a)>>,
      Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<std::size(t)>>>;

  TestExecSpaceMDRangePolicyA pt12(es, a, a);
  Kokkos::MDRangePolicy pd12(es, a, a);
  ASSERT_TRUE((std::is_same_v<decltype(pd12), decltype(pt12)>));

  Kokkos::MDRangePolicy<Kokkos::Rank<std::size(a)>> pt13(des, a, a, aa);
  Kokkos::MDRangePolicy pd13(des, a, a, aa);
  ASSERT_TRUE((std::is_same_v<decltype(pd13), decltype(pt13)>));

  Kokkos::MDRangePolicy<Kokkos::Rank<std::size(a)>> pt14(notEs, a, a, aa);
  Kokkos::MDRangePolicy pd14(notEs, a, a, aa);
  ASSERT_TRUE((std::is_same_v<decltype(pd14), decltype(pt14)>));

  Kokkos::MDRangePolicy<SomeExecutionSpace, Kokkos::Rank<std::size(a)>> pt15(
      ses, a, a, aa);
  Kokkos::MDRangePolicy pd15(ses, a, a, aa);
  ASSERT_TRUE((std::is_same_v<decltype(pd15), decltype(pt15)>));

  TestExecSpaceMDRangePolicyA pt16(es, a, a, aa);
  Kokkos::MDRangePolicy pd16(es, a, a, aa);
  ASSERT_TRUE((std::is_same_v<decltype(pd16), decltype(pt16)>));
}

TEST(TEST_CATEGORY, md_range_policy_construction_from_arrays) {
  {
    // Check that construction from Kokkos::Array of long compiles for backwards
    // compability.  This was broken in
    // https://github.com/kokkos/kokkos/pull/3527/commits/88ea8eec6567c84739d77bdd25fdbc647fae28bb#r512323639
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>> p1(
        Kokkos::Array<long, 2>{{0, 1}}, Kokkos::Array<long, 2>{{2, 3}});
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>> p2(
        Kokkos::Array<long, 2>{{0, 1}}, Kokkos::Array<long, 2>{{2, 3}});
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>> p3(
        Kokkos::Array<long, 2>{{0, 1}}, Kokkos::Array<long, 2>{{2, 3}},
        Kokkos::Array<long, 1>{{4}});
  }
  {
    // Check that construction from Kokkos::Array of the specified index type
    // works.
    using index_type = unsigned long long;
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>,
                          Kokkos::IndexType<index_type>>
        p1(Kokkos::Array<index_type, 2>{{0, 1}},
           Kokkos::Array<index_type, 2>{{2, 3}});
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>,
                          Kokkos::IndexType<index_type>>
        p2(Kokkos::Array<index_type, 2>{{0, 1}},
           Kokkos::Array<index_type, 2>{{2, 3}});
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>,
                          Kokkos::IndexType<index_type>>
        p3(Kokkos::Array<index_type, 2>{{0, 1}},
           Kokkos::Array<index_type, 2>{{2, 3}},
           Kokkos::Array<index_type, 1>{{4}});
  }
  {
    // Check that construction from double-braced initliazer list
    // works.
    using index_type = unsigned long long;
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>> p1({{0, 1}},
                                                              {{2, 3}});
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>,
                          Kokkos::IndexType<index_type>>
        p2({{0, 1}}, {{2, 3}});
  }

  more_md_range_policy_construction_test<char>();
  more_md_range_policy_construction_test<int>();
  more_md_range_policy_construction_test<unsigned long>();
  more_md_range_policy_construction_test<std::int64_t>();
}

template <class WorkTag, class Policy>
constexpr auto set_worktag(Policy const& policy) {
  static_assert(Kokkos::is_execution_policy<Policy>::value, "");
  using PolicyWithWorkTag =
      Kokkos::Impl::WorkTagTrait::policy_with_trait<Policy, WorkTag>;
  return PolicyWithWorkTag{policy};
}

TEST(TEST_CATEGORY, policy_set_worktag) {
  struct SomeWorkTag {};
  struct OtherWorkTag {};

  Kokkos::RangePolicy<> p1;
  static_assert(std::is_void<decltype(p1)::work_tag>::value, "");

  auto p2 = set_worktag<SomeWorkTag>(p1);
  static_assert(std::is_same<decltype(p2)::work_tag, SomeWorkTag>::value, "");

  auto p3 = set_worktag<OtherWorkTag>(p2);
  static_assert(std::is_same<decltype(p3)::work_tag, OtherWorkTag>::value, "");

  // NOTE this does not currently compile
  // auto p4 = set_worktag<void>(p3);
  // static_assert(std::is_void<decltype(p4)::work_tag>::value, "");
}
}  // namespace Test
