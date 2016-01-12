/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// Questions? Contact  H. Carter Edwards (hcedwar@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <sstream>
#include <iostream>

struct SomeTag{};

template< class ExecutionSpace >
class TestRangePolicy {
public:
  TestRangePolicy() {
    test_compile_time_parameters();
  }
private:
  void test_compile_time_parameters() {
    {
      typedef Kokkos::RangePolicy<> policy_t;
      typedef typename policy_t::execution_space execution_space;
      typedef typename policy_t::index_type      index_type;
      typedef typename policy_t::schedule_type   schedule_type;
      typedef typename policy_t::work_tag        work_tag;

      ASSERT_TRUE((std::is_same<execution_space ,Kokkos::DefaultExecutionSpace       >::value));
      ASSERT_TRUE((std::is_same<index_type      ,typename execution_space::size_type >::value));
      ASSERT_TRUE((std::is_same<schedule_type   ,Kokkos::Schedule<Kokkos::Static>    >::value));
      ASSERT_TRUE((std::is_same<work_tag        ,void                                >::value));
    }
    {
      typedef Kokkos::RangePolicy<ExecutionSpace> policy_t;
      typedef typename policy_t::execution_space execution_space;
      typedef typename policy_t::index_type      index_type;
      typedef typename policy_t::schedule_type   schedule_type;
      typedef typename policy_t::work_tag        work_tag;

      ASSERT_TRUE((std::is_same<execution_space ,ExecutionSpace                      >::value));
      ASSERT_TRUE((std::is_same<index_type      ,typename execution_space::size_type >::value));
      ASSERT_TRUE((std::is_same<schedule_type   ,Kokkos::Schedule<Kokkos::Static>    >::value));
      ASSERT_TRUE((std::is_same<work_tag        ,void                                >::value));
    }
    {
      typedef Kokkos::RangePolicy<ExecutionSpace,Kokkos::Schedule<Kokkos::Dynamic> > policy_t;
      typedef typename policy_t::execution_space execution_space;
      typedef typename policy_t::index_type      index_type;
      typedef typename policy_t::schedule_type   schedule_type;
      typedef typename policy_t::work_tag        work_tag;

      ASSERT_TRUE((std::is_same<execution_space ,ExecutionSpace                      >::value));
      ASSERT_TRUE((std::is_same<index_type      ,typename execution_space::size_type >::value));
      ASSERT_TRUE((std::is_same<schedule_type   ,Kokkos::Schedule<Kokkos::Dynamic>   >::value));
      ASSERT_TRUE((std::is_same<work_tag        ,void                                >::value));
    }
    {
      typedef Kokkos::RangePolicy<ExecutionSpace,Kokkos::Schedule<Kokkos::Dynamic>,Kokkos::IndexType<long> > policy_t;
      typedef typename policy_t::execution_space execution_space;
      typedef typename policy_t::index_type      index_type;
      typedef typename policy_t::schedule_type   schedule_type;
      typedef typename policy_t::work_tag        work_tag;

      ASSERT_TRUE((std::is_same<execution_space ,ExecutionSpace                      >::value));
      ASSERT_TRUE((std::is_same<index_type      ,long                                >::value));
      ASSERT_TRUE((std::is_same<schedule_type   ,Kokkos::Schedule<Kokkos::Dynamic>   >::value));
      ASSERT_TRUE((std::is_same<work_tag        ,void                                >::value));
    }
    {
      typedef Kokkos::RangePolicy<Kokkos::IndexType<long>, ExecutionSpace,Kokkos::Schedule<Kokkos::Dynamic> > policy_t;
      typedef typename policy_t::execution_space execution_space;
      typedef typename policy_t::index_type      index_type;
      typedef typename policy_t::schedule_type   schedule_type;
      typedef typename policy_t::work_tag        work_tag;

      ASSERT_TRUE((std::is_same<execution_space ,ExecutionSpace                      >::value));
      ASSERT_TRUE((std::is_same<index_type      ,long                                >::value));
      ASSERT_TRUE((std::is_same<schedule_type   ,Kokkos::Schedule<Kokkos::Dynamic>   >::value));
      ASSERT_TRUE((std::is_same<work_tag        ,void                                >::value));
    }
    {
      typedef Kokkos::RangePolicy<ExecutionSpace,Kokkos::Schedule<Kokkos::Dynamic>,Kokkos::IndexType<long>,SomeTag > policy_t;
      typedef typename policy_t::execution_space execution_space;
      typedef typename policy_t::index_type      index_type;
      typedef typename policy_t::schedule_type   schedule_type;
      typedef typename policy_t::work_tag        work_tag;

      ASSERT_TRUE((std::is_same<execution_space ,ExecutionSpace                      >::value));
      ASSERT_TRUE((std::is_same<index_type      ,long                                >::value));
      ASSERT_TRUE((std::is_same<schedule_type   ,Kokkos::Schedule<Kokkos::Dynamic>   >::value));
      ASSERT_TRUE((std::is_same<work_tag        ,SomeTag                             >::value));
    }
    {
      typedef Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>,ExecutionSpace,Kokkos::IndexType<long>,SomeTag > policy_t;
      typedef typename policy_t::execution_space execution_space;
      typedef typename policy_t::index_type      index_type;
      typedef typename policy_t::schedule_type   schedule_type;
      typedef typename policy_t::work_tag        work_tag;

      ASSERT_TRUE((std::is_same<execution_space ,ExecutionSpace                      >::value));
      ASSERT_TRUE((std::is_same<index_type      ,long                                >::value));
      ASSERT_TRUE((std::is_same<schedule_type   ,Kokkos::Schedule<Kokkos::Dynamic>   >::value));
      ASSERT_TRUE((std::is_same<work_tag        ,SomeTag                             >::value));
    }
    {
      typedef Kokkos::RangePolicy<SomeTag,Kokkos::Schedule<Kokkos::Dynamic>,Kokkos::IndexType<long>,ExecutionSpace > policy_t;
      typedef typename policy_t::execution_space execution_space;
      typedef typename policy_t::index_type      index_type;
      typedef typename policy_t::schedule_type   schedule_type;
      typedef typename policy_t::work_tag        work_tag;

      ASSERT_TRUE((std::is_same<execution_space ,ExecutionSpace                      >::value));
      ASSERT_TRUE((std::is_same<index_type      ,long                                >::value));
      ASSERT_TRUE((std::is_same<schedule_type   ,Kokkos::Schedule<Kokkos::Dynamic>   >::value));
      ASSERT_TRUE((std::is_same<work_tag        ,SomeTag                             >::value));
    }
    {
      typedef Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic> > policy_t;
      typedef typename policy_t::execution_space execution_space;
      typedef typename policy_t::index_type      index_type;
      typedef typename policy_t::schedule_type   schedule_type;
      typedef typename policy_t::work_tag        work_tag;

      ASSERT_TRUE((std::is_same<execution_space ,Kokkos::DefaultExecutionSpace                      >::value));
      ASSERT_TRUE((std::is_same<index_type      ,typename execution_space::size_type >::value));
      ASSERT_TRUE((std::is_same<schedule_type   ,Kokkos::Schedule<Kokkos::Dynamic>   >::value));
      ASSERT_TRUE((std::is_same<work_tag        ,void                                >::value));
    }
    {
      typedef Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>,Kokkos::IndexType<long> > policy_t;
      typedef typename policy_t::execution_space execution_space;
      typedef typename policy_t::index_type      index_type;
      typedef typename policy_t::schedule_type   schedule_type;
      typedef typename policy_t::work_tag        work_tag;

      ASSERT_TRUE((std::is_same<execution_space ,Kokkos::DefaultExecutionSpace       >::value));
      ASSERT_TRUE((std::is_same<index_type      ,long                                >::value));
      ASSERT_TRUE((std::is_same<schedule_type   ,Kokkos::Schedule<Kokkos::Dynamic>   >::value));
      ASSERT_TRUE((std::is_same<work_tag        ,void                                >::value));
    }
    {
      typedef Kokkos::RangePolicy<Kokkos::IndexType<long>, Kokkos::Schedule<Kokkos::Dynamic> > policy_t;
      typedef typename policy_t::execution_space execution_space;
      typedef typename policy_t::index_type      index_type;
      typedef typename policy_t::schedule_type   schedule_type;
      typedef typename policy_t::work_tag        work_tag;

      ASSERT_TRUE((std::is_same<execution_space ,Kokkos::DefaultExecutionSpace       >::value));
      ASSERT_TRUE((std::is_same<index_type      ,long                                >::value));
      ASSERT_TRUE((std::is_same<schedule_type   ,Kokkos::Schedule<Kokkos::Dynamic>   >::value));
      ASSERT_TRUE((std::is_same<work_tag        ,void                                >::value));
    }
    {
      typedef Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>,Kokkos::IndexType<long>,SomeTag > policy_t;
      typedef typename policy_t::execution_space execution_space;
      typedef typename policy_t::index_type      index_type;
      typedef typename policy_t::schedule_type   schedule_type;
      typedef typename policy_t::work_tag        work_tag;

      ASSERT_TRUE((std::is_same<execution_space ,Kokkos::DefaultExecutionSpace       >::value));
      ASSERT_TRUE((std::is_same<index_type      ,long                                >::value));
      ASSERT_TRUE((std::is_same<schedule_type   ,Kokkos::Schedule<Kokkos::Dynamic>   >::value));
      ASSERT_TRUE((std::is_same<work_tag        ,SomeTag                             >::value));
    }
    {
      typedef Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>,Kokkos::IndexType<long>,SomeTag > policy_t;
      typedef typename policy_t::execution_space execution_space;
      typedef typename policy_t::index_type      index_type;
      typedef typename policy_t::schedule_type   schedule_type;
      typedef typename policy_t::work_tag        work_tag;

      ASSERT_TRUE((std::is_same<execution_space ,Kokkos::DefaultExecutionSpace       >::value));
      ASSERT_TRUE((std::is_same<index_type      ,long                                >::value));
      ASSERT_TRUE((std::is_same<schedule_type   ,Kokkos::Schedule<Kokkos::Dynamic>   >::value));
      ASSERT_TRUE((std::is_same<work_tag        ,SomeTag                             >::value));
    }
    {
      typedef Kokkos::RangePolicy<SomeTag,Kokkos::Schedule<Kokkos::Dynamic>,Kokkos::IndexType<long> > policy_t;
      typedef typename policy_t::execution_space execution_space;
      typedef typename policy_t::index_type      index_type;
      typedef typename policy_t::schedule_type   schedule_type;
      typedef typename policy_t::work_tag        work_tag;

      ASSERT_TRUE((std::is_same<execution_space ,Kokkos::DefaultExecutionSpace       >::value));
      ASSERT_TRUE((std::is_same<index_type      ,long                                >::value));
      ASSERT_TRUE((std::is_same<schedule_type   ,Kokkos::Schedule<Kokkos::Dynamic>   >::value));
      ASSERT_TRUE((std::is_same<work_tag        ,SomeTag                             >::value));
    }
  }
};
