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
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_CORE_FWD_HPP
#define KOKKOS_CORE_FWD_HPP

//----------------------------------------------------------------------------
// Kokkos_Macros.hpp does introspection on configuration options
// and compiler environment then sets a collection of #define macros.

#include <Kokkos_Macros.hpp>
#include <impl/Kokkos_Utilities.hpp>

#include <Kokkos_UniqueToken.hpp>
#include <Kokkos_MasterLock.hpp>

//----------------------------------------------------------------------------
// Have assumed a 64bit build (8byte pointers) throughout the code base.

static_assert(sizeof(void *) == 8,
              "Kokkos assumes 64-bit build; i.e., 8-byte pointers");

//----------------------------------------------------------------------------

namespace Kokkos {

struct AUTO_t {
  KOKKOS_INLINE_FUNCTION
  constexpr const AUTO_t &operator()() const { return *this; }
};

namespace {
/**\brief Token to indicate that a parameter's value is to be automatically
 * selected */
constexpr AUTO_t AUTO = Kokkos::AUTO_t();
}  // namespace

struct InvalidType {};

}  // namespace Kokkos

//----------------------------------------------------------------------------
// Forward declarations for class inter-relationships

namespace Kokkos {

class HostSpace;  ///< Memory space for main process and CPU execution spaces
class AnonymousSpace;

template <class ExecutionSpace, class MemorySpace>
struct Device;

#include <KokkosCore_Config_FwdBackend.hpp>

}  // namespace Kokkos

#include <Kokkos_Set_Default_Spaces.hpp>

namespace Kokkos {

namespace Impl {

//----------------------------------------------------------------------------
// Detect the active execution space and define its memory space.
// This is used to verify whether a running kernel can access
// a given memory space.
template <class ActiveSpace, class MemorySpace>
struct VerifyExecutionCanAccessMemorySpace {
  enum { value = 0 };
};

template <class Space>
struct VerifyExecutionCanAccessMemorySpace<Space, Space> {
  enum { value = 1 };
  KOKKOS_INLINE_FUNCTION static void verify(void) {}
  KOKKOS_INLINE_FUNCTION static void verify(const void *) {}
};
}  // namespace Impl

}  // namespace Kokkos

#define KOKKOS_RESTRICT_EXECUTION_TO_DATA(DATA_SPACE, DATA_PTR) \
  Kokkos::Impl::VerifyExecutionCanAccessMemorySpace<            \
      Kokkos::Impl::ActiveExecutionMemorySpace, DATA_SPACE>::verify(DATA_PTR)

#define KOKKOS_RESTRICT_EXECUTION_TO_(DATA_SPACE)    \
  Kokkos::Impl::VerifyExecutionCanAccessMemorySpace< \
      Kokkos::Impl::ActiveExecutionMemorySpace, DATA_SPACE>::verify()

//----------------------------------------------------------------------------

namespace Kokkos {
void fence();
}

//----------------------------------------------------------------------------

namespace Kokkos {

namespace Impl {

template <class DstSpace, class SrcSpace,
          class ExecutionSpace = typename DstSpace::execution_space>
struct DeepCopy;

template <class ViewType, class Layout, class ExecSpace, int Rank,
          typename iType>
struct ViewFillETIAvail;

template <class ViewType, class Layout = typename ViewType::array_layout,
          class ExecSpace = typename ViewType::execution_space,
          int Rank = ViewType::Rank, typename iType = int64_t,
          bool EtiAvail =
              ViewFillETIAvail<ViewType, Layout, ExecSpace, Rank, iType>::value>
struct ViewFill;

template <class ViewTypeA, class ViewTypeB, class Layout, class ExecSpace,
          int Rank, typename iType>
struct ViewCopyETIAvail;

template <class ViewTypeA, class ViewTypeB, class Layout, class ExecSpace,
          int Rank, typename iType,
          bool EtiAvail = ViewCopyETIAvail<ViewTypeA, ViewTypeB, Layout,
                                           ExecSpace, Rank, iType>::value>
struct ViewCopy;

template <class Functor, class Policy, class EnableFunctor = void,
          class EnablePolicy = void>
struct FunctorPolicyExecutionSpace;

//----------------------------------------------------------------------------
/// \class ParallelFor
/// \brief Implementation of the ParallelFor operator that has a
///   partial specialization for the device.
///
/// This is an implementation detail of parallel_for.  Users should
/// skip this and go directly to the nonmember function parallel_for.
template <class FunctorType, class ExecPolicy,
          class ExecutionSpace = typename Impl::FunctorPolicyExecutionSpace<
              FunctorType, ExecPolicy>::execution_space>
class ParallelFor;

/// \class ParallelReduce
/// \brief Implementation detail of parallel_reduce.
///
/// This is an implementation detail of parallel_reduce.  Users should
/// skip this and go directly to the nonmember function parallel_reduce.
template <class FunctorType, class ExecPolicy, class ReducerType = InvalidType,
          class ExecutionSpace = typename Impl::FunctorPolicyExecutionSpace<
              FunctorType, ExecPolicy>::execution_space>
class ParallelReduce;

/// \class ParallelScan
/// \brief Implementation detail of parallel_scan.
///
/// This is an implementation detail of parallel_scan.  Users should
/// skip this and go directly to the documentation of the nonmember
/// template function Kokkos::parallel_scan.
template <class FunctorType, class ExecPolicy,
          class ExecutionSapce = typename Impl::FunctorPolicyExecutionSpace<
              FunctorType, ExecPolicy>::execution_space>
class ParallelScan;

template <class FunctorType, class ExecPolicy, class ReturnType = InvalidType,
          class ExecutionSapce = typename Impl::FunctorPolicyExecutionSpace<
              FunctorType, ExecPolicy>::execution_space>
class ParallelScanWithTotal;

}  // namespace Impl

template <class ScalarType, class Space = HostSpace>
struct Sum;
template <class ScalarType, class Space = HostSpace>
struct Prod;
template <class ScalarType, class Space = HostSpace>
struct Min;
template <class ScalarType, class Space = HostSpace>
struct Max;
template <class ScalarType, class Space = HostSpace>
struct MinMax;
template <class ScalarType, class Index, class Space = HostSpace>
struct MinLoc;
template <class ScalarType, class Index, class Space = HostSpace>
struct MaxLoc;
template <class ScalarType, class Index, class Space = HostSpace>
struct MinMaxLoc;
template <class ScalarType, class Space = HostSpace>
struct BAnd;
template <class ScalarType, class Space = HostSpace>
struct BOr;
template <class ScalarType, class Space = HostSpace>
struct LAnd;
template <class ScalarType, class Space = HostSpace>
struct LOr;

}  // namespace Kokkos
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE
namespace Kokkos {
template <class ScalarType>
struct MinMaxScalar;
template <class ScalarType, class Index>
struct MinMaxLocScalar;
template <class ScalarType, class Index>
struct ValLocScalar;

namespace Experimental {
using Kokkos::BAnd;
using Kokkos::BOr;
using Kokkos::LAnd;
using Kokkos::LOr;
using Kokkos::Max;
using Kokkos::MaxLoc;
using Kokkos::Min;
using Kokkos::MinLoc;
using Kokkos::MinMax;
using Kokkos::MinMaxLoc;
using Kokkos::MinMaxLocScalar;
using Kokkos::MinMaxScalar;
using Kokkos::Prod;
using Kokkos::Sum;
using Kokkos::ValLocScalar;
}  // namespace Experimental
}  // namespace Kokkos
#endif

#endif /* #ifndef KOKKOS_CORE_FWD_HPP */
