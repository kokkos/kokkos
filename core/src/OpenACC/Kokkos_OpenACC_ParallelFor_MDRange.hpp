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

#ifndef KOKKOS_OPENACC_PARALLEL_FOR_MDRANGE_HPP
#define KOKKOS_OPENACC_PARALLEL_FOR_MDRANGE_HPP

#include <OpenACC/Kokkos_OpenACC.hpp>
#include <OpenACC/Kokkos_OpenACC_FunctorAdapter.hpp>
#include <Kokkos_Parallel.hpp>

namespace Kokkos::Experimental::Impl {

template <class Functor, class Policy, int Rank = Policy::rank>
struct OpenACCParallelForHelper {
  OpenACCParallelForHelper(Functor const&, Policy const&, int) {
    static_assert(std::is_void_v<Functor>, "not implemented");
  }
};

}  // namespace Kokkos::Experimental::Impl

template <class Functor, class... Traits>
class Kokkos::Impl::ParallelFor<Functor, Kokkos::MDRangePolicy<Traits...>,
                                Kokkos::Experimental::OpenACC> {
  using Policy = MDRangePolicy<Traits...>;
  Kokkos::Experimental::Impl::FunctorAdapter<Functor, Policy> m_functor;
  Policy m_policy;

 public:
  ParallelFor(Functor const& functor, Policy const& policy)
      : m_functor(functor), m_policy(policy) {}

  void execute() const {
    static_assert(Policy::rank < 7 && Policy::rank > 1,
                  "OpenACC Backend MDRangePolicy Error: Unsupported rank...");
    static_assert(Policy::inner_direction == Iterate::Left ||
                  Policy::inner_direction == Iterate::Right);
    int const async_arg = m_policy.space().acc_async_queue();
    Kokkos::Experimental::Impl::OpenACCParallelForHelper(m_functor, m_policy,
                                                         async_arg);
  }
};

template <class Functor, class... Traits>
struct Kokkos::Experimental::Impl::OpenACCParallelForHelper<
    Functor, Kokkos::MDRangePolicy<Traits...>, 2> {
  using Policy = MDRangePolicy<Traits...>;
  OpenACCParallelForHelper(Functor const& functor, Policy const& policy,
                           int async_arg) {
    auto const begin1 = policy.m_lower[1];
    auto const end1   = policy.m_upper[1];
    auto const begin0 = policy.m_lower[0];
    auto const end0   = policy.m_upper[0];

    if ((end0 <= begin0) || (end1 <= begin1)) {
      return;
    }

    if constexpr (Policy::inner_direction == Iterate::Left) {
#pragma acc parallel loop gang vector collapse(2) copyin(functor) \
    async(async_arg)
      for (auto i1 = begin1; i1 < end1; ++i1) {
        for (auto i0 = begin0; i0 < end0; ++i0) {
          functor(i0, i1);
        }
      }
    } else if constexpr (Policy::inner_direction == Iterate::Right) {
#pragma acc parallel loop gang vector collapse(2) copyin(functor) \
    async(async_arg)
      for (auto i0 = begin0; i0 < end0; ++i0) {
        for (auto i1 = begin1; i1 < end1; ++i1) {
          functor(i0, i1);
        }
      }
    } else {
      static_assert(false,
                    "Kokkos Error: implementation bug in the OpenACC backend");
    }
  }
};

template <class Functor, class... Traits>
struct Kokkos::Experimental::Impl::OpenACCParallelForHelper<
    Functor, Kokkos::MDRangePolicy<Traits...>, 3> {
  using Policy = MDRangePolicy<Traits...>;
  OpenACCParallelForHelper(Functor const& functor, Policy const& policy,
                           int async_arg) {
    auto const begin2 = policy.m_lower[2];
    auto const end2   = policy.m_upper[2];
    auto const begin1 = policy.m_lower[1];
    auto const end1   = policy.m_upper[1];
    auto const begin0 = policy.m_lower[0];
    auto const end0   = policy.m_upper[0];

    if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2)) {
      return;
    }

    if constexpr (Policy::inner_direction == Iterate::Left) {
#pragma acc parallel loop gang vector collapse(3) copyin(functor) \
    async(async_arg)
      for (auto i2 = begin2; i2 < end2; ++i2) {
        for (auto i1 = begin1; i1 < end1; ++i1) {
          for (auto i0 = begin0; i0 < end0; ++i0) {
            functor(i0, i1, i2);
          }
        }
      }
    } else if constexpr (Policy::inner_direction == Iterate::Right) {
#pragma acc parallel loop gang vector collapse(3) copyin(functor) \
    async(async_arg)
      for (auto i0 = begin0; i0 < end0; ++i0) {
        for (auto i1 = begin1; i1 < end1; ++i1) {
          for (auto i2 = begin2; i2 < end2; ++i2) {
            functor(i0, i1, i2);
          }
        }
      }
    } else {
      static_assert(false,
                    "Kokkos Error: implementation bug in the OpenACC backend");
    }
  }
};

template <class Functor, class... Traits>
struct Kokkos::Experimental::Impl::OpenACCParallelForHelper<
    Functor, Kokkos::MDRangePolicy<Traits...>, 4> {
  using Policy = MDRangePolicy<Traits...>;
  OpenACCParallelForHelper(Functor const& functor, Policy const& policy,
                           int async_arg) {
    auto const begin3 = policy.m_lower[3];
    auto const end3   = policy.m_upper[3];
    auto const begin2 = policy.m_lower[2];
    auto const end2   = policy.m_upper[2];
    auto const begin1 = policy.m_lower[1];
    auto const end1   = policy.m_upper[1];
    auto const begin0 = policy.m_lower[0];
    auto const end0   = policy.m_upper[0];

    if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
        (end3 <= begin3)) {
      return;
    }

    if constexpr (Policy::inner_direction == Iterate::Left) {
#pragma acc parallel loop gang vector collapse(4) copyin(functor) \
    async(async_arg)
      for (auto i3 = begin3; i3 < end3; ++i3) {
        for (auto i2 = begin2; i2 < end2; ++i2) {
          for (auto i1 = begin1; i1 < end1; ++i1) {
            for (auto i0 = begin0; i0 < end0; ++i0) {
              functor(i0, i1, i2, i3);
            }
          }
        }
      }
    } else if constexpr (Policy::inner_direction == Iterate::Right) {
#pragma acc parallel loop gang vector collapse(4) copyin(functor) \
    async(async_arg)
      for (auto i0 = begin0; i0 < end0; ++i0) {
        for (auto i1 = begin1; i1 < end1; ++i1) {
          for (auto i2 = begin2; i2 < end2; ++i2) {
            for (auto i3 = begin3; i3 < end3; ++i3) {
              functor(i0, i1, i2, i3);
            }
          }
        }
      }
    } else {
      static_assert(false,
                    "Kokkos Error: implementation bug in the OpenACC backend");
    }
  }
};

template <class Functor, class... Traits>
struct Kokkos::Experimental::Impl::OpenACCParallelForHelper<
    Functor, Kokkos::MDRangePolicy<Traits...>, 5> {
  using Policy = MDRangePolicy<Traits...>;
  OpenACCParallelForHelper(Functor const& functor, Policy const& policy,
                           int async_arg) {
    auto const begin4 = policy.m_lower[4];
    auto const end4   = policy.m_upper[4];
    auto const begin3 = policy.m_lower[3];
    auto const end3   = policy.m_upper[3];
    auto const begin2 = policy.m_lower[2];
    auto const end2   = policy.m_upper[2];
    auto const begin1 = policy.m_lower[1];
    auto const end1   = policy.m_upper[1];
    auto const begin0 = policy.m_lower[0];
    auto const end0   = policy.m_upper[0];

    if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
        (end3 <= begin3) || (end4 <= begin4)) {
      return;
    }

    if constexpr (Policy::inner_direction == Iterate::Left) {
#pragma acc parallel loop gang vector collapse(5) copyin(functor) \
    async(async_arg)
      for (auto i4 = begin4; i4 < end4; ++i4) {
        for (auto i3 = begin3; i3 < end3; ++i3) {
          for (auto i2 = begin2; i2 < end2; ++i2) {
            for (auto i1 = begin1; i1 < end1; ++i1) {
              for (auto i0 = begin0; i0 < end0; ++i0) {
                functor(i0, i1, i2, i3, i4);
              }
            }
          }
        }
      }
    } else if constexpr (Policy::inner_direction == Iterate::Right) {
#pragma acc parallel loop gang vector collapse(5) copyin(functor) \
    async(async_arg)
      for (auto i0 = begin0; i0 < end0; ++i0) {
        for (auto i1 = begin1; i1 < end1; ++i1) {
          for (auto i2 = begin2; i2 < end2; ++i2) {
            for (auto i3 = begin3; i3 < end3; ++i3) {
              for (auto i4 = begin4; i4 < end4; ++i4) {
                functor(i0, i1, i2, i3, i4);
              }
            }
          }
        }
      }
    } else {
      static_assert(false,
                    "Kokkos Error: implementation bug in the OpenACC backend");
    }
  }
};

template <class Functor, class... Traits>
struct Kokkos::Experimental::Impl::OpenACCParallelForHelper<
    Functor, Kokkos::MDRangePolicy<Traits...>, 6> {
  using Policy = MDRangePolicy<Traits...>;
  OpenACCParallelForHelper(Functor const& functor, Policy const& policy,
                           int async_arg) {
    auto const begin5 = policy.m_lower[5];
    auto const end5   = policy.m_upper[5];
    auto const begin4 = policy.m_lower[4];
    auto const end4   = policy.m_upper[4];
    auto const begin3 = policy.m_lower[3];
    auto const end3   = policy.m_upper[3];
    auto const begin2 = policy.m_lower[2];
    auto const end2   = policy.m_upper[2];
    auto const begin1 = policy.m_lower[1];
    auto const end1   = policy.m_upper[1];
    auto const begin0 = policy.m_lower[0];
    auto const end0   = policy.m_upper[0];

    if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
        (end3 <= begin3) || (end4 <= begin4) || (end5 <= begin5)) {
      return;
    }

    if constexpr (Policy::inner_direction == Iterate::Left) {
#pragma acc parallel loop gang vector collapse(6) copyin(functor) \
    async(async_arg)
      for (auto i5 = begin5; i5 < end5; ++i5) {
        for (auto i4 = begin4; i4 < end4; ++i4) {
          for (auto i3 = begin3; i3 < end3; ++i3) {
            for (auto i2 = begin2; i2 < end2; ++i2) {
              for (auto i1 = begin1; i1 < end1; ++i1) {
                for (auto i0 = begin0; i0 < end0; ++i0) {
                  functor(i0, i1, i2, i3, i4, i5);
                }
              }
            }
          }
        }
      }
    } else if constexpr (Policy::inner_direction == Iterate::Right) {
#pragma acc parallel loop gang vector collapse(6) copyin(functor) \
    async(async_arg)
      for (auto i0 = begin0; i0 < end0; ++i0) {
        for (auto i1 = begin1; i1 < end1; ++i1) {
          for (auto i2 = begin2; i2 < end2; ++i2) {
            for (auto i3 = begin3; i3 < end3; ++i3) {
              for (auto i4 = begin4; i4 < end4; ++i4) {
                for (auto i5 = begin5; i5 < end5; ++i5) {
                  functor(i0, i1, i2, i3, i4, i5);
                }
              }
            }
          }
        }
      }
    } else {
      static_assert(false,
                    "Kokkos Error: implementation bug in the OpenACC backend");
    }
  }
};

#endif
