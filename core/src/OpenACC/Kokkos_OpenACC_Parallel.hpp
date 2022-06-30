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

#ifndef KOKKOS_OPENACC_PARALLEL_HPP
#define KOKKOS_OPENACC_PARALLEL_HPP

#include <sstream>
#include <Kokkos_Parallel.hpp>
#include <OpenACC/Kokkos_OpenACC_Exec.hpp>
#include <impl/Kokkos_FunctorAdapter.hpp>

#define KOKKOS_IMPL_LOCK_FREE_HIERARCHICAL

namespace Kokkos {
namespace Impl {

template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::RangePolicy<Traits...>,
                  Kokkos::Experimental::OpenACC> {
 private:
  using Policy    = Kokkos::RangePolicy<Traits...>;
  using WorkTag   = typename Policy::work_tag;
  using WorkRange = typename Policy::WorkRange;
  using Member    = typename Policy::member_type;

  const FunctorType m_functor;
  const Policy m_policy;

 public:
  inline void execute() const { execute_impl<WorkTag>(); }

  template <class TagType>
  inline void execute_impl() const {
    OpenACCExec::verify_is_process(
        "Kokkos::Experimental::OpenACC parallel_for");
    OpenACCExec::verify_initialized(
        "Kokkos::Experimental::OpenACC parallel_for");

    const auto begin = m_policy.begin();
    const auto end   = m_policy.end();

    if (end <= begin) return;

    const FunctorType a_functor(m_functor);

    if constexpr (std::is_void<TagType>::value) {
#pragma acc parallel loop gang vector copyin(a_functor)
      for (auto i = begin; i < end; i++) a_functor(i);
    } else {
#pragma acc parallel loop gang vector copyin(a_functor)
      for (auto i = begin; i < end; i++) a_functor(TagType(), i);
    }
  }

  inline ParallelFor(const FunctorType& arg_functor, Policy arg_policy)
      : m_functor(arg_functor), m_policy(arg_policy) {}
};

template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::MDRangePolicy<Traits...>,
                  Kokkos::Experimental::OpenACC> {
 private:
  using Policy  = Kokkos::MDRangePolicy<Traits...>;
  using WorkTag = typename Policy::work_tag;
  using Member  = typename Policy::member_type;

  const FunctorType m_functor;
  const Policy m_policy;

 public:
  template <class TagType, int Rank>
  inline std::enable_if_t<Rank == 2> execute_functor(
      const FunctorType& functor, const Policy& policy) const {
    const FunctorType a_functor(functor);

    int begin1 = policy.m_lower[1];
    int end1   = policy.m_upper[1];
    int begin2 = policy.m_lower[0];
    int end2   = policy.m_upper[0];

    if constexpr (std::is_void<TagType>::value) {
#pragma acc parallel loop gang vector collapse(2) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          a_functor(i1, i0);
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(2) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          a_functor(TagType(), i1, i0);
        }
      }
    }
  }

  template <class TagType, int Rank>
  inline std::enable_if_t<Rank == 3> execute_functor(
      const FunctorType& functor, const Policy& policy) const {
    const FunctorType a_functor(functor);
    int begin1 = policy.m_lower[2];
    int end1   = policy.m_upper[2];
    int begin2 = policy.m_lower[1];
    int end2   = policy.m_upper[1];
    int begin3 = policy.m_lower[0];
    int end3   = policy.m_upper[0];

    if constexpr (std::is_void<TagType>::value) {
#pragma acc parallel loop gang vector collapse(3) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            a_functor(i2, i1, i0);
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(3) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            a_functor(TagType(), i2, i1, i0);
          }
        }
      }
    }
  }

  template <class TagType, int Rank>
  inline std::enable_if_t<Rank == 4> execute_functor(
      const FunctorType& functor, const Policy& policy) const {
    const FunctorType a_functor(functor);
    int begin1 = policy.m_lower[3];
    int end1   = policy.m_upper[3];
    int begin2 = policy.m_lower[2];
    int end2   = policy.m_upper[2];
    int begin3 = policy.m_lower[1];
    int end3   = policy.m_upper[1];
    int begin4 = policy.m_lower[0];
    int end4   = policy.m_upper[0];

    if constexpr (std::is_void<TagType>::value) {
#pragma acc parallel loop gang vector collapse(4) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              a_functor(i3, i2, i1, i0);
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(4) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              a_functor(TagType(), i3, i2, i1, i0);
            }
          }
        }
      }
    }
  }

  template <class TagType, int Rank>
  inline std::enable_if_t<Rank == 5> execute_functor(
      const FunctorType& functor, const Policy& policy) const {
    const FunctorType a_functor(functor);
    int begin1 = policy.m_lower[4];
    int end1   = policy.m_upper[4];
    int begin2 = policy.m_lower[3];
    int end2   = policy.m_upper[3];
    int begin3 = policy.m_lower[2];
    int end3   = policy.m_upper[2];
    int begin4 = policy.m_lower[1];
    int end4   = policy.m_upper[1];
    int begin5 = policy.m_lower[0];
    int end5   = policy.m_upper[0];

    if constexpr (std::is_void<TagType>::value) {
#pragma acc parallel loop gang vector collapse(5) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                a_functor(i4, i3, i2, i1, i0);
              }
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(5) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                a_functor(TagType(), i4, i3, i2, i1, i0);
              }
            }
          }
        }
      }
    }
  }

  template <class TagType, int Rank>
  inline std::enable_if_t<Rank == 6> execute_functor(
      const FunctorType& functor, const Policy& policy) const {
    const FunctorType a_functor(functor);
    int begin1 = policy.m_lower[5];
    int end1   = policy.m_upper[5];
    int begin2 = policy.m_lower[4];
    int end2   = policy.m_upper[4];
    int begin3 = policy.m_lower[3];
    int end3   = policy.m_upper[3];
    int begin4 = policy.m_lower[2];
    int end4   = policy.m_upper[2];
    int begin5 = policy.m_lower[1];
    int end5   = policy.m_upper[1];
    int begin6 = policy.m_lower[0];
    int end6   = policy.m_upper[0];

    if constexpr (std::is_void<TagType>::value) {
#pragma acc parallel loop gang vector collapse(6) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  a_functor(i5, i4, i3, i2, i1, i0);
                }
              }
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(6) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  a_functor(TagType(), i5, i4, i3, i2, i1, i0);
                }
              }
            }
          }
        }
      }
    }
  }

  template <class TagType, int Rank>
  inline std::enable_if_t<Rank == 7> execute_functor(
      const FunctorType& functor, const Policy& policy) const {
    int begin1 = policy.m_lower[6];
    int end1   = policy.m_upper[6];
    int begin2 = policy.m_lower[5];
    int end2   = policy.m_upper[5];
    int begin3 = policy.m_lower[4];
    int end3   = policy.m_upper[4];
    int begin4 = policy.m_lower[3];
    int end4   = policy.m_upper[3];
    int begin5 = policy.m_lower[2];
    int end5   = policy.m_upper[2];
    int begin6 = policy.m_lower[1];
    int end6   = policy.m_upper[1];
    int begin7 = policy.m_lower[0];
    int end7   = policy.m_upper[0];
    const FunctorType a_functor(functor);

    if constexpr (std::is_void<TagType>::value) {
#pragma acc parallel loop gang vector collapse(7) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  for (auto i6 = begin7; i6 < end7; i6++) {
                    a_functor(i6, i5, i4, i3, i2, i1, i0);
                  }
                }
              }
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(7) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  for (auto i6 = begin7; i6 < end7; i6++) {
                    a_functor(TagType(), i6, i5, i4, i3, i2, i1, i0);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  template <class TagType, int Rank>
  inline std::enable_if_t<Rank == 8> execute_functor(
      const FunctorType& functor, const Policy& policy) const {
    const FunctorType a_functor(functor);
    int begin1 = policy.m_lower[7];
    int end1   = policy.m_upper[7];
    int begin2 = policy.m_lower[6];
    int end2   = policy.m_upper[6];
    int begin3 = policy.m_lower[5];
    int end3   = policy.m_upper[5];
    int begin4 = policy.m_lower[4];
    int end4   = policy.m_upper[4];
    int begin5 = policy.m_lower[3];
    int end5   = policy.m_upper[3];
    int begin6 = policy.m_lower[2];
    int end6   = policy.m_upper[2];
    int begin7 = policy.m_lower[1];
    int end7   = policy.m_upper[1];
    int begin8 = policy.m_lower[0];
    int end8   = policy.m_upper[0];

    if constexpr (std::is_void<TagType>::value) {
#pragma acc parallel loop gang vector collapse(8) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  for (auto i6 = begin7; i6 < end7; i6++) {
                    for (auto i7 = begin8; i7 < end8; i7++) {
                      a_functor(i7, i6, i5, i4, i3, i2, i1, i0);
                    }
                  }
                }
              }
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(8) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  for (auto i6 = begin7; i6 < end7; i6++) {
                    for (auto i7 = begin8; i7 < end8; i7++) {
                      a_functor(TagType(), i7, i6, i5, i4, i3, i2, i1, i0);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  inline void execute() const { execute_impl<WorkTag>(); }

  template <class TagType>
  inline void execute_impl() const {
    OpenACCExec::verify_is_process(
        "Kokkos::Experimental::OpenACC parallel_for");
    OpenACCExec::verify_initialized(
        "Kokkos::Experimental::OpenACC parallel_for");

    if (Policy::rank < 9) {
      execute_functor<WorkTag, Policy::rank>(m_functor, m_policy);
    } else {
      printf("Rank >= 9 not supported\n");
    }
  }

  inline ParallelFor(const FunctorType& arg_functor, Policy arg_policy)
      : m_functor(arg_functor), m_policy(arg_policy) {}
};

template <class FunctorType, class... Properties>
class ParallelFor<FunctorType, Kokkos::TeamPolicy<Properties...>,
                  Kokkos::Experimental::OpenACC> {
 private:
  using Policy = Kokkos::Impl::TeamPolicyInternal<Kokkos::Experimental::OpenACC,
                                                  Properties...>;
  using WorkTag = typename Policy::work_tag;
  using Member  = typename Policy::member_type;

  const FunctorType m_functor;
  const Policy m_policy;

 public:
  inline void execute() const { execute_impl<WorkTag>(); }

  template <class TagType>
  inline void execute_impl() const {
    OpenACCExec::verify_is_process(
        "Kokkos::Experimental::OpenACC parallel_for");
    OpenACCExec::verify_initialized(
        "Kokkos::Experimental::OpenACC parallel_for");

    auto league_size   = m_policy.league_size();
    auto team_size     = m_policy.team_size();
    auto vector_length = m_policy.impl_vector_length();

    const FunctorType a_functor(m_functor);

#ifdef KOKKOS_ENABLE_COLLAPSE_HIERARCHICAL_CONSTRUCTS
#pragma acc parallel loop gang vector num_gangs(league_size) \
    vector_length(team_size* vector_length) copyin(a_functor)
    for (int i = 0; i < league_size * team_size * vector_length; i++) {
      int league_id = i / (team_size * vector_length);
      typename Policy::member_type team(league_id, league_size, team_size,
                                        vector_length);
      if constexpr (std::is_void<TagType>::value)
        a_functor(team);
      else
        a_functor(TagType(), team);
    }
#else
#pragma acc parallel loop gang num_gangs(league_size) num_workers(team_size) \
    vector_length(vector_length) copyin(a_functor)
    for (int i = 0; i < league_size; i++) {
      int league_id = i;
      typename Policy::member_type team(league_id, league_size, team_size,
                                        vector_length);
      if constexpr (std::is_void<TagType>::value)
        a_functor(team);
      else
        a_functor(TagType(), team);
    }
#endif
  }

  inline ParallelFor(const FunctorType& arg_functor, Policy arg_policy)
      : m_functor(arg_functor), m_policy(arg_policy) {}
};

}  // namespace Impl
}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template <class FunctorType, class ReducerType, class... Traits>
class ParallelReduce<FunctorType, Kokkos::RangePolicy<Traits...>, ReducerType,
                     Kokkos::Experimental::OpenACC> {
 private:
  using Policy = Kokkos::RangePolicy<Traits...>;

  using WorkTag = typename Policy::work_tag;

  using ReducerConditional =
      Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                         FunctorType, ReducerType>;
  using ReducerTypeFwd = typename ReducerConditional::type;
  using WorkTagFwd =
      std::conditional_t<std::is_same<InvalidType, ReducerType>::value, WorkTag,
                         void>;
  using ValueTraits =
      Kokkos::Impl::FunctorValueTraits<ReducerTypeFwd, WorkTagFwd>;
  using ValueInit = Kokkos::Impl::FunctorValueInit<FunctorType, WorkTag>;

  using value_type   = typename ValueTraits::value_type;
  using pointer_type = typename ValueTraits::pointer_type;

  using Analysis =
      FunctorAnalysis<FunctorPatternInterface::REDUCE, Policy, FunctorType>;

  const FunctorType m_functor;
  const Policy m_policy;
  const ReducerType m_reducer;
  const pointer_type m_result_ptr;

 public:
  inline void execute() const { execute_impl<WorkTag>(); }

  template <class TagType>
  inline void execute_impl() const {
    OpenACCExec::verify_is_process(
        "Kokkos::Experimental::OpenACC parallel_reduce");
    OpenACCExec::verify_initialized(
        "Kokkos::Experimental::OpenACC parallel_reduce");
    const auto begin = m_policy.begin();
    const auto end   = m_policy.end();

    if (end <= begin) return;

    const FunctorType a_functor(m_functor);
    value_type tmp;
    ValueInit::init(a_functor, &tmp);
    static constexpr int UseReducer = is_reducer_type<ReducerType>::value;

    if constexpr (!UseReducer) {
      if constexpr (std::is_void<TagType>::value) {
#pragma acc parallel loop gang vector reduction(+ : tmp) copyin(a_functor)
        for (auto i = begin; i < end; i++) a_functor(i, tmp);
        m_result_ptr[0] = tmp;
      } else {
#pragma acc parallel loop gang vector reduction(+ : tmp) copyin(a_functor)
        for (auto i = begin; i < end; i++) a_functor(TagType(), i, tmp);
        m_result_ptr[0] = tmp;
      }
    } else {
      OpenACCReducerWrapper<ReducerType, FunctorType, Policy, TagType>::reduce(
          tmp, m_policy, a_functor);
      m_result_ptr[0] = tmp;
    }
  }

  template <class ViewType>
  inline ParallelReduce(
      const FunctorType& arg_functor, Policy arg_policy,
      const ViewType& arg_result_view,
      std::enable_if_t<Kokkos::is_view<ViewType>::value &&
                           !Kokkos::is_reducer_type<ReducerType>::value,
                       void*> = nullptr)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(InvalidType()),
        m_result_ptr(arg_result_view.data()) {}

  inline ParallelReduce(const FunctorType& arg_functor, Policy arg_policy,
                        const ReducerType& reducer)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(reducer),
        m_result_ptr(reducer.view().data()) {}
};

template <class FunctorType, class ReducerType, class... Traits>
class ParallelReduce<FunctorType, Kokkos::MDRangePolicy<Traits...>, ReducerType,
                     Kokkos::Experimental::OpenACC> {
 private:
  using Policy = Kokkos::MDRangePolicy<Traits...>;

  using WorkTag = typename Policy::work_tag;

  using ReducerConditional =
      Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                         FunctorType, ReducerType>;
  using ReducerTypeFwd = typename ReducerConditional::type;
  using WorkTagFwd =
      std::conditional_t<std::is_same<InvalidType, ReducerType>::value, WorkTag,
                         void>;
  using ValueTraits =
      Kokkos::Impl::FunctorValueTraits<ReducerTypeFwd, WorkTagFwd>;
  using ValueInit = Kokkos::Impl::FunctorValueInit<FunctorType, WorkTag>;

  using value_type   = typename ValueTraits::value_type;
  using pointer_type = typename ValueTraits::pointer_type;

  using Analysis =
      FunctorAnalysis<FunctorPatternInterface::REDUCE, Policy, FunctorType>;

  const FunctorType m_functor;
  const Policy m_policy;
  const ReducerType m_reducer;
  const pointer_type m_result_ptr;

 public:
  template <class TagType, int Rank>
  inline std::enable_if_t<Rank == 2> execute_functor(
      const FunctorType& functor, const Policy& policy) const {
    const FunctorType a_functor(functor);
    value_type tmp;
    ValueInit::init(a_functor, &tmp);
    int begin1 = policy.m_lower[1];
    int end1   = policy.m_upper[1];
    int begin2 = policy.m_lower[0];
    int end2   = policy.m_upper[0];

    static constexpr int UseReducer = is_reducer_type<ReducerType>::value;

    if constexpr (!UseReducer) {
      if constexpr (std::is_void<TagType>::value) {
#pragma acc parallel loop gang vector collapse(2) reduction(+:tmp) copyin(a_functor)
        for (auto i0 = begin1; i0 < end1; i0++) {
          for (auto i1 = begin2; i1 < end2; i1++) {
            a_functor(i1, i0, tmp);
          }
        }
        m_result_ptr[0] = tmp;
      } else {
#pragma acc parallel loop gang vector collapse(2) reduction(+:tmp) copyin(a_functor)
        for (auto i0 = begin1; i0 < end1; i0++) {
          for (auto i1 = begin2; i1 < end2; i1++) {
            a_functor(TagType(), i1, i0, tmp);
          }
        }
        m_result_ptr[0] = tmp;
      }
    } else {
      OpenACCReducerWrapperMD_Rank2<ReducerType, FunctorType, Policy,
                                    TagType>::reduce(tmp, m_policy, a_functor);
      m_result_ptr[0] = tmp;
    }
  }

  template <class TagType, int Rank>
  inline std::enable_if_t<Rank == 3> execute_functor(
      const FunctorType& functor, const Policy& policy) const {
    const FunctorType a_functor(functor);
    value_type tmp;
    ValueInit::init(a_functor, &tmp);
    int begin1 = policy.m_lower[2];
    int end1   = policy.m_upper[2];
    int begin2 = policy.m_lower[1];
    int end2   = policy.m_upper[1];
    int begin3 = policy.m_lower[0];
    int end3   = policy.m_upper[0];

    static constexpr int UseReducer = is_reducer_type<ReducerType>::value;

    if constexpr (!UseReducer) {
      if constexpr (std::is_void<TagType>::value) {
#pragma acc parallel loop gang vector collapse(3) reduction(+:tmp) copyin(a_functor)
        for (auto i0 = begin1; i0 < end1; i0++) {
          for (auto i1 = begin2; i1 < end2; i1++) {
            for (auto i2 = begin3; i2 < end3; i2++) {
              a_functor(i2, i1, i0, tmp);
            }
          }
        }
        m_result_ptr[0] = tmp;
      } else {
#pragma acc parallel loop gang vector collapse(3) reduction(+:tmp) copyin(a_functor)
        for (auto i0 = begin1; i0 < end1; i0++) {
          for (auto i1 = begin2; i1 < end2; i1++) {
            for (auto i2 = begin3; i2 < end3; i2++) {
              a_functor(TagType(), i2, i1, i0, tmp);
            }
          }
        }
        m_result_ptr[0] = tmp;
      }
    } else {
      OpenACCReducerWrapperMD_Rank3<ReducerType, FunctorType, Policy,
                                    TagType>::reduce(tmp, m_policy, a_functor);
      m_result_ptr[0] = tmp;
    }
  }

  template <class TagType, int Rank>
  inline std::enable_if_t<Rank == 4> execute_functor(
      const FunctorType& functor, const Policy& policy) const {
    const FunctorType a_functor(functor);
    value_type tmp;
    ValueInit::init(a_functor, &tmp);
    int begin1 = policy.m_lower[3];
    int end1   = policy.m_upper[3];
    int begin2 = policy.m_lower[2];
    int end2   = policy.m_upper[2];
    int begin3 = policy.m_lower[1];
    int end3   = policy.m_upper[1];
    int begin4 = policy.m_lower[0];
    int end4   = policy.m_upper[0];

    static constexpr int UseReducer = is_reducer_type<ReducerType>::value;

    if constexpr (!UseReducer) {
      if constexpr (std::is_void<TagType>::value) {
#pragma acc parallel loop gang vector collapse(4) reduction(+:tmp) copyin(a_functor)
        for (auto i0 = begin1; i0 < end1; i0++) {
          for (auto i1 = begin2; i1 < end2; i1++) {
            for (auto i2 = begin3; i2 < end3; i2++) {
              for (auto i3 = begin4; i3 < end4; i3++) {
                a_functor(i3, i2, i1, i0, tmp);
              }
            }
          }
        }
        m_result_ptr[0] = tmp;
      } else {
#pragma acc parallel loop gang vector collapse(4) reduction(+:tmp) copyin(a_functor)
        for (auto i0 = begin1; i0 < end1; i0++) {
          for (auto i1 = begin2; i1 < end2; i1++) {
            for (auto i2 = begin3; i2 < end3; i2++) {
              for (auto i3 = begin4; i3 < end4; i3++) {
                a_functor(TagType(), i3, i2, i1, i0, tmp);
              }
            }
          }
        }
        m_result_ptr[0] = tmp;
      }
    } else {
      OpenACCReducerWrapperMD_Rank4<ReducerType, FunctorType, Policy,
                                    TagType>::reduce(tmp, m_policy, a_functor);
      m_result_ptr[0] = tmp;
    }
  }

  template <class TagType, int Rank>
  inline std::enable_if_t<Rank == 5> execute_functor(
      const FunctorType& functor, const Policy& policy) const {
    const FunctorType a_functor(functor);
    value_type tmp;
    ValueInit::init(a_functor, &tmp);
    int begin1 = policy.m_lower[4];
    int end1   = policy.m_upper[4];
    int begin2 = policy.m_lower[3];
    int end2   = policy.m_upper[3];
    int begin3 = policy.m_lower[2];
    int end3   = policy.m_upper[2];
    int begin4 = policy.m_lower[1];
    int end4   = policy.m_upper[1];
    int begin5 = policy.m_lower[0];
    int end5   = policy.m_upper[0];

    static constexpr int UseReducer = is_reducer_type<ReducerType>::value;

    if constexpr (!UseReducer) {
      if constexpr (std::is_void<TagType>::value) {
#pragma acc parallel loop gang vector collapse(5) reduction(+:tmp) copyin(a_functor)
        for (auto i0 = begin1; i0 < end1; i0++) {
          for (auto i1 = begin2; i1 < end2; i1++) {
            for (auto i2 = begin3; i2 < end3; i2++) {
              for (auto i3 = begin4; i3 < end4; i3++) {
                for (auto i4 = begin5; i4 < end5; i4++) {
                  a_functor(i4, i3, i2, i1, i0, tmp);
                }
              }
            }
          }
        }
        m_result_ptr[0] = tmp;
      } else {
#pragma acc parallel loop gang vector collapse(5) reduction(+:tmp) copyin(a_functor)
        for (auto i0 = begin1; i0 < end1; i0++) {
          for (auto i1 = begin2; i1 < end2; i1++) {
            for (auto i2 = begin3; i2 < end3; i2++) {
              for (auto i3 = begin4; i3 < end4; i3++) {
                for (auto i4 = begin5; i4 < end5; i4++) {
                  a_functor(TagType(), i4, i3, i2, i1, i0, tmp);
                }
              }
            }
          }
        }
        m_result_ptr[0] = tmp;
      }
    } else {
      OpenACCReducerWrapperMD_Rank5<ReducerType, FunctorType, Policy,
                                    TagType>::reduce(tmp, m_policy, a_functor);
      m_result_ptr[0] = tmp;
    }
  }

  template <class TagType, int Rank>
  inline std::enable_if_t<Rank == 6> execute_functor(
      const FunctorType& functor, const Policy& policy) const {
    const FunctorType a_functor(functor);
    value_type tmp;
    ValueInit::init(a_functor, &tmp);
    int begin1 = policy.m_lower[5];
    int end1   = policy.m_upper[5];
    int begin2 = policy.m_lower[4];
    int end2   = policy.m_upper[4];
    int begin3 = policy.m_lower[3];
    int end3   = policy.m_upper[3];
    int begin4 = policy.m_lower[2];
    int end4   = policy.m_upper[2];
    int begin5 = policy.m_lower[1];
    int end5   = policy.m_upper[1];
    int begin6 = policy.m_lower[0];
    int end6   = policy.m_upper[0];

    static constexpr int UseReducer = is_reducer_type<ReducerType>::value;

    if constexpr (!UseReducer) {
      if constexpr (std::is_void<TagType>::value) {
#pragma acc parallel loop gang vector collapse(6) reduction(+:tmp) copyin(a_functor)
        for (auto i0 = begin1; i0 < end1; i0++) {
          for (auto i1 = begin2; i1 < end2; i1++) {
            for (auto i2 = begin3; i2 < end3; i2++) {
              for (auto i3 = begin4; i3 < end4; i3++) {
                for (auto i4 = begin5; i4 < end5; i4++) {
                  for (auto i5 = begin6; i5 < end6; i5++) {
                    a_functor(i5, i4, i3, i2, i1, i0, tmp);
                  }
                }
              }
            }
          }
        }
        m_result_ptr[0] = tmp;
      } else {
#pragma acc parallel loop gang vector collapse(6) reduction(+:tmp) copyin(a_functor)
        for (auto i0 = begin1; i0 < end1; i0++) {
          for (auto i1 = begin2; i1 < end2; i1++) {
            for (auto i2 = begin3; i2 < end3; i2++) {
              for (auto i3 = begin4; i3 < end4; i3++) {
                for (auto i4 = begin5; i4 < end5; i4++) {
                  for (auto i5 = begin6; i5 < end6; i5++) {
                    a_functor(TagType(), i5, i4, i3, i2, i1, i0, tmp);
                  }
                }
              }
            }
          }
        }
        m_result_ptr[0] = tmp;
      }
    } else {
      OpenACCReducerWrapperMD_Rank6<ReducerType, FunctorType, Policy,
                                    TagType>::reduce(tmp, m_policy, a_functor);
      m_result_ptr[0] = tmp;
    }
  }

  template <class TagType, int Rank>
  inline std::enable_if_t<Rank == 7> execute_functor(
      const FunctorType& functor, const Policy& policy) const {
    int begin1 = policy.m_lower[6];
    int end1   = policy.m_upper[6];
    int begin2 = policy.m_lower[5];
    int end2   = policy.m_upper[5];
    int begin3 = policy.m_lower[4];
    int end3   = policy.m_upper[4];
    int begin4 = policy.m_lower[3];
    int end4   = policy.m_upper[3];
    int begin5 = policy.m_lower[2];
    int end5   = policy.m_upper[2];
    int begin6 = policy.m_lower[1];
    int end6   = policy.m_upper[1];
    int begin7 = policy.m_lower[0];
    int end7   = policy.m_upper[0];
    const FunctorType a_functor(functor);
    value_type tmp;
    ValueInit::init(a_functor, &tmp);

    static constexpr int UseReducer = is_reducer_type<ReducerType>::value;

    if constexpr (!UseReducer) {
      if constexpr (std::is_void<TagType>::value) {
#pragma acc parallel loop gang vector collapse(7) reduction(+:tmp) copyin(a_functor)
        for (auto i0 = begin1; i0 < end1; i0++) {
          for (auto i1 = begin2; i1 < end2; i1++) {
            for (auto i2 = begin3; i2 < end3; i2++) {
              for (auto i3 = begin4; i3 < end4; i3++) {
                for (auto i4 = begin5; i4 < end5; i4++) {
                  for (auto i5 = begin6; i5 < end6; i5++) {
                    for (auto i6 = begin7; i6 < end7; i6++) {
                      a_functor(i6, i5, i4, i3, i2, i1, i0, tmp);
                    }
                  }
                }
              }
            }
          }
        }
        m_result_ptr[0] = tmp;
      } else {
#pragma acc parallel loop gang vector collapse(7) reduction(+:tmp) copyin(a_functor)
        for (auto i0 = begin1; i0 < end1; i0++) {
          for (auto i1 = begin2; i1 < end2; i1++) {
            for (auto i2 = begin3; i2 < end3; i2++) {
              for (auto i3 = begin4; i3 < end4; i3++) {
                for (auto i4 = begin5; i4 < end5; i4++) {
                  for (auto i5 = begin6; i5 < end6; i5++) {
                    for (auto i6 = begin7; i6 < end7; i6++) {
                      a_functor(TagType(), i6, i5, i4, i3, i2, i1, i0, tmp);
                    }
                  }
                }
              }
            }
          }
        }
        m_result_ptr[0] = tmp;
      }
    } else {
      OpenACCReducerWrapperMD_Rank7<ReducerType, FunctorType, Policy,
                                    TagType>::reduce(tmp, m_policy, a_functor);
      m_result_ptr[0] = tmp;
    }
  }

  template <class TagType, int Rank>
  inline std::enable_if_t<Rank == 8> execute_functor(
      const FunctorType& functor, const Policy& policy) const {
    const FunctorType a_functor(functor);
    value_type tmp;
    ValueInit::init(a_functor, &tmp);
    int begin1 = policy.m_lower[7];
    int end1   = policy.m_upper[7];
    int begin2 = policy.m_lower[6];
    int end2   = policy.m_upper[6];
    int begin3 = policy.m_lower[5];
    int end3   = policy.m_upper[5];
    int begin4 = policy.m_lower[4];
    int end4   = policy.m_upper[4];
    int begin5 = policy.m_lower[3];
    int end5   = policy.m_upper[3];
    int begin6 = policy.m_lower[2];
    int end6   = policy.m_upper[2];
    int begin7 = policy.m_lower[1];
    int end7   = policy.m_upper[1];
    int begin8 = policy.m_lower[0];
    int end8   = policy.m_upper[0];

    static constexpr int UseReducer = is_reducer_type<ReducerType>::value;

    if constexpr (!UseReducer) {
      if constexpr (std::is_void<TagType>::value) {
#pragma acc parallel loop gang vector collapse(8) reduction(+:tmp) copyin(a_functor)
        for (auto i0 = begin1; i0 < end1; i0++) {
          for (auto i1 = begin2; i1 < end2; i1++) {
            for (auto i2 = begin3; i2 < end3; i2++) {
              for (auto i3 = begin4; i3 < end4; i3++) {
                for (auto i4 = begin5; i4 < end5; i4++) {
                  for (auto i5 = begin6; i5 < end6; i5++) {
                    for (auto i6 = begin7; i6 < end7; i6++) {
                      for (auto i7 = begin8; i7 < end8; i7++) {
                        a_functor(i7, i6, i5, i4, i3, i2, i1, i0, tmp);
                      }
                    }
                  }
                }
              }
            }
          }
        }
        m_result_ptr[0] = tmp;
      } else {
#pragma acc parallel loop gang vector collapse(8) reduction(+:tmp) copyin(a_functor)
        for (auto i0 = begin1; i0 < end1; i0++) {
          for (auto i1 = begin2; i1 < end2; i1++) {
            for (auto i2 = begin3; i2 < end3; i2++) {
              for (auto i3 = begin4; i3 < end4; i3++) {
                for (auto i4 = begin5; i4 < end5; i4++) {
                  for (auto i5 = begin6; i5 < end6; i5++) {
                    for (auto i6 = begin7; i6 < end7; i6++) {
                      for (auto i7 = begin8; i7 < end8; i7++) {
                        a_functor(TagType(), i7, i6, i5, i4, i3, i2, i1, i0,
                                  tmp);
                      }
                    }
                  }
                }
              }
            }
          }
        }
        m_result_ptr[0] = tmp;
      }
    } else {
      OpenACCReducerWrapperMD_Rank8<ReducerType, FunctorType, Policy,
                                    TagType>::reduce(tmp, m_policy, a_functor);
      m_result_ptr[0] = tmp;
    }
  }

  inline void execute() const { execute_impl<WorkTag>(); }

  template <class TagType>
  inline void execute_impl() const {
    OpenACCExec::verify_is_process(
        "Kokkos::Experimental::OpenACC parallel_reduce");
    OpenACCExec::verify_initialized(
        "Kokkos::Experimental::OpenACC parallel_reduce");

    if (Policy::rank < 9) {
      execute_functor<WorkTag, Policy::rank>(m_functor, m_policy);
    } else {
      printf("Rank >= 9 not supported\n");
    }
  }

  template <class ViewType>
  inline ParallelReduce(
      const FunctorType& arg_functor, Policy arg_policy,
      const ViewType& arg_result_view,
      std::enable_if_t<Kokkos::is_view<ViewType>::value &&
                           !Kokkos::is_reducer_type<ReducerType>::value,
                       void*> = nullptr)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(InvalidType()),
        m_result_ptr(arg_result_view.data()) {}

  inline ParallelReduce(const FunctorType& arg_functor, Policy arg_policy,
                        const ReducerType& reducer)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(reducer),
        m_result_ptr(reducer.view().data()) {}
};

template <class FunctorType, class ReducerType, class... Properties>
class ParallelReduce<FunctorType, Kokkos::TeamPolicy<Properties...>,
                     ReducerType, Kokkos::Experimental::OpenACC> {
 private:
  using Policy = Kokkos::Impl::TeamPolicyInternal<Kokkos::Experimental::OpenACC,
                                                  Properties...>;

  using WorkTag = typename Policy::work_tag;

  using ReducerConditional =
      Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                         FunctorType, ReducerType>;
  using ReducerTypeFwd = typename ReducerConditional::type;
  using WorkTagFwd =
      std::conditional_t<std::is_same<InvalidType, ReducerType>::value, WorkTag,
                         void>;
  using ValueTraits =
      Kokkos::Impl::FunctorValueTraits<ReducerTypeFwd, WorkTagFwd>;
  using ValueInit = Kokkos::Impl::FunctorValueInit<FunctorType, WorkTag>;

  using value_type   = typename ValueTraits::value_type;
  using pointer_type = typename ValueTraits::pointer_type;

  using Analysis =
      FunctorAnalysis<FunctorPatternInterface::REDUCE, Policy, FunctorType>;

  const FunctorType m_functor;
  const Policy m_policy;
  const ReducerType m_reducer;
  const pointer_type m_result_ptr;

 public:
  inline void execute() const { execute_impl<WorkTag>(); }

  template <class TagType>
  inline void execute_impl() const {
    OpenACCExec::verify_is_process(
        "Kokkos::Experimental::OpenACC parallel_reduce");
    OpenACCExec::verify_initialized(
        "Kokkos::Experimental::OpenACC parallel_reduce");

    auto league_size   = m_policy.league_size();
    auto team_size     = m_policy.team_size();
    auto vector_length = m_policy.impl_vector_length();

    const FunctorType a_functor(m_functor);
    value_type tmp;
    ValueInit::init(a_functor, &tmp);
    static constexpr int UseReducer = is_reducer_type<ReducerType>::value;

    if constexpr (!UseReducer) {
#ifdef KOKKOS_ENABLE_COLLAPSE_HIERARCHICAL_CONSTRUCTS
#pragma acc parallel loop gang vector reduction(+ : tmp) copyin(a_functor)
      for (int i = 0; i < league_size; i++) {
        int league_id = i;
        typename Policy::member_type team(league_id, league_size, 1,
                                          vector_length);
        if constexpr (std::is_void<TagType>::value)
          a_functor(team, tmp);
        else
          a_functor(TagType(), team, tmp);
      }
#else
#pragma acc parallel loop gang num_gangs(league_size) num_workers(team_size)  vector_length(vector_length) reduction(+:tmp) copyin(a_functor)
      for (int i = 0; i < league_size; i++) {
        int league_id = i;
        typename Policy::member_type team(league_id, league_size, team_size,
                                          vector_length);
        if constexpr (std::is_void<TagType>::value)
          a_functor(team, tmp);
        else
          a_functor(TagType(), team, tmp);
      }
#endif
      m_result_ptr[0] = tmp;
    } else {
      OpenACCReducerWrapperTeams<ReducerType, FunctorType, Policy,
                                 TagType>::reduce(tmp, m_policy, a_functor);
      m_result_ptr[0] = tmp;
    }
  }

  template <class ViewType>
  inline ParallelReduce(
      const FunctorType& arg_functor, Policy arg_policy,
      const ViewType& arg_result_view,
      std::enable_if_t<Kokkos::is_view<ViewType>::value &&
                           !Kokkos::is_reducer_type<ReducerType>::value,
                       void*> = nullptr)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(InvalidType()),
        m_result_ptr(arg_result_view.data()) {}

  inline ParallelReduce(const FunctorType& arg_functor, Policy arg_policy,
                        const ReducerType& reducer)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(reducer),
        m_result_ptr(reducer.view().data()) {}
};

}  // namespace Impl
}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

//[FIXME_OPENACC] Incorrect implementation
KOKKOS_INLINE_FUNCTION void acc_ext_barrier() {}

//[FIXME_OPENACC] works only if NVHPC is used.
KOKKOS_INLINE_FUNCTION int acc_ext_vectoridx() {
#ifdef __USE_NVHPC__
  return __pgi_vectoridx();
#else
  return 0;
#endif
}

//[FIXME_OPENACC] Compile-time errors (Unsupported local variable) when compiled
// by NVHPC V22.2
template <class FunctorType, class... Traits>
class ParallelScan<FunctorType, Kokkos::RangePolicy<Traits...>,
                   Kokkos::Experimental::OpenACC> {
 protected:
  using Policy = Kokkos::RangePolicy<Traits...>;

  using WorkTag   = typename Policy::work_tag;
  using WorkRange = typename Policy::WorkRange;
  using Member    = typename Policy::member_type;
  using idx_type  = typename Policy::index_type;

  using ValueTraits = Kokkos::Impl::FunctorValueTraits<FunctorType, WorkTag>;
  using ValueInit   = Kokkos::Impl::FunctorValueInit<FunctorType, WorkTag>;
  using ValueJoin   = Kokkos::Impl::FunctorValueJoin<FunctorType, WorkTag>;
  using ValueOps    = Kokkos::Impl::FunctorValueOps<FunctorType, WorkTag>;

  using value_type     = typename ValueTraits::value_type;
  using pointer_type   = typename ValueTraits::pointer_type;
  using reference_type = typename ValueTraits::reference_type;

  const FunctorType m_functor;
  const Policy m_policy;

  template <class TagType>
  std::enable_if_t<std::is_void<TagType>::value> call_with_tag(
      const FunctorType& f, const idx_type& idx, value_type& val,
      const bool& is_final) const {
    f(idx, val, is_final);
  }
  template <class TagType>
  std::enable_if_t<!std::is_void<TagType>::value> call_with_tag(
      const FunctorType& f, const idx_type& idx, value_type& val,
      const bool& is_final) const {
    f(WorkTag(), idx, val, is_final);
  }

 public:
  void impl_execute(
      Kokkos::View<value_type**, Kokkos::LayoutRight,
                   Kokkos::Experimental::OpenACCSpace>
          element_values,
      Kokkos::View<value_type*, Kokkos::Experimental::OpenACCSpace>
          chunk_values,
      Kokkos::View<unsigned long long int, Kokkos::Experimental::OpenACCSpace>
          count) const {
    const idx_type N          = m_policy.end() - m_policy.begin();
    const idx_type chunk_size = 128;
    const idx_type n_chunks   = (N + chunk_size - 1) / chunk_size;
    idx_type nteams           = n_chunks > 512 ? 512 : n_chunks;
    idx_type team_size        = 128;

    FunctorType a_functor(m_functor);
    std::cerr << "Kokkos::Experimental::OpenACC ERROR: parallel_scan() is not "
                 "supported for now; exit!"
              << std::endl;
    exit(0);
#if 0
#pragma acc parallel loop gang num_gangs(nteams) vector_length(team_size) \
    copyin(a_functor)
    for (idx_type team_id = 0; team_id < n_chunks; ++team_id) {
      {
        const idx_type local_offset = team_id * chunk_size;
#pragma acc loop vector 
        for (idx_type i = 0; i < chunk_size; ++i) {
          const idx_type idx = local_offset + i;
          value_type val;
          ValueInit::init(a_functor, &val);
          if (idx < N) call_with_tag<WorkTag>(a_functor, idx, val, false);
          element_values(team_id, i) = val;
        }
        //acc_ext_barrier();
        if (acc_ext_vectoridx() == 0) {
          value_type sum;
          ValueInit::init(a_functor, &sum);
          for (idx_type i = 0; i < chunk_size; ++i) {
            ValueJoin::join(a_functor, &sum, &element_values(team_id, i));
            element_values(team_id, i) = sum;
          }
          chunk_values(team_id) = sum;
        }
        //acc_ext_barrier();
        if (acc_ext_vectoridx() == 0) {
          if (Kokkos::atomic_fetch_add(&count(), 1ULL) == n_chunks - 1) {
            value_type sum;
            ValueInit::init(a_functor, &sum);
            for (idx_type i = 0; i < n_chunks; ++i) {
              ValueJoin::join(a_functor, &sum, &chunk_values(i));
              chunk_values(i) = sum;
            }
          }
        }
      }
    }

#pragma acc parallel loop gang num_gangs(nteams) vector_length(team_size) \
    copyin(a_functor)
    for (idx_type team_id = 0; team_id < n_chunks; ++team_id) {
      {
        const idx_type local_offset = team_id * chunk_size;
        value_type offset_value;
        if (team_id > 0)
          offset_value = chunk_values(team_id - 1);
        else
          ValueInit::init(a_functor, &offset_value);
#pragma acc loop vector
        for (idx_type i = 0; i < chunk_size; ++i) {
          const idx_type idx = local_offset + i;
          value_type local_offset_value;
          if (i > 0) {
            local_offset_value = element_values(team_id, i - 1);
            ValueJoin::join(a_functor, &local_offset_value, &offset_value);
          } else
            local_offset_value = offset_value;
          if (idx < N)
            call_with_tag<WorkTag>(a_functor, idx, local_offset_value, true);
        }
      }
    }
#endif
  }

  void execute() const {
    OpenACCExec::verify_is_process(
        "Kokkos::Experimental::OpenACC parallel_for");
    OpenACCExec::verify_initialized(
        "Kokkos::Experimental::OpenACC parallel_for");
    const idx_type N          = m_policy.end() - m_policy.begin();
    const idx_type chunk_size = 128;
    const idx_type n_chunks   = (N + chunk_size - 1) / chunk_size;

    // This could be scratch memory per team
    Kokkos::View<value_type**, Kokkos::LayoutRight,
                 Kokkos::Experimental::OpenACCSpace>
        element_values("element_values", n_chunks, chunk_size);
    Kokkos::View<value_type*, Kokkos::Experimental::OpenACCSpace> chunk_values(
        "chunk_values", n_chunks);
    Kokkos::View<unsigned long long int, Kokkos::Experimental::OpenACCSpace>
        count("Count");

    impl_execute(element_values, chunk_values, count);
  }

  //----------------------------------------

  ParallelScan(const FunctorType& arg_functor, const Policy& arg_policy)
      : m_functor(arg_functor), m_policy(arg_policy) {}

  //----------------------------------------
};

template <class FunctorType, class ReturnType, class... Traits>
class ParallelScanWithTotal<FunctorType, Kokkos::RangePolicy<Traits...>,
                            ReturnType, Kokkos::Experimental::OpenACC>
    : public ParallelScan<FunctorType, Kokkos::RangePolicy<Traits...>,
                          Kokkos::Experimental::OpenACC> {
  using base_t     = ParallelScan<FunctorType, Kokkos::RangePolicy<Traits...>,
                              Kokkos::Experimental::OpenACC>;
  using value_type = typename base_t::value_type;
  value_type& m_returnvalue;

 public:
  void execute() const {
    OpenACCExec::verify_is_process(
        "Kokkos::Experimental::OpenACC parallel_for");
    OpenACCExec::verify_initialized(
        "Kokkos::Experimental::OpenACC parallel_for");
    const int64_t N        = base_t::m_policy.end() - base_t::m_policy.begin();
    const int chunk_size   = 128;
    const int64_t n_chunks = (N + chunk_size - 1) / chunk_size;

    if (N > 0) {
      // This could be scratch memory per team
      Kokkos::View<value_type**, Kokkos::LayoutRight,
                   Kokkos::Experimental::OpenACCSpace>
          element_values("element_values", n_chunks, chunk_size);
      Kokkos::View<value_type*, Kokkos::Experimental::OpenACCSpace>
          chunk_values("chunk_values", n_chunks);
      Kokkos::View<unsigned long long int, Kokkos::Experimental::OpenACCSpace>
          count("Count");

      base_t::impl_execute(element_values, chunk_values, count);

      const int size = base_t::ValueTraits::value_size(base_t::m_functor);
      DeepCopy<HostSpace, Kokkos::Experimental::OpenACCSpace>(
          &m_returnvalue, chunk_values.data() + (n_chunks - 1), size);
    } else {
      m_returnvalue = 0;
    }
  }

  ParallelScanWithTotal(const FunctorType& arg_functor,
                        const typename base_t::Policy& arg_policy,
                        ReturnType& arg_returnvalue)
      : base_t(arg_functor, arg_policy), m_returnvalue(arg_returnvalue) {}
};

}  // namespace Impl
}  // namespace Kokkos

namespace Kokkos {
namespace Impl {

template <typename iType>
struct TeamThreadRangeBoundariesStruct<iType, OpenACCExecTeamMember> {
  using index_type = iType;
  const iType start;
  const iType end;
  const OpenACCExecTeamMember& team;

  inline TeamThreadRangeBoundariesStruct(const OpenACCExecTeamMember& thread_,
                                         iType count)
      : start(0), end(count), team(thread_) {}
  inline TeamThreadRangeBoundariesStruct(const OpenACCExecTeamMember& thread_,
                                         iType begin_, iType end_)
      : start(begin_), end(end_), team(thread_) {}
};

template <typename iType>
struct ThreadVectorRangeBoundariesStruct<iType, OpenACCExecTeamMember> {
  using index_type = iType;
  const index_type start;
  const index_type end;
  const OpenACCExecTeamMember& team;

  inline ThreadVectorRangeBoundariesStruct(const OpenACCExecTeamMember& thread_,
                                           index_type count)
      : start(0), end(count), team(thread_) {}
  inline ThreadVectorRangeBoundariesStruct(const OpenACCExecTeamMember& thread_,
                                           index_type begin_, index_type end_)
      : start(begin_), end(end_), team(thread_) {}
};

template <typename iType>
struct TeamVectorRangeBoundariesStruct<iType, OpenACCExecTeamMember> {
  using index_type = iType;
  const index_type start;
  const index_type end;
  const OpenACCExecTeamMember& team;

  inline TeamVectorRangeBoundariesStruct(const OpenACCExecTeamMember& thread_,
                                         index_type count)
      : start(0), end(count), team(thread_) {}
  inline TeamVectorRangeBoundariesStruct(const OpenACCExecTeamMember& thread_,
                                         index_type begin_, index_type end_)
      : start(begin_), end(end_), team(thread_) {}
};

}  // namespace Impl
}  // namespace Kokkos

#undef KOKKOS_IMPL_LOCK_FREE_HIERARCHICAL
#endif /* KOKKOS_OPENACC_PARALLEL_HPP */
