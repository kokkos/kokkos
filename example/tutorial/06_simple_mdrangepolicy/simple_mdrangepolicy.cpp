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

#include <Kokkos_Core.hpp>
#include <cstdio>

//
// MDRangePolicy example with parallel_for and parallel_reduce:
//   1. Start up Kokkos
//   2. Execute a parallel_for loop in the default execution space,
//      using a functor to define the loop body
//   3. Shut down Kokkos
//
// Two examples are provided:
// Example 1: Rank 2 case with minimal default parameters and arguments used
//            in the MDRangePolicy
//
// Example 2: Rank 3 case with additional outer/inner iterate pattern parameters
//            and tile dims passed to the ctor

// Simple functor for computing/storing the product of indices in a View v
template <class ViewType>
struct MDFunctor {
  typedef long value_type;

  ViewType v;
  size_t size;

  MDFunctor(const ViewType& v_, const size_t size_) : v(v_), size(size_) {}

  // 2D case - used by parallel_for
  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j) const {
    v(i, j) = i * j;  // compute the product of indices
  }

  // 3D case - used by parallel_for
  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j, const int k) const {
    v(i, j, k) = i * j * k;  // compute the product of indices
  }

  // 2D case - reduction
  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j, value_type& incorrect_count) const {
    if (v(i, j) != i * j) {
      incorrect_count += 1;
    }
  }

  // 3D case - reduction
  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j, const int k,
                  value_type& incorrect_count) const {
    if (v(i, j, k) != i * j * k) {
      incorrect_count += 1;
    }
  }
};

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  // Bound(s) for MDRangePolicy
  const int n = 100;

  // ViewType typedefs for Rank<2>, Rank<3> for example usage
  typedef double ScalarType;
  typedef typename Kokkos::View<ScalarType**> ViewType_2D;
  typedef typename Kokkos::View<ScalarType***> ViewType_3D;

  /////////////////////////////////////////////////////////////////////////////
  // Explanation of MDRangePolicy usage, template parameters, constructor
  // arguments
  //
  // MDRangePolicy typedefs for Rank<2>, Rank<3> cases
  // Required template parameters:
  //   Kokkos::Rank<N>: where N=rank
  //
  // Optional template parameters to Rank<...>:
  //   Kokkos::Iterate::{Default,Left,Right}: Outer iteration pattern across
  //   tiles;
  //     defaults based on the execution space similar to Kokkos::Layout
  //   Kokkos::Iterate::{Default,Left,Right}: Inner iteration pattern within
  //   tiles;
  //     defaults based on the execution space similar to Kokkos::Layout
  //
  //   e.g. typedef Rank<2, Iterate::Left, Iterate::Left> rank2ll;
  //
  //
  // Optional template parameters to MDRangePolicy:
  //   ExecutionSpace: Kokkos::Serial, Kokkos::OpenMP, Kokkos::Cuda, etc.
  //
  //   Kokkos::IndexType< T >: where T = int, long, unsigned int, etc.
  //
  //   struct Tag{}: A user-provided tag for tagging functor operators
  //
  //   e.g. 1:  MDRangePolicy< Kokkos::Serial, Rank<2, Iterate::Left,
  //   Iterate::Left>, IndexType<int>, Tag > mdpolicy; e.g. 2:  MDRangePolicy<
  //   Kokkos::Serial, rank2ll, IndexType<int>, Tag > mdpolicy;
  //
  //
  // Required arguments to ctor:
  //   {{ l0, l1, ... }}: Lower bounds, provided as Kokkos::Array or
  //   std::initializer_list
  //   {{ u0, u1, ... }}: Upper bounds, provided as Kokkos::Array or
  //   std::initializer_list
  //
  // Optional arguments to ctor:
  //   {{ t0, t1, ... }}: Tile dimensions, provided as Kokkos::Array or
  //   std::initializer_list
  //                      defaults based on the execution space
  //
  //  e.g. mdpolicy( {{0,0}}, {{u0,u1}}, {{t0,t1}};
  //
  /////////////////////////////////////////////////////////////////////////////

  // Example 1:
  long incorrect_count_2d = 0;
  {
    // Rank<2> Case: Rank is provided, all other parameters are default
    typedef typename Kokkos::Experimental::MDRangePolicy<
        Kokkos::Experimental::Rank<2> >
        MDPolicyType_2D;

    // Construct 2D MDRangePolicy: lower and upper bounds provided, tile dims
    // defaulted
    MDPolicyType_2D mdpolicy_2d({{0, 0}}, {{n, n}});

    // Construct a 2D view to store result of product of indices
    ViewType_2D v2("v2", n, n);

    // Execute parallel_for with rank 2 MDRangePolicy
    Kokkos::parallel_for("md2d", mdpolicy_2d, MDFunctor<ViewType_2D>(v2, n));

    // Check results with a parallel_reduce using the MDRangePolicy
    Kokkos::parallel_reduce("md2dredux", mdpolicy_2d,
                            MDFunctor<ViewType_2D>(v2, n), incorrect_count_2d);

    printf("Rank 2 MDRangePolicy incorrect count: %ld\n",
           incorrect_count_2d);  // should be 0
  }

  // Example 2:
  long incorrect_count_3d = 0;
  {
    // Rank<3> Case: Rank, inner iterate pattern, outer iterate pattern provided
    typedef typename Kokkos::Experimental::MDRangePolicy<
        Kokkos::Experimental::Rank<3, Kokkos::Experimental::Iterate::Left,
                                   Kokkos::Experimental::Iterate::Left> >
        MDPolicyType_3D;

    // Construct 3D MDRangePolicy: lower, upper bounds, tile dims provided
    MDPolicyType_3D mdpolicy_3d({{0, 0, 0}}, {{n, n, n}}, {{4, 4, 4}});

    // Construct a 3D view to store result of product of indices
    ViewType_3D v3("v3", n, n, n);

    // Execute parallel_for with rank 3 MDRangePolicy
    Kokkos::parallel_for("md3d", mdpolicy_3d, MDFunctor<ViewType_3D>(v3, n));

    // Check results with a parallel_reduce using the MDRangePolicy
    Kokkos::parallel_reduce("md3dredux", mdpolicy_3d,
                            MDFunctor<ViewType_3D>(v3, n), incorrect_count_3d);

    printf("Rank 3 MDRangePolicy incorrect count: %ld\n",
           incorrect_count_3d);  // should be 0
  }

  Kokkos::finalize();

  return (incorrect_count_2d == long(0) && incorrect_count_3d == long(0)) ? 0
                                                                          : -1;
}
