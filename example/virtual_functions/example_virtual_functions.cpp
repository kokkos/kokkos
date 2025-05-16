//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <Kokkos_Core.hpp>

class Foo {
 protected:
  int val;

 public:
  KOKKOS_FUNCTION
  Foo() { val = 0; }

  KOKKOS_FUNCTION KOKKOS_VIRTUAL int value() { return 0; };

// FIXME_SYCL Virtual destructors aren't supported yet.
#ifndef KOKKOS_ENABLE_SYCL
  KOKKOS_FUNCTION KOKKOS_VIRTUAL ~Foo() {}
#endif
};

class Foo_1 : public Foo {
 public:
  KOKKOS_FUNCTION
  Foo_1() { val = 1; }

  KOKKOS_FUNCTION KOKKOS_VIRTUAL int value() override { return val + 10; };
};

class Foo_2 : public Foo {
 public:
  KOKKOS_FUNCTION
  Foo_2() { val = 2; }

  KOKKOS_FUNCTION KOKKOS_VIRTUAL int value() override { return val + 20; };
};

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace    = ExecutionSpace::memory_space;

  {
    Foo* f_1 =
        static_cast<Foo*>(Kokkos::kokkos_malloc<MemorySpace>(sizeof(Foo_1)));
    Foo* f_2 =
        static_cast<Foo*>(Kokkos::kokkos_malloc<MemorySpace>(sizeof(Foo_2)));

    Kokkos::parallel_for(
        "CreateObjects", Kokkos::RangePolicy<ExecutionSpace>(0, 1),
        KOKKOS_LAMBDA(const int&) {
          new (f_1) Foo_1();
          new (f_2) Foo_2();
        });
    Kokkos::fence();

    int value_1, value_2;
    Kokkos::parallel_reduce(
        "CheckValues", Kokkos::RangePolicy<ExecutionSpace>(0, 1),
        KOKKOS_LAMBDA(const int&, int& lsum) { lsum = f_1->value(); }, value_1);

    Kokkos::parallel_reduce(
        "CheckValues", Kokkos::RangePolicy<ExecutionSpace>(0, 1),
        KOKKOS_LAMBDA(const int&, int& lsum) { lsum = f_2->value(); }, value_2);
    printf("Values: %i %i\n", value_1, value_2);

// FIXME_SYCL Virtual destructors aren't supported yet.
#ifndef KOKKOS_ENABLE_SYCL
    Kokkos::parallel_for(
        "DestroyObjects", Kokkos::RangePolicy<ExecutionSpace>(0, 1),
        KOKKOS_LAMBDA(const int&) {
          f_1->~Foo();
          f_2->~Foo();
        });
    Kokkos::fence();
#endif

    Kokkos::kokkos_free(f_1);
    Kokkos::kokkos_free(f_2);
  }

  Kokkos::finalize();
}
