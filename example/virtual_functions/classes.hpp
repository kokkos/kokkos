#include<Kokkos_Core.hpp>


class Foo {
  protected:
    int val;
  public:
    KOKKOS_FUNCTION
    Foo();
    KOKKOS_FUNCTION
    virtual int value() { return 0; }; 
};

class Foo_1: Foo {
  public:
    KOKKOS_FUNCTION
    Foo_1();
    KOKKOS_FUNCTION
    int value();
};

class Foo_2: Foo {
  public:
    KOKKOS_FUNCTION
    Foo_2();
    KOKKOS_FUNCTION
    int value();
};


