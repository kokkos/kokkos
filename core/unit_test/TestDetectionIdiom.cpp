#include <Kokkos_DetectionIdiom.hpp>

#define STATIC_ASSERT(cond) static_assert(cond, "");

void test_nonesuch() {
  using Kokkos::nonesuch;
  STATIC_ASSERT(!std::is_constructible<nonesuch>::value);
  STATIC_ASSERT(!std::is_destructible<nonesuch>::value);
  STATIC_ASSERT(!std::is_copy_constructible<nonesuch>::value);
  STATIC_ASSERT(!std::is_move_constructible<nonesuch>::value);
#ifdef KOKKOS_ENABLE_CXX17
  STATIC_ASSERT(!std::is_aggregate<nonesuch>::value);
#endif
}

#undef STATIC_ASSERT

namespace Example {
// Example from https://en.cppreference.com/w/cpp/experimental/is_detected
template <class T>
using copy_assign_t = decltype(std::declval<T&>() = std::declval<const T&>());

struct Meow {};
struct Purr {
  void operator=(const Purr&) = delete;
};

static_assert(Kokkos::is_detected<copy_assign_t, Meow>::value,
              "Meow should be copy assignable!");
static_assert(!Kokkos::is_detected<copy_assign_t, Purr>::value,
              "Purr should not be copy assignable!");
static_assert(Kokkos::is_detected_exact<Meow&, copy_assign_t, Meow>::value,
              "Copy assignment of Meow should return Meow&!");

template <class T>
using diff_t = typename T::difference_type;

template <class Ptr>
using difference_type = Kokkos::detected_or_t<std::ptrdiff_t, diff_t, Ptr>;

struct Woof {
  using difference_type = int;
};
struct Bark {};

static_assert(std::is_same<difference_type<Woof>, int>::value,
              "Woof's difference_type should be int!");
static_assert(std::is_same<difference_type<Bark>, std::ptrdiff_t>::value,
              "Bark's difference_type should be ptrdiff_t!");
}  // namespace Example

int main() {}
