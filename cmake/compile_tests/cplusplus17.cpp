#include <type_traits>

int main() {
  // _v versions of type traits were added in C++17
  constexpr bool same = std::is_same_v<double, int>;

  return same;
}
