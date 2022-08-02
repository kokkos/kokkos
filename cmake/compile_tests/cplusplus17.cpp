#include <type_traits>

int main() {
  // _v versions of type traits were added in C++17
  if constexpr (std::is_same_v<double, int>)
    return 0;
  else
    return 1;
}
