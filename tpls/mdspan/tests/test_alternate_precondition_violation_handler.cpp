#include <stdexcept>

#define MDSPAN_IMPL_PRECONDITION_VIOLATION_HANDLER(cond, file, line) \
  do { \
    throw std::logic_error{"precondition failure"}; \
  } while (0);

#include <mdspan/mdspan.hpp>
#include <gtest/gtest.h>

TEST(mdspan_macros, alternate_precondition_violation_handler)
{
  ASSERT_THROW(MDSPAN_IMPL_PRECONDITION(false), std::logic_error);
}

TEST(mdspan_macros, alternate_precondition_check_constexpr_invocable)
{
  struct fn
  {
    constexpr auto operator()() const
    {
      MDSPAN_IMPL_PRECONDITION(1 + 1 == 2);
      return 42;
    }
  };

  static constexpr auto x = fn{}();
  (void)x;
}
