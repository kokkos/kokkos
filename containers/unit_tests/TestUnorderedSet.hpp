#ifndef KOKKOS_TEST_UNORDERED_SET_HPP
#define KOKKOS_TEST_UNORDERED_SET_HPP

/**
 * @file
 *
 * The file contains tests that are specific to the specialization of
 * @ref UnorderedMap as a set.
 *
 * Therefore, the number of tests is much smaller than in @ref
 * TestUnorderedMap.hpp, since only a subset of capabilities are semantically
 * different.
 *
 * @note The main difference is that the values view has a zero size for a set.
 */

#include <random>

#include <gtest/gtest.h>

#include <Kokkos_Random.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <Kokkos_Sort.hpp>
#include <Kokkos_UnorderedMap.hpp>

namespace Test {

template <typename map_t>
struct Inserter {
  using policy_type = Kokkos::RangePolicy<TEST_EXECSPACE>;
  using index_type  = typename policy_type::index_type;

  map_t map;

  //! Insert keys from @c 0 to @p max.
  template <typename execution_space, typename T>
  void apply(const execution_space& space, const std::string& label,
             const T& max) {
    Kokkos::parallel_for(
        label, policy_type(space, 0, max),
        KOKKOS_CLASS_LAMBDA(const index_type index) {
          Kokkos::UnorderedMapInsertResult res = map.insert(index);
          if (!res.success()) Kokkos::abort("Cannot insert index.");
        });
  }
};

template <typename map_t>
struct Eraser {
  using policy_type = Kokkos::RangePolicy<TEST_EXECSPACE>;
  using index_type  = typename policy_type::index_type;

  map_t map;

  //! Erase keys from @p keys.
  template <typename execution_space, typename keys_t>
  [[nodiscard]] bool apply(const execution_space& space,
                           const std::string& label, keys_t&& keys) {
    bool current = true;
    current      = current && map.begin_erase(space);
    Kokkos::parallel_for(
        label, policy_type(space, 0, keys.size()),
        KOKKOS_CLASS_LAMBDA(const index_type index) {
          if(!map.erase(keys(index))) Kokkos::abort("Cannot erase index.");
        });
    return current && map.end_erase(space);
  }
};

//! @test Ensure that inserting, erasing and rehasing an unordered set works.
TEST(TEST_CATEGORY, UnorderedSet_insert_erase_and_rehash) {
  /**
   * The set will be inserted for @ref size_all entries,
   * and then @ref size_erased will be erased from it before rehash.
   * Once rehasing is done, the capacity of the set is expected to
   * be smaller than the one of the initial set.
   */
  constexpr size_t size_all    = 2579;
  constexpr size_t size_erased = 568;

  using uset_type = Kokkos::UnorderedMap<size_t, void, TEST_EXECSPACE>;
  static_assert(uset_type::is_set);

  using key_view_type =
      Kokkos::View<typename uset_type::key_type[size_erased], TEST_EXECSPACE>;

  //! Ensure that the space instance is not the default one.
  const auto space =
      Kokkos::Experimental::partition_space(TEST_EXECSPACE{}, 1)[0];

  //! Initialize the set.
  uset_type uset(Kokkos::view_alloc(space, "test uset"), size_all);
  const auto initial_capacity = uset.capacity();

  ASSERT_GE(uset.capacity(), size_all);
  ASSERT_EQ(uset.size(), 0u);

  //! Insert @ref size_all keys.
  Inserter<uset_type>{uset}.apply(
      space, "uset test - insert " + std::to_string(size_all) + " values",
      size_all);
  space.fence();

  ASSERT_GE(uset.capacity(), size_all);
  ASSERT_EQ(uset.size(), size_all);

  /// Always try to erase truly random indices by initializing the random pool
  /// randomly.
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<int> dist(0, std::numeric_limits<int>::max());

  /// Generate random indices between @c 0 and @ref size_all. Those will be
  /// erased.
  Kokkos::Random_XorShift64_Pool<TEST_EXECSPACE> generator(dist(rng));
  key_view_type keys_erased(Kokkos::view_alloc(
      Kokkos::WithoutInitializing, space, "keys that will be erased"));
  Kokkos::fill_random(space, keys_erased, generator, size_all);
  Kokkos::sort(space, keys_erased);

  //! Filter out any duplicated index.
  namespace KE = Kokkos::Experimental;
  const auto keys_erased_end =
      KE::unique(space, KE::begin(keys_erased), KE::end(keys_erased));
  const auto keys_erased_size =
      KE::distance(KE::begin(keys_erased), keys_erased_end);
  const auto expected_uset_size = size_all - keys_erased_size;

  //! Start erasing at random keys.
  Eraser<uset_type> eraser {uset};
  ASSERT_TRUE(eraser.apply(
      space,
      "uset test - erase " + std::to_string(keys_erased_size) + " values",
      Kokkos::subview(keys_erased,
                      Kokkos::make_pair(size_t(0), size_t(keys_erased_size)))));
  space.fence();

  ASSERT_GE(uset.capacity(), size_all);
  ASSERT_EQ(uset.size(), expected_uset_size);

  //! Rehashing should decrease the set capacity.
  uset.rehash(space, expected_uset_size);

  space.fence();

  ASSERT_LE(uset.capacity(), initial_capacity);
  ASSERT_EQ(uset.size(), expected_uset_size);
}

}  // namespace Test

#endif // KOKKOS_TEST_UNORDERED_SET_HPP
