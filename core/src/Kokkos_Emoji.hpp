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

#ifndef KOKKOS_EMOJI_HPP
#define KOKKOS_EMOJI_HPP

#include <Kokkos_Core.hpp>

#define 🫘👷 KOKKOS_FUNCTION
#define 🫘🐑 KOKKOS_LAMBDA

namespace 🫘 {
  /**
   * Initialization.
   */
  void 🌅(int& argc, char* argv[]) { Kokkos::initialize(argc, argv); }

  void 🌇() { Kokkos::finalize(); }

  using 🔭💂 = Kokkos::ScopeGuard;

  /**
   * Spaces.
   */
  using 💘🔪🌌 = Kokkos::DefaultExecutionSpace;

  using 💘🏠🔪🌌 = Kokkos::DefaultHostExecutionSpace;

  /**
   * Parallel constructs.
   */
  template <class... 🧹🧬>
  void 🚅🚅(🧹🧬&& ... 🧹) {
    Kokkos::parallel_for(std::forward<🧹🧬>(🧹)...);
  }

  template <class... 🧹🧬>
  void 🚅🩻(🧹🧬&& ... 🧹) {
    Kokkos::parallel_scan(std::forward<🧹🧬>(🧹)...);
  }

  template <class... 🧹🧬>
  void 🚅🌱(🧹🧬&& ... 🧹) {
    Kokkos::parallel_reduce(std::forward<🧹🧬>(🧹)...);
  }

  /**
   * Policies.
   */
  template <class... 🧹🧬>
  using 📏🚔 = Kokkos::RangePolicy<🧹🧬...>;

  template <class... 🧹🧬>
  using 🧊📏🚔 = Kokkos::MDRangePolicy<🧹🧬...>;

  template <class... 🧹🧬>
  using 🤼🚔 = Kokkos::TeamPolicy<🧹🧬...>;

  template <class... 🧹🧬>
  auto 🤼🧶📏(🧹🧬&& ... 🧹)
      ->decltype(Kokkos::TeamThreadRange(std::forward<🧹🧬>(🧹)...)) {
    return Kokkos::TeamThreadRange(std::forward<🧹🧬>(🧹)...);
  }

  template <class... 🧹🧬>
  using 🤼🧶🧊📏 = Kokkos::TeamThreadMDRange<🧹🧬...>;

  template <class... 🧹🧬>
  auto 🧶🧭📏(🧹🧬&& ... 🧹)
      ->decltype(Kokkos::ThreadVectorRange(std::forward<🧹🧬>(🧹)...)) {
    return Kokkos::ThreadVectorRange(std::forward<🧹🧬>(🧹)...);
  }

  template <class... 🧹🧬>
  using 🧶🧭🧊📏 = Kokkos::ThreadVectorMDRange<🧹🧬...>;

  template <class... 🧹🧬>
  auto 🤼🧭📏(🧹🧬&& ... 🧹)
      ->decltype(Kokkos::TeamVectorRange(std::forward<🧹🧬>(🧹)...)) {
    return Kokkos::TeamVectorRange(std::forward<🧹🧬>(🧹)...);
  }

  template <class... 🧹🧬>
  using 🤼🧭🧊📏 = Kokkos::TeamVectorMDRange<🧹🧬...>;

  /**
   * Fences.
   */
  template <class... 🧹🧬>
  void 🚧(🧹🧬&& ... 🧹) {
    Kokkos::fence(std::forward<🧹🧬>(🧹)...);
  }

  /**
   * Views.
   */
  template <class... 🧹🧬>
  using 👁️ = Kokkos::View<🧹🧬...>;

  // XXX missing include
  // template <class... 🧹🧬>
  // using 👁️👁️ = Kokkos::DualView<🧹🧬...>;

  // TODO
  // template <class... 🧹🧬>
  // using 👁️👁️👁️ = Kokkos::TripleView<🧹🧬...>;

  /**
   * Mirrors.
   */
  template <class... 🧹🧬>
  auto 🏗️🪞(🧹🧬&& ... 🧹)
      ->decltype(Kokkos::create_mirror(std::forward<🧹🧬>(🧹)...)) {
    return Kokkos::create_mirror(std::forward<🧹🧬>(🧹)...);
  }

  template <class... 🧹🧬>
  auto 🏗️🪞👁️(🧹🧬&& ... 🧹)
      ->decltype(Kokkos::create_mirror_view(std::forward<🧹🧬>(🧹)...)) {
    return Kokkos::create_mirror_view(std::forward<🧹🧬>(🧹)...);
  }

  template <class... 🧹🧬>
  auto 🏗️🪞👁️👯(🧹🧬&& ... 🧹)
      ->decltype(Kokkos::create_mirror_view_and_copy(
          std::forward<🧹🧬>(🧹)...)) {
    return Kokkos::create_mirror_view_and_copy(std::forward<🧹🧬>(🧹)...);
  }

  /**
   * Data movements.
   */
  template <class... 🧹🧬>
  void 🕳️👯(🧹🧬&& ... 🧹) {
    Kokkos::deep_copy(std::forward<🧹🧬>(🧹)...);
  }

  /**
   * Print on screen.
   */
  template <class... 🧹🧬>
  void 🖨️🛠️(🧹🧬&& ... 🧹) {
    Kokkos::print_configuration(std::forward<🧹🧬>(🧹)...);
  }

  template <class... 🧹🧬>
    🫘👷 void 🖨️📄(🧹🧬&& ... 🧹) {
    Kokkos::printf(std::forward<🧹🧬>(🧹)...);
  }
}  // namespace 🫘

#endif  // ifndef KOKKOS_EMOJI_HPP
