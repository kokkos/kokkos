/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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

#ifndef KOKKOS_VIEWHOOKS_HPP
#define KOKKOS_VIEWHOOKS_HPP

#include <Kokkos_Macros.hpp>
#include <Kokkos_HostSpace.hpp>
#include <Kokkos_Layout.hpp>

#include <functional>
#include <memory>
#include <type_traits>

namespace Kokkos {
template <class DataType, class... Properties>
class View;

template <class DataType, class... Properties>
struct ViewTraits;

namespace Impl {
template <class DataType, class... Properties>
View<typename ViewTraits<DataType, Properties...>::non_const_data_type,
     typename std::conditional<
         std::is_same<LayoutLeft, typename ViewTraits<DataType, Properties...>::
                                      array_layout>::value,
         LayoutLeft, LayoutRight>::type,
     HostSpace, MemoryTraits<Unmanaged> >
make_unmanaged_view_like(View<DataType, Properties...> view,
                         unsigned char *buff) {
  using traits_type   = ViewTraits<DataType, Properties...>;
  using new_data_type = typename traits_type::non_const_data_type;

  // Use a contiguous layout type. Keep layout left if it already is, otherwise
  // use layout right
  using layout_type = typename std::conditional<
      std::is_same<LayoutLeft, typename traits_type::array_layout>::value,
      LayoutLeft, LayoutRight>::type;

  using new_view_type =
      View<new_data_type, layout_type, HostSpace, MemoryTraits<Unmanaged> >;

  return new_view_type(
      reinterpret_cast<typename new_view_type::pointer_type>(buff),
      view.rank_dynamic > 0 ? view.extent(0) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      view.rank_dynamic > 1 ? view.extent(1) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      view.rank_dynamic > 2 ? view.extent(2) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      view.rank_dynamic > 3 ? view.extent(3) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      view.rank_dynamic > 4 ? view.extent(4) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      view.rank_dynamic > 5 ? view.extent(5) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      view.rank_dynamic > 6 ? view.extent(6) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      view.rank_dynamic > 7 ? view.extent(7) : KOKKOS_IMPL_CTOR_DEFAULT_ARG);
}
}  // namespace Impl

struct ViewMetadata {
  size_t span;
  bool span_is_contiguous;
  void *data;
  std::string label;
  // virtual ConstViewHolderBase *clone() const = 0;
  size_t data_type_size;
  bool is_hostspace;
};

class ConstViewHolderBase {
 public:
  virtual ~ConstViewHolderBase() = default;

  size_t span() const { return m_metadata.span; }
  bool span_is_contiguous() const { return m_metadata.span_is_contiguous; }
  // const void *data() const ;
  void *data() const { return m_metadata.data; }
  std::string label() const { return m_metadata.label; }

  size_t data_type_size() const { return m_metadata.data_type_size; }
  bool is_hostspace() const noexcept { return m_metadata.is_hostspace; }

  virtual void deep_copy_to_buffer(unsigned char *buff) = 0;

 private:
  ViewMetadata m_metadata;
};

class ViewHolderBase : public ConstViewHolderBase {
 public:
  virtual ~ViewHolderBase() = default;

  // virtual void *data() = 0;
  // virtual ViewHolderBase *clone() const = 0;
  virtual void deep_copy_from_buffer(unsigned char *buff) = 0;
};

template <typename View, typename Enable = void>
class ViewHolder : public ViewHolderBase {
 public:
  virtual ~ViewHolder() = default;

  explicit ViewHolder(const View &view)
      : m_metadata(m_view.span(), m_view.span_is_contiguous(), m_view.data(),
                   m_view.label(), sizeof(typename View::value_type),
                   std::is_same<typename View::memory_space, HostSpace>::value),
        m_view(view) {}

  void deep_copy_to_buffer(unsigned char *buff) override {
    auto unmanaged = Impl::make_unmanaged_view_like(m_view, buff);
    deep_copy(unmanaged, m_view);
  }

  void deep_copy_from_buffer(unsigned char *buff) override {
    auto unmanaged = Impl::make_unmanaged_view_like(m_view, buff);
    deep_copy(m_view, unmanaged);
  }

 private:
  View m_view;
};

template <class View>
class ViewHolder<View, typename std::enable_if<std::is_const<
                           typename View::value_type>::value>::type>
    : public ConstViewHolderBase {
 public:
  virtual ~ViewHolder() = default;

  explicit ViewHolder(const View &view)
      : m_metadata(m_view.span(), m_view.span_is_contiguous(), m_view.data(),
                   m_view.label(), sizeof(typename View::value_type),
                   std::is_same<typename View::memory_space, HostSpace>::value),
        m_view(view) {}

  void deep_copy_to_buffer(unsigned char *buff) override {
    auto unmanaged = Impl::make_unmanaged_view_like(m_view, buff);
    deep_copy(unmanaged, m_view);
  }

 private:
  View m_view;
};

struct ViewHooks {
  using callback_type       = std::function<void(ViewHolderBase &)>;
  using const_callback_type = std::function<void(ConstViewHolderBase &)>;

  template <typename F, typename ConstF>
  static void set(F &&fun, ConstF &&const_fun) {
    s_callback       = std::forward<F>(fun);
    s_const_callback = std::forward<ConstF>(const_fun);
  }

  static void clear() {
    s_callback       = callback_type{};
    s_const_callback = const_callback_type{};
  }

  static bool is_set() noexcept {
    return static_cast<bool>(s_callback) || static_cast<bool>(s_const_callback);
  }

  template <class DataType, class... Properties>
  static void call(View<DataType, Properties...> &view) {
    callback_type tmp_callback;
    const_callback_type tmp_const_callback;

    std::swap(s_callback, tmp_callback);
    std::swap(s_const_callback, tmp_const_callback);

    auto holder = ViewHolder<View<DataType, Properties...> >(view);

    do_call(tmp_callback, tmp_const_callback, std::move(holder));

    std::swap(s_callback, tmp_callback);
    std::swap(s_const_callback, tmp_const_callback);
  }

 private:
  static void do_call(callback_type _cb, const_callback_type _ccb,
                      ViewHolderBase &&view) {
    if (_cb) {
      _cb(view);
    }
  }

  static void do_call(callback_type _cb, const_callback_type _ccb,
                      ConstViewHolderBase &&view) {
    if (_ccb) _ccb(view);
  }

  static callback_type s_callback;
  static const_callback_type s_const_callback;
};

namespace Impl {
template <class ViewType, class Traits = typename ViewType::traits,
          class Enabled = void>
struct ViewHooksCaller {
  static void call(ViewType &view) {}
};

template <class ViewType, class Traits>
struct ViewHooksCaller<
    ViewType, Traits,
    typename std::enable_if<!std::is_same<typename Traits::memory_space,
                                          AnonymousSpace>::value>::type> {
  static void call(ViewType &view) {
    if (ViewHooks::is_set()) ViewHooks::call(view);
  }
};
}  // namespace Impl

}  // namespace Kokkos

#endif  // KOKKOS_VIEWHOOKS_HPP
