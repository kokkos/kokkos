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
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
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

#ifndef KOKKOS_VIEWHOLDER_HPP
#define KOKKOS_VIEWHOLDER_HPP

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_HostSpace.hpp>

#include <memory>
#include <type_traits>
#include <string>

namespace Kokkos {
// Forward declaration from View
// This needs to be here to avoid a circular dependency; it's
// necessary to see if the view holder can be assignable to a CPU buffer
// for the purposes of type erasure
template <class T1, class T2>
struct is_always_assignable_impl;

namespace Experimental {
namespace Impl {

template <class DataType, class... Properties>
using unmanaged_view_type_like =
    View<typename View<DataType, Properties...>::traits::non_const_data_type,
         typename View<DataType, Properties...>::traits::array_layout,
         HostSpace, MemoryTraits<Unmanaged>>;

template <class DataType, class... Properties>
unmanaged_view_type_like<DataType, Properties...> make_unmanaged_view_like(
    View<DataType, Properties...> view, unsigned char *buff) {
  static_assert(
      !std::is_same<
          typename View<DataType, Properties...>::traits::memory_space,
          AnonymousSpace>::value,
      "make_unmanaged_view_like can't create an anonymous space view");

  using new_view_type = unmanaged_view_type_like<DataType, Properties...>;

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

class ConstViewHolderImplBase {
 public:
  virtual ~ConstViewHolderImplBase() = default;

  size_t span() const { return m_span; }
  bool span_is_contiguous() const { return m_span_is_contiguous; }
  const void *data() const { return m_data; }
  std::string label() const { return m_label; }

  size_t data_type_size() const { return m_data_type_size; }
  bool is_host_space() const noexcept { return m_is_host_space; }

  virtual void deep_copy_to_buffer(unsigned char *buff) = 0;
  virtual ConstViewHolderImplBase *clone() const        = 0;

 protected:
  ConstViewHolderImplBase(std::size_t span, bool span_is_contiguous,
                          const void *data, std::string label,
                          std::size_t data_type_size, bool is_host_space)
      : m_span(span),
        m_span_is_contiguous(span_is_contiguous),
        m_data(data),
        m_label(std::move(label)),
        m_data_type_size(data_type_size),
        m_is_host_space(is_host_space) {}

 private:
  size_t m_span             = 0;
  bool m_span_is_contiguous = false;
  const void *m_data        = nullptr;
  std::string m_label;
  size_t m_data_type_size = 0;
  bool m_is_host_space     = false;
};

class ViewHolderImplBase {
 public:
  virtual ~ViewHolderImplBase() = default;

  size_t span() const { return m_span; }
  bool span_is_contiguous() const { return m_span_is_contiguous; }
  void *data() const { return m_data; }
  std::string label() const { return m_label; }

  size_t data_type_size() const { return m_data_type_size; }
  bool is_host_space() const noexcept { return m_is_host_space; }

  virtual void deep_copy_to_buffer(unsigned char *buff)   = 0;
  virtual void deep_copy_from_buffer(unsigned char *buff) = 0;
  virtual ViewHolderImplBase *clone() const               = 0;

 protected:
  ViewHolderImplBase(std::size_t span, bool span_is_contiguous, void *data,
                     std::string label, std::size_t data_type_size,
                     bool is_host_space)
      : m_span(span),
        m_span_is_contiguous(span_is_contiguous),
        m_data(data),
        m_label(std::move(label)),
        m_data_type_size(data_type_size),
        m_is_host_space(is_host_space) {}

 private:
  size_t m_span             = 0;
  bool m_span_is_contiguous = false;
  void *m_data              = nullptr;
  std::string m_label;
  size_t m_data_type_size = 0;
  bool m_is_host_space     = false;
};

template <typename SrcViewType, typename DstViewType, typename Enabled = void>
struct ViewHolderImplDeepCopyImpl {
  static void copy_to_unmanaged(SrcViewType &, void *) {
    Kokkos::Impl::throw_runtime_exception(
        "Cannot deep copy a view holder to an incompatible view");
  }

  static void copy_from_unmanaged(DstViewType &, const void *) {
    Kokkos::Impl::throw_runtime_exception(
        "Cannot deep copy from a host unmanaged view holder to an incompatible "
        "view");
  }
};

template <typename SrcViewType, typename DstViewType>
struct ViewHolderImplDeepCopyImpl<
    SrcViewType, DstViewType,
    std::enable_if_t<is_always_assignable_impl<
        typename std::remove_reference<DstViewType>::type,
        typename std::remove_const<
            typename std::remove_reference<SrcViewType>::type>::type>::value>> {
  static void copy_to_unmanaged(SrcViewType &_src, void *_buff) {
    auto dst = make_unmanaged_view_like(_src, _buff);
    deep_copy(dst, _src);
  }

  static void copy_from_unmanaged(DstViewType &_dst, const void *_buff) {
    auto src = Impl::make_unmanaged_view_like(_dst, _buff);
    deep_copy(_dst, src);
  }
};

template <typename View, typename Enable = void>
class ViewHolderImpl : public ViewHolderImplBase {
  static_assert(
      !std::is_same<typename View::traits::memory_space, AnonymousSpace>::value,
      "ViewHolder can't hold anonymous space view");

 public:
  virtual ~ViewHolderImpl() = default;

  explicit ViewHolderImpl(const View &view)
      : ViewHolderImplBase(
            view.span(), view.span_is_contiguous(), view.data(), view.label(),
            sizeof(typename View::value_type),
            std::is_same<typename View::memory_space, HostSpace>::value),
        m_view(view) {}

  void deep_copy_to_buffer(unsigned char *buff) override {
    using dst_type = unmanaged_view_type_like<View>;
    ViewHolderImplDeepCopyImpl<View, dst_type>::copy_to_unmanaged(m_view, buff);
  }

  void deep_copy_from_buffer(unsigned char *buff) override {
    using src_type = unmanaged_view_type_like<View>;
    ViewHolderImplDeepCopyImpl<src_type, View>::copy_from_unmanaged(m_view,
                                                                    buff);
  }

  ViewHolderImpl *clone() const override { return new ViewHolderImpl(m_view); }

 private:
  View m_view;
};

template <class View>
class ViewHolderImpl<View, typename std::enable_if<std::is_const<
                               typename View::value_type>::value>::type>
    : public ConstViewHolderImplBase {
  static_assert(
      !std::is_same<typename View::traits::memory_space, AnonymousSpace>::value,
      "ViewHolder can't hold anonymous space view");

 public:
  virtual ~ViewHolderImpl() = default;

  explicit ViewHolderImpl(const View &view)
      : ConstViewHolderImplBase(
            view.span(), view.span_is_contiguous(), view.data(), view.label(),
            sizeof(typename View::value_type),
            std::is_same<typename View::memory_space, HostSpace>::value),
        m_view(view) {}

  void deep_copy_to_buffer(unsigned char *buff) override {
    using dst_type = unmanaged_view_type_like<View>;
    ViewHolderImplDeepCopyImpl<View, dst_type>::copy_to_unmanaged(m_view, buff);
  }

  ViewHolderImpl *clone() const override { return new ViewHolderImpl(m_view); }

 private:
  View m_view;
};
}  // namespace Impl

class ConstViewHolder {
 public:
  ConstViewHolder() = default;

  ConstViewHolder(const ConstViewHolder &other)
      : m_impl(other.m_impl ? other.m_impl->clone() : nullptr) {}

  ConstViewHolder(ConstViewHolder &&other) { std::swap(m_impl, other.m_impl); }

  ConstViewHolder &operator=(const ConstViewHolder &other) {
    m_impl = std::unique_ptr<Impl::ConstViewHolderImplBase>(
        other.m_impl ? other.m_impl->clone() : nullptr);
    return *this;
  }

  ConstViewHolder &operator=(ConstViewHolder &&other) {
    std::swap(m_impl, other.m_impl);
    return *this;
  }

  const void *data() const { return m_impl ? m_impl->data() : nullptr; }

 private:
  template <typename V>
  friend auto make_view_holder(const V &view);

  template <typename... Args>
  explicit ConstViewHolder(const View<Args...> &view)
      : m_impl(std::make_unique<Impl::ViewHolderImpl<View<Args...>>>(view)) {}

  std::unique_ptr<Impl::ConstViewHolderImplBase> m_impl;
};

class ViewHolder {
 public:
  ViewHolder() = default;

  ViewHolder(const ViewHolder &other)
      : m_impl(other.m_impl ? other.m_impl->clone() : nullptr) {}

  ViewHolder(ViewHolder &&other) { std::swap(m_impl, other.m_impl); }

  ViewHolder &operator=(const ViewHolder &other) {
    m_impl = std::unique_ptr<Impl::ViewHolderImplBase>(
        other.m_impl ? other.m_impl->clone() : nullptr);
    return *this;
  }

  ViewHolder &operator=(ViewHolder &&other) {
    std::swap(m_impl, other.m_impl);
    return *this;
  }

  void *data() const { return m_impl ? m_impl->data() : nullptr; }

 private:
  template <typename V>
  friend auto make_view_holder(const V &view);

  template <typename... Args>
  explicit ViewHolder(const View<Args...> &view)
      : m_impl(std::make_unique<Impl::ViewHolderImpl<View<Args...>>>(view)) {}

  std::unique_ptr<Impl::ViewHolderImplBase> m_impl;
};

template <typename V>
auto make_view_holder(const V &view) {
  using holder_type =
      typename std::conditional<std::is_const<typename V::value_type>::value,
                                ConstViewHolder, ViewHolder>::type;
  return holder_type(view);
}

}  // namespace Experimental

}  // namespace Kokkos

#endif  // KOKKOS_VIEWHOLDER_HPP
