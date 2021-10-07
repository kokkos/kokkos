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

#ifndef KOKKOS_VIEWHOLDER_HPP
#define KOKKOS_VIEWHOLDER_HPP

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

class ConstViewHolderImplBase {
 public:
  virtual ~ConstViewHolderImplBase() = default;

  size_t span() const { return m_span; }
  bool span_is_contiguous() const { return m_span_is_contiguous; }
  const void *data() const { return m_data; }
  std::string label() const { return m_label; }

  size_t data_type_size() const { return m_data_type_size; }
  bool is_hostspace() const noexcept { return m_is_hostspace; }

  virtual void deep_copy_to_buffer(unsigned char *buff) = 0;
  virtual ConstViewHolderImplBase *clone() const             = 0;

 protected:

    ConstViewHolderImplBase( std::size_t span, bool span_is_contiguous,
                      const void *data, std::string label, std::size_t data_type_size,
                      bool is_hostspace )
                      : m_span( span ), m_span_is_contiguous( span_is_contiguous ),
                      m_data( data ), m_label( std::move( label ) ),
                      m_data_type_size( data_type_size ), m_is_hostspace( is_hostspace )
                      {}

 private:

  size_t m_span = 0;
  bool m_span_is_contiguous = false;
  const void *m_data = nullptr;
  std::string m_label;
  size_t m_data_type_size = 0;
  bool m_is_hostspace = false;
};

class ViewHolderImplBase {
 public:
  virtual ~ViewHolderImplBase() = default;

  size_t span() const { return m_span; }
  bool span_is_contiguous() const { return m_span_is_contiguous; }
  void *data() const { return m_data; }
  std::string label() const { return m_label; }

  size_t data_type_size() const { return m_data_type_size; }
  bool is_hostspace() const noexcept { return m_is_hostspace; }

  virtual void deep_copy_to_buffer(unsigned char *buff) = 0;
  virtual void deep_copy_from_buffer(unsigned char *buff) = 0;
  virtual ViewHolderImplBase *clone() const                    = 0;

 protected:

  ViewHolderImplBase( std::size_t span, bool span_is_contiguous,
                     void *data, std::string label, std::size_t data_type_size,
                     bool is_hostspace )
      : m_span( span ), m_span_is_contiguous( span_is_contiguous ),
        m_data( data ), m_label( std::move( label ) ),
        m_data_type_size( data_type_size ), m_is_hostspace( is_hostspace )
  {}

 private:

  size_t m_span = 0;
  bool m_span_is_contiguous = false;
  void *m_data = nullptr;
  std::string m_label;
  size_t m_data_type_size = 0;
  bool m_is_hostspace = false;
};

template <typename View, typename Enable = void>
class ViewHolderImpl : public ViewHolderImplBase {
 public:
  virtual ~ViewHolderImpl() = default;

  explicit ViewHolderImpl(const View &view)
      : ViewHolderImplBase(
        view.span(), view.span_is_contiguous(), view.data(),
        view.label(), sizeof(typename View::value_type),
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

  ViewHolderImpl *clone() const override { return new ViewHolderImpl(m_view); }

 private:
  View m_view;
};

template <class View>
class ViewHolderImpl<View, typename std::enable_if<std::is_const<
                               typename View::value_type>::value>::type>
    : public ConstViewHolderImplBase {
 public:
  virtual ~ViewHolderImpl() = default;

  explicit ViewHolderImpl(const View &view)
      : ConstViewHolderImplBase(
          view.span(), view.span_is_contiguous(), view.data(),
          view.label(), sizeof(typename View::value_type),
            std::is_same<typename View::memory_space, HostSpace>::value),
        m_view(view) {}

  void deep_copy_to_buffer(unsigned char *buff) override {
    auto unmanaged = Impl::make_unmanaged_view_like(m_view, buff);
    deep_copy(unmanaged, m_view);
  }

  ViewHolderImpl *clone() const override { return new ViewHolderImpl(m_view); }

 private:
  View m_view;
};
}  // namespace Impl

namespace Experimental {

class ConstViewHolder {
 public:

  ConstViewHolder() = default;

  ConstViewHolder( const ConstViewHolder &other )
  : m_impl( other.m_impl ? other.m_impl->clone() : nullptr )
  {}

  ConstViewHolder( ConstViewHolder &&other ) {
    std::swap(m_impl, other.m_impl);
  }

  ConstViewHolder &operator=( const ConstViewHolder &other )
  {
    m_impl = std::unique_ptr<Impl::ConstViewHolderImplBase>(other.m_impl ? other.m_impl->clone() : nullptr);
    return *this;
  }

  ConstViewHolder &operator=( ConstViewHolder &&other ) {
    std::swap(m_impl, other.m_impl);
    return *this;
  }

  const void *data() const { return m_impl ? m_impl->data() : nullptr; }

 private:

  template< typename V >
  friend auto make_view_holder(const V &view);

  template <typename... Args>
  explicit ConstViewHolder(const View<Args...> &view)
    : m_impl(std::make_unique<Impl::ViewHolderImpl<View<Args...>>>(view)) {
  }

  std::unique_ptr<Impl::ConstViewHolderImplBase> m_impl;
};

class ViewHolder {
 public:

  ViewHolder() = default;

  ViewHolder( const ViewHolder &other )
    : m_impl( other.m_impl ? other.m_impl->clone() : nullptr )
  {}

  ViewHolder( ViewHolder &&other ) {
    std::swap(m_impl, other.m_impl);
  }

  ViewHolder &operator=( const ViewHolder &other )
  {
    m_impl = std::unique_ptr<Impl::ViewHolderImplBase>(other.m_impl ? other.m_impl->clone() : nullptr);
    return *this;
  }

  ViewHolder &operator=( ViewHolder &&other ) {
    std::swap(m_impl, other.m_impl);
    return *this;
  }

  void *data() const { return m_impl ? m_impl->data() : nullptr; }

 private:

  template< typename V >
  friend auto make_view_holder(const V &view);

  template <typename... Args>
  explicit ViewHolder(const View<Args...> &view)
  : m_impl(std::make_unique<Impl::ViewHolderImpl<View<Args...>>>(view)) {

  }

  std::unique_ptr<Impl::ViewHolderImplBase> m_impl;
};

template< typename V >
auto make_view_holder( const V &view ) {
  using holder_type = typename std::conditional<std::is_const<
      typename V::value_type >::value, ConstViewHolder, ViewHolder >::type;
  return holder_type( view );
}


}  // namespace Experimental

}  // namespace Kokkos

#endif  // KOKKOS_VIEWHOLDER_HPP
