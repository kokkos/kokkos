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

#ifndef KOKKOS_VIEWHOOKS_HPP
#define KOKKOS_VIEWHOOKS_HPP

#include <Kokkos_Core_fwd.hpp>
#include <functional>
#include <memory>
#include <type_traits>
#include <map>
#include <string>

// default implementation for view holder specialization
namespace Kokkos {
namespace Experimental {
namespace Impl {

template <class ViewType, class Enable = void>
class ViewHookDeepCopy;

template <class ViewType, class Enable = void>
class ViewHookUpdate {
 public:
  static inline void update_view(ViewType &, const void *) {}
  static constexpr const char *m_name = "ConstDefault";
};

template <class ViewType, class Enable = void>
class ViewHookCopyView {
 public:
  static inline void copy_view(ViewType &, const void *) {}
  static inline void copy_view(const void *, ViewType &) {}
  static constexpr const char *m_name = "ConstDefault";
};

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

namespace Kokkos {
namespace Experimental {

class ViewHolderBase {
 public:
  virtual size_t span() const                      = 0;
  virtual bool span_is_contiguous() const          = 0;
  virtual void *data() const                       = 0;
  virtual void *rec_ptr() const                    = 0;
  virtual std::string label() const noexcept       = 0;
  virtual bool is_data_type_const() const noexcept = 0;

  virtual ViewHolderBase *clone() const      = 0;
  virtual size_t data_type_size() const      = 0;
  virtual bool is_hostspace() const noexcept = 0;

  // the following are implemented in specialization classes.
  // View Holder is only a pass through implementation
  // deep copy contiguous buffers
  virtual void deep_copy_to_buffer(unsigned char *buff)   = 0;
  virtual void deep_copy_from_buffer(unsigned char *buff) = 0;
  // copy view contents that aren't contiguous
  virtual void copy_view_to_buffer(unsigned char *buff)   = 0;
  virtual void copy_view_from_buffer(unsigned char *buff) = 0;
  // update view contents
  virtual void update_view(const void *) = 0;
  virtual ~ViewHolderBase()              = default;
};

}  // namespace Experimental
}  // namespace Kokkos

namespace Kokkos {
namespace Experimental {

// ViewHolderCopy derives from ViewHolderBase and it
// implement the pure virtual functions above.
// the internal view has to be a copy so that downstream
// containers can catch the result...
template <typename View>
class ViewHolderCopy : public ViewHolderBase {
 public:
  using view_type    = View;
  using memory_space = typename view_type::memory_space;
  using view_hook_deepcopy =
      Kokkos::Experimental::Impl::ViewHookDeepCopy<view_type>;
  using view_hook_copyview =
      Kokkos::Experimental::Impl::ViewHookCopyView<view_type>;
  explicit ViewHolderCopy(const view_type &view) : m_view(view, true) {}
  ViewHolderCopy(const ViewHolderCopy &rhs) : m_view(rhs.m_view, true) {}
  virtual ~ViewHolderCopy() = default;

  size_t span() const override { return m_view.span(); }
  bool span_is_contiguous() const override {
    return m_view.span_is_contiguous();
  }
  void *data() const override { return (void *)m_view.data(); };

  void *rec_ptr() const override {
    return (void *)m_view.impl_track().template get_record<memory_space>();
  }

  ViewHolderCopy *clone() const override { return new ViewHolderCopy(*this); }

  std::string label() const noexcept override { return m_view.label(); }
  virtual bool is_data_type_const() const noexcept override {
    return (std::is_const<typename view_type::value_type>::value);
  }
  size_t data_type_size() const noexcept override {
    return sizeof(typename View::value_type);
  }
  bool is_hostspace() const noexcept override {
    return std::is_same<memory_space, HostSpace>::value;
  }

  void deep_copy_to_buffer(unsigned char *buff) override {
    view_hook_deepcopy::deep_copy(buff, m_view);
  }

  void deep_copy_from_buffer(unsigned char *buff) override {
    view_hook_deepcopy::deep_copy(m_view, buff);
  }

  void copy_view_to_buffer(unsigned char *buff) override {
    view_hook_copyview::copy_view(buff, m_view);
  }

  void copy_view_from_buffer(unsigned char *buff) override {
    view_hook_copyview::copy_view(m_view, buff);
  }

  // copy of view cannot perform update...
  void update_view(const void * /* src_rec */) override {}

 private:
  view_type m_view;
};

// ViewHolderRef derives from ViewHolderBase and it
// implement the pure virtual functions above.
// the internal view must be a reference for update_view
// to work
template <typename View>
class ViewHolderRef : public ViewHolderBase {
 public:
  using view_type    = View;
  using memory_space = typename view_type::memory_space;
  using view_hook_deepcopy =
      Kokkos::Experimental::Impl::ViewHookDeepCopy<view_type>;
  using view_hook_copyview =
      Kokkos::Experimental::Impl::ViewHookCopyView<view_type>;
  using view_hook_update =
      Kokkos::Experimental::Impl::ViewHookUpdate<view_type>;
  explicit ViewHolderRef(view_type &view) : m_view(view) {}
  virtual ~ViewHolderRef() = default;

  size_t span() const override { return m_view.span(); }
  bool span_is_contiguous() const override {
    return m_view.span_is_contiguous();
  }
  void *data() const override { return (void *)m_view.data(); };

  void *rec_ptr() const override {
    return (void *)m_view.impl_track().template get_record<memory_space>();
  }

  ViewHolderRef *clone() const override { return new ViewHolderRef(*this); }

  std::string label() const noexcept override { return m_view.label(); }
  virtual bool is_data_type_const() const noexcept override {
    return (std::is_const<typename view_type::value_type>::value);
  }
  size_t data_type_size() const noexcept override {
    return sizeof(typename View::value_type);
  }
  bool is_hostspace() const noexcept override {
    return std::is_same<memory_space, HostSpace>::value;
  }

  void deep_copy_to_buffer(unsigned char *buff) override {
    view_hook_deepcopy::deep_copy(buff, m_view);
  }

  void deep_copy_from_buffer(unsigned char *buff) override {
    view_hook_deepcopy::deep_copy(m_view, buff);
  }

  void copy_view_to_buffer(unsigned char *buff) override {
    view_hook_copyview::copy_view(buff, m_view);
  }

  void copy_view_from_buffer(unsigned char *buff) override {
    view_hook_copyview::copy_view(m_view, buff);
  }

  void update_view(const void *src_rec) override {
    view_hook_update::update_view(m_view, src_rec);
  }

 private:
  view_type &m_view;
};

struct ViewHookCallerBase {
  virtual bool apply_to_copy_construct()                   = 0;
  virtual void do_call(ViewHolderBase &)                   = 0;
  virtual void do_call(ViewHolderBase &, ViewHolderBase &) = 0;
  virtual ~ViewHookCallerBase();
};

template <class F, class ConstF = void>
struct ViewHookCaller;

template <class F>
struct ViewHookCaller<F, void> : public ViewHookCallerBase {
  F fun;
  inline ViewHookCaller(F &_f) : fun(_f) {}
  inline virtual ~ViewHookCaller() {}
  virtual bool apply_to_copy_construct() { return false; }
  // this is a no-op for the single case
  virtual void do_call(ViewHolderBase &, ViewHolderBase &) {}
  virtual void do_call(ViewHolderBase &vh) { fun(vh); }
};

template <class F, class ConstF>
struct ViewHookCaller : public ViewHookCallerBase {
  F fun;
  ConstF const_fun;
  inline ViewHookCaller(F &_f, ConstF &_constf) : fun(_f), const_fun(_constf) {}
  inline virtual ~ViewHookCaller() {}
  virtual bool apply_to_copy_construct() { return false; }
  // this is a no-op for the single case
  virtual void do_call(ViewHolderBase &, ViewHolderBase &) {}
  virtual void do_call(ViewHolderBase &vh) {
    if (vh.is_data_type_const()) {
      const_fun(vh);
    } else {
      fun(vh);
    }
  }
};

template <class F, class ConstF = void>
struct ViewHookCopyCaller;

template <class F>
struct ViewHookCopyCaller<F, void> : public ViewHookCallerBase {
  F fun;
  inline ViewHookCopyCaller(F &_f) : fun(_f) {}
  inline virtual ~ViewHookCopyCaller() {}
  virtual bool apply_to_copy_construct() { return true; }
  virtual void do_call(ViewHolderBase &dst, ViewHolderBase &src) {
    fun(dst, src);
  }
  // this is a no-op for the double case
  virtual void do_call(ViewHolderBase &) {}
};

template <class F, class ConstF>
struct ViewHookCopyCaller : public ViewHookCallerBase {
  F fun;
  ConstF const_fun;
  inline ViewHookCopyCaller(F &_f, ConstF &_constf)
      : fun(_f), const_fun(_constf) {}
  inline virtual ~ViewHookCopyCaller() {}
  virtual bool apply_to_copy_construct() { return true; }
  virtual void do_call(ViewHolderBase &dst, ViewHolderBase &src) {
    if (dst.is_data_type_const()) {
      const_fun(dst, src);
    } else {
      fun(dst, src);
    }
  }
  // this is a no-op for the double case
  virtual void do_call(ViewHolderBase &) {}
};

struct ViewHooks {
  static std::map<std::string, ViewHookCallerBase *> s_map_callers;

  // Return a new ViewHookCaller struct.
  //    - Because the captured functor (or lambda) cannot be legitimately
  //      cast to a function pointer (whatever is captured requires a container)
  //      and the result is templated on the functor itself, the calling context
  //      must own the object.  we will provide create and clear functions that
  //      the user can call to register the callback functors.
  //      usage:  auto vhc = create_view_hook_caller( [=](ViewHolderBase&){},
  //      [=](ViewHolderBase&){}, true );
  template <typename F, typename ConstF>
  static ViewHookCaller<F, ConstF> *create_view_hook_caller(
      F &&fun, ConstF &&const_fun) {
    return new ViewHookCaller<F, ConstF>(fun, const_fun);
  }

  template <typename F>
  static ViewHookCaller<F, void> *create_view_hook_caller(F &&fun) {
    return new ViewHookCaller<F>(fun);
  }

  // Same as above, but the template class expects the functor to take two
  // parameters.
  template <typename F, typename ConstF>
  static ViewHookCopyCaller<F, ConstF> *create_view_hook_copy_caller(
      F &&fun, ConstF &&const_fun) {
    return new ViewHookCopyCaller<F, ConstF>(fun, const_fun);
  }

  template <typename F>
  static ViewHookCopyCaller<F, void> *create_view_hook_copy_caller(F &&fun) {
    return new ViewHookCopyCaller<F>(fun);
  }

  template <typename VHC>
  static void set(const std::string &name, VHC &&vhc) {
    s_map_callers[name] = vhc;
  }

  template <typename F, typename ConstF>
  static void clear(const std::string &name, ViewHookCaller<F, ConstF> *ptr) {
    if (s_map_callers.find(name) != s_map_callers.end()) {
      s_map_callers.erase(name);
    }
    if (ptr != nullptr) delete ptr;
  }

  template <typename F>
  static void clear(const std::string &name, ViewHookCaller<F, void> *ptr) {
    if (s_map_callers.find(name) != s_map_callers.end()) {
      s_map_callers.erase(name);
    }
    if (ptr != nullptr) delete ptr;
  }

  template <typename F, typename ConstF>
  static void clear(const std::string &name,
                    ViewHookCopyCaller<F, ConstF> *ptr) {
    if (s_map_callers.find(name) != s_map_callers.end()) {
      s_map_callers.erase(name);
    }
    if (ptr != nullptr) delete ptr;
  }

  template <typename F>
  static void clear(const std::string &name, ViewHookCopyCaller<F, void> *ptr) {
    if (s_map_callers.find(name) != s_map_callers.end()) {
      s_map_callers.erase(name);
    }
    if (ptr != nullptr) delete ptr;
  }
  static void clear() { s_map_callers.clear(); }

  static bool is_set() noexcept { return !(s_map_callers.empty()); }

  template <class DataType, class... Properties>
  static void call(const View<DataType, Properties...> &view) {
    using view_holder_type =
        ViewHolderCopy<const View<DataType, Properties...> >;
    view_holder_type holder(view);

    for (const auto &caller : s_map_callers) {
      if (!caller.second->apply_to_copy_construct()) {
        caller.second->do_call(holder);
      }
    }
  }

  template <class DataType, class... Properties>
  static typename std::enable_if<
      (!std::is_same<
           typename Kokkos::ViewTraits<DataType, Properties...>::value_type,
           typename Kokkos::ViewTraits<
               DataType, Properties...>::const_value_type>::value &&
       !std::is_same<
           typename Kokkos::ViewTraits<DataType, Properties...>::memory_space,
           Kokkos::AnonymousSpace>::value),
      void>::type
  call(View<DataType, Properties...> &dst,
       const View<DataType, Properties...> &src) {
    using non_const_view_holder = ViewHolderRef<View<DataType, Properties...> >;
    using const_view_holder =
        ViewHolderCopy<const View<DataType, Properties...> >;
    const_view_holder src_holder(src);
    non_const_view_holder dst_holder(dst);

    for (const auto &caller : s_map_callers) {
      if (caller.second->apply_to_copy_construct()) {
        caller.second->do_call(dst_holder, src_holder);
      } else {
        caller.second->do_call(src_holder);
      }
    }
  }

  // we have to treat anonymous space different...both have to be const...
  template <class DataType, class... Properties>
  static typename std::enable_if<
      (std::is_same<
           typename Kokkos::ViewTraits<DataType, Properties...>::value_type,
           typename Kokkos::ViewTraits<
               DataType, Properties...>::const_value_type>::value ||
       std::is_same<
           typename Kokkos::ViewTraits<DataType, Properties...>::memory_space,
           Kokkos::AnonymousSpace>::value),
      void>::type
  call(View<DataType, Properties...> &dst,
       const View<DataType, Properties...> &src) {
    using view_holder_type =
        ViewHolderCopy<const View<DataType, Properties...> >;
    view_holder_type src_holder(src);
    view_holder_type dst_holder(dst);

    for (const auto &caller : s_map_callers) {
      if (caller.second->apply_to_copy_construct()) {
        caller.second->do_call(dst_holder, src_holder);
      } else {
        caller.second->do_call(src_holder);
      }
    }
  }
};

}  // namespace Experimental
}  // namespace Kokkos

#endif  // KOKKOS_VIEWHOOKS_HPP
