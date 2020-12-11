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
#include <impl/Kokkos_Profiling_Interface.hpp>

// default implementation for view holder specialization
namespace Kokkos {
namespace Impl {
/** \breif   Here we need 2 attorney classes to befriend the private methods
 *           in View which make ViewHooks work properly. Here is what
 *           we need.
 *
 *      ViewHooksCopyAttorney
 *           1. private constructor to avoid calling view hooks recursively
 *      ViewHooksReferenceAttorney
 *           1. private method to expose/update underlying data pointer
 *           2. private method to expose underlying record pointer
 */
template <class ViewType, bool CopyView>
class ViewHooksAttorney;

template <class ViewType>
class ViewHooksAttorney<ViewType, true> {
 public:
  using view_type = ViewType;

 private:
  view_type v;

 public:
  ViewHooksAttorney(const view_type &v_) : v(v_, true) {}
  ViewHooksAttorney(const ViewHooksAttorney &rhs) : v(rhs.v, true) {}
  const view_type &get_view() const { return v; }
};

/* ViewHooksAttorney - update version
 *    this version needs to access the source view, the target map
 *    and the target tracker. This gives a hook the ability to
 *    modify the new view's data handle and tracking record
 */
template <class ViewType>
class ViewHooksAttorney<ViewType, false> {
 public:
  using view_type    = ViewType;
  using map_type     = typename view_type::map_type;
  using track_type   = Kokkos::Impl::SharedAllocationTracker;
  using memory_space = typename view_type::memory_space;

 private:
  view_type &v;
  map_type &m;
  track_type &t;

 public:
  ViewHooksAttorney(view_type &v_, map_type &m_, track_type &t_)
      : v(v_), m(m_), t(t_) {}
  view_type &get_view() const { return v; }

  template <class HandleType>
  void update_data_handle(HandleType ptr) {
    m.m_impl_handle = ptr;
  }
  template <class RecordType>
  void assign_view_record(RecordType rec) {
    t.clear();
    t.assign_allocated_record_to_uninitialized(rec);
  }
  void *rec_ptr() { return (void *)t.template get_record<memory_space>(); }
};

}  // namespace Impl

namespace Experimental {

template <class ViewAttorneyType, class Enable = void>
class ViewHookUpdate {
 public:
  static inline void update_view(ViewAttorneyType &) {}
  static constexpr const char *m_name = "ConstDefault";
};

template <class ViewType, class Enable = void>
class ViewHookCopyView {
 public:
  static inline void copy_view(const ViewType &, const void *) {}
  static inline void copy_view(const void *, const ViewType &) {}
  static constexpr const char *m_name = "ConstDefault";
};

class ViewHolderBase {
 public:
  virtual size_t span() const                                          = 0;
  virtual bool span_is_contiguous() const                              = 0;
  virtual void *data() const                                           = 0;
  virtual std::string label() const noexcept                           = 0;
  virtual Kokkos::Profiling::SpaceHandle space_handle() const noexcept = 0;
  virtual bool is_data_type_const() const noexcept                     = 0;
  virtual ViewHolderBase *clone() const                                = 0;
  virtual size_t data_type_size() const                                = 0;
  virtual bool is_hostspace() const noexcept                           = 0;

  // the following are implemented in specialization classes.
  // View Holder is only a pass through implementation
  // copy view contents to/from host buffer
  virtual void copy_view_to_buffer(unsigned char *buff)   = 0;
  virtual void copy_view_from_buffer(unsigned char *buff) = 0;
  // update view contents
  virtual void update_view() = 0;
  virtual ~ViewHolderBase()  = default;
};

// ViewHolderCopy derives from ViewHolderBase and it
// implement the pure virtual functions above.
// the view data member has to be a copy so that downstream
// containers can catch the result...
template <class ViewType, bool CopyView>
class ViewHolder;

template <class ViewType>
class ViewHolder<ViewType, true> : public ViewHolderBase {
 public:
  using view_type          = ViewType;
  using view_hook_attorney = Kokkos::Impl::ViewHooksAttorney<view_type, true>;
  using memory_space       = typename view_type::memory_space;
  using view_hook_copyview = Kokkos::Experimental::ViewHookCopyView<view_type>;
  // this invokes a special View copy constructor that avoids recursively
  // invoking the ViewHooks, hence the "true" below
  explicit ViewHolder(const view_type &view) : m_view_att(view) {}
  ViewHolder(const ViewHolder &rhs) : m_view_att(rhs.m_view_att) {}
  virtual ~ViewHolder() = default;

  size_t span() const override { return m_view_att.get_view().span(); }
  bool span_is_contiguous() const override {
    return m_view_att.get_view().span_is_contiguous();
  }
  void *data() const override { return (void *)m_view_att.get_view().data(); };

  ViewHolderBase *clone() const override { return new ViewHolder(*this); }

  std::string label() const noexcept override {
    return m_view_att.get_view().label();
  }
  Kokkos::Profiling::SpaceHandle space_handle() const noexcept override {
    return Kokkos::Profiling::make_space_handle(memory_space::name());
  }
  bool is_data_type_const() const noexcept override {
    return (std::is_const<typename view_type::value_type>::value);
  }
  size_t data_type_size() const noexcept override {
    return sizeof(typename view_type::value_type);
  }
  bool is_hostspace() const noexcept override {
    return std::is_same<memory_space, HostSpace>::value;
  }

  void copy_view_to_buffer(unsigned char *buff) override {
    view_hook_copyview::copy_view(buff, m_view_att.get_view());
  }

  void copy_view_from_buffer(unsigned char *buff) override {
    view_hook_copyview::copy_view(m_view_att.get_view(), buff);
  }

  // copy of view cannot perform update...
  void update_view() override {}

 private:
  view_hook_attorney m_view_att;
};

// ViewHolder derives from ViewHolderBase and it
// implement the pure virtual functions above.
// the internal view must be a reference for update_view
// to work
template <class ViewType>
class ViewHolder<ViewType, false> : public ViewHolderBase {
 public:
  using view_type          = ViewType;
  using view_hook_attorney = Kokkos::Impl::ViewHooksAttorney<view_type, false>;
  using map_type           = typename view_hook_attorney::map_type;
  using track_type         = Kokkos::Impl::SharedAllocationTracker;
  using memory_space       = typename view_type::memory_space;
  using view_hook_copyview = Kokkos::Experimental::ViewHookCopyView<view_type>;
  using view_hook_update =
      Kokkos::Experimental::ViewHookUpdate<view_hook_attorney>;
  explicit ViewHolder(view_type &view, map_type &map, track_type &tracker)
      : m_view_att(view, map, tracker) {}
  virtual ~ViewHolder() = default;

  size_t span() const override { return m_view_att.get_view().span(); }
  bool span_is_contiguous() const override {
    return m_view_att.get_view().span_is_contiguous();
  }
  void *data() const override { return (void *)m_view_att.get_view().data(); };

  ViewHolderBase *clone() const override { return new ViewHolder(*this); }

  std::string label() const noexcept override {
    return m_view_att.get_view().label();
  }
  Kokkos::Profiling::SpaceHandle space_handle() const noexcept override {
    return Kokkos::Profiling::make_space_handle(memory_space::name());
  }
  bool is_data_type_const() const noexcept override {
    return (std::is_const<typename view_type::value_type>::value);
  }
  size_t data_type_size() const noexcept override {
    return sizeof(typename view_type::value_type);
  }
  bool is_hostspace() const noexcept override {
    return std::is_same<memory_space, HostSpace>::value;
  }

  void copy_view_to_buffer(unsigned char *buff) override {
    view_hook_copyview::copy_view(buff, m_view_att.get_view());
  }

  void copy_view_from_buffer(unsigned char *buff) override {
    view_hook_copyview::copy_view(m_view_att.get_view(), buff);
  }

  void update_view() override { view_hook_update::update_view(m_view_att); }

 private:
  view_hook_attorney m_view_att;
};

template <typename T>
struct is_view_hook_functor {
 private:
  template <typename, typename = std::true_type>
  struct have : std::false_type {};
  template <typename U>
  struct have<U,
              typename std::is_base_of<typename U::view_hook_functor, U>::type>
      : std::true_type {};

 public:
  static constexpr bool value = is_view_hook_functor::template have<
      typename std::remove_cv<T>::type>::value;
};

struct ViewHookCopyFunctor {};
struct ViewHookRefFunctor {};

struct ViewHookCallerBase {
  virtual void do_call(ViewHolderBase &) = 0;
  virtual ~ViewHookCallerBase();
};

/* ViewHookCaller is the container that holds the hook callback
 * functor. There are two varieties.  On that only handles
 * non-const datatypes and one that handles both const and
 * non-const
 */
template <class Functor, class ConstFunctor = void>
struct ViewHookCaller;

template <class Functor>
struct ViewHookCaller<Functor, void> : public ViewHookCallerBase {
  Functor fun;
  ViewHookCaller(Functor &_f) : fun(_f) {}
  // this is a no-op for the single case
  void do_call(ViewHolderBase &vh) override { fun(vh); }
};

template <class Functor, class ConstFunctor>
struct ViewHookCaller : public ViewHookCallerBase {
  Functor fun;
  ConstFunctor const_fun;
  ViewHookCaller(Functor &_f, ConstFunctor &_constf)
      : fun(_f), const_fun(_constf) {}
  // this is a no-op for the single case
  void do_call(ViewHolderBase &vh) override {
    if (vh.is_data_type_const()) {
      const_fun(vh);
    } else {
      fun(vh);
    }
  }
};

class ViewHooks {
 public:
  using track_type = Kokkos::Impl::SharedAllocationTracker;
  enum HookCallerTypeEnum : int { CopyHookCaller = 0, RefHookCaller };

 private:
  using caller_map_type =
      std::map<std::string, std::unique_ptr<ViewHookCallerBase>>;
  caller_map_type s_callers;
  std::map<std::string, int> s_active_callers;
  std::map<std::string, int> s_deactive_callers;

  bool caller_type_active(HookCallerTypeEnum caller_type) {
    for (auto &enabled_it : s_active_callers) {
      if ((enabled_it.second) == caller_type) {
        return true;
      }
    }
    return false;
  }

 public:
  void register_hook_caller(const std::string &name,
                            std::unique_ptr<ViewHookCallerBase> caller,
                            HookCallerTypeEnum hct = CopyHookCaller) {
    s_callers[name]        = std::move(caller);
    s_active_callers[name] = hct;
  }

  void deactivate_hook_caller(const std::string &name) {
    auto found = s_active_callers.find(name);
    if (found != s_active_callers.end()) {
      s_deactive_callers[name] = found->second;
      s_active_callers.erase(found);
    }
  }

  void reactivate_hook_caller(const std::string &name) {
    auto found = s_deactive_callers.find(name);
    if (found != s_deactive_callers.end()) {
      s_active_callers[name] = found->second;
      s_deactive_callers.erase(found);
    }
  }

  void remove_hook_caller(const std::string &name) {
    s_callers.erase(name);
    s_active_callers.erase(name);
    s_deactive_callers.erase(name);
  }

  static ViewHooks &get_instance();

  bool is_set() noexcept { return s_active_callers.size() > 0; }

  // const view
  template <class DataType, class... Properties>
  void call(const View<DataType, Properties...> &vt) {
    using view_type             = View<DataType, Properties...>;
    using view_copy_holder_type = ViewHolder<const view_type, true>;
    view_copy_holder_type copy_holder(vt);

    for (const auto &caller : s_callers) {
      auto active = s_active_callers.find(caller.first);
      if (active != s_active_callers.end()) {
        if (active->second == CopyHookCaller) {
          caller.second->do_call(copy_holder);
        }
      }
    }
  }

  // overload specialization not allowing data change.
  template <class DataType, class... Properties>
  std::enable_if_t<
      (std::is_same<
           typename Kokkos::ViewTraits<DataType, Properties...>::value_type,
           typename Kokkos::ViewTraits<
               DataType, Properties...>::const_value_type>::value ||
       std::is_same<
           typename Kokkos::ViewTraits<DataType, Properties...>::memory_space,
           Kokkos::AnonymousSpace>::value),
      void>
  call(const View<DataType, Properties...> &vt,
       typename Kokkos::Impl::ViewHooksAttorney<View<DataType, Properties...>,
                                                false>::map_type &,
       const track_type &) {
    call(vt);
  }

  // overload specialization allowing data change
  template <class DataType, class... Properties>
  std::enable_if_t<
      !(std::is_same<
            typename Kokkos::ViewTraits<DataType, Properties...>::value_type,
            typename Kokkos::ViewTraits<
                DataType, Properties...>::const_value_type>::value ||
        std::is_same<
            typename Kokkos::ViewTraits<DataType, Properties...>::memory_space,
            Kokkos::AnonymousSpace>::value),
      void>
  call(const View<DataType, Properties...> &vt,
       typename Kokkos::Impl::ViewHooksAttorney<View<DataType, Properties...>,
                                                false>::map_type &map,
       const track_type &tracker) {
    using view_type             = View<DataType, Properties...>;
    using view_copy_holder_type = ViewHolder<const view_type, true>;
    using view_ref_holder_type  = ViewHolder<view_type, false>;
    view_copy_holder_type copy_holder(vt);
    view_ref_holder_type ref_holder(const_cast<view_type &>(vt), map,
                                    const_cast<track_type &>(tracker));

    for (const auto &caller : s_callers) {
      auto active = s_active_callers.find(caller.first);
      if (active != s_active_callers.end()) {
        if (active->second == CopyHookCaller) {
          caller.second->do_call(copy_holder);
        } else if (active->second == RefHookCaller) {
          caller.second->do_call(ref_holder);
        }
      }
    }
  }
};

template <class Functor>
std::enable_if_t<(std::is_same<typename Functor::view_hook_functor_type,
                               ViewHookRefFunctor>::value),
                 ViewHooks::HookCallerTypeEnum>
get_hook_caller_type() {
  return ViewHooks::RefHookCaller;
}

template <class Functor>
std::enable_if_t<!(std::is_same<typename Functor::view_hook_functor_type,
                                ViewHookRefFunctor>::value),
                 ViewHooks::HookCallerTypeEnum>
get_hook_caller_type() {
  return ViewHooks::CopyHookCaller;
}

// add_view_hook_caller
//    - Three flavors, three parameter which assumes copy caller,
//      two parameter with copy caller and one parameter with ref caller
template <typename Functor, typename ConstFunctor>
void add_view_hook_caller(const std::string &name, Functor &&fun,
                          ConstFunctor &&const_fun) {
  auto caller =
      std::make_unique<ViewHookCaller<Functor, ConstFunctor>>(fun, const_fun);
  ViewHooks::get_instance().register_hook_caller(name, std::move(caller));
}

template <typename Functor>
std::enable_if_t<!is_view_hook_functor<Functor>::value, void>
add_view_hook_caller(const std::string &name, Functor &&fun) {
  auto caller = std::make_unique<ViewHookCaller<Functor, void>>(fun);
  ViewHooks::get_instance().register_hook_caller(name, std::move(caller));
}

template <typename Functor>
std::enable_if_t<is_view_hook_functor<Functor>::value, void>
add_view_hook_caller(const std::string &name, Functor &&fun) {
  auto caller = std::make_unique<ViewHookCaller<Functor, void>>(fun);
  ViewHooks::HookCallerTypeEnum caller_type = get_hook_caller_type<Functor>();
  ViewHooks::get_instance().register_hook_caller(name, std::move(caller),
                                                 caller_type);
}

void deactivate_view_hook_caller(const std::string &name);
void reactivate_view_hook_caller(const std::string &name);
void remove_view_hook_caller(const std::string &name);

}  // namespace Experimental
}  // namespace Kokkos

#endif  // KOKKOS_VIEWHOOKS_HPP
